"""
Matrix Recall Benchmark Runner.

Tests a model's ability to locate and return specific values from a 10×10
random integer matrix based on natural-language lookup questions.
Pure recall — no computation required.
"""

import ast
import json
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from ....agents.base_agent import BaseAgent, BenchmarkResponse, EvaluationResult
from ....evaluation.cost_tracker import CostTracker
from ....utils.adaptive_limiter import ThreadSafeAdaptiveLimiter
from ....utils.trace import TraceCapture, QuestionTrace

from .generator import MatrixRecallTaskGenerator


class MatrixRecallRunner:
    """
    Runner for Matrix Recall evaluation.

    Generates 10×10 matrices, asks natural-language lookup questions at
    three difficulty tiers (easy / medium / hard), and scores exact-match
    accuracy to isolate *recall* from other cognitive skills.
    """

    def __init__(
        self,
        agent: BaseAgent,
        cost_tracker: Optional[CostTracker] = None,
        verbose: bool = False,
        concurrency: int = 1,
        run_dir: Optional[Path] = None,
        num_tasks: int = 200,
        matrix_size: int = 10,
        difficulty_distribution: Optional[Dict[str, float]] = None,
    ):
        self.agent = agent
        self.cost_tracker = cost_tracker
        self.verbose = verbose
        self.concurrency = concurrency
        self.run_dir = run_dir
        self.num_tasks = num_tasks
        self.matrix_size = matrix_size
        self.difficulty_distribution = difficulty_distribution  # None = use generator default

        self._lock = Lock()
        self._correct_count = 0
        self._total_count = 0
        self._output_dir: Optional[Path] = None
        self._difficulty_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"correct": 0, "total": 0}
        )

        self.task_generator = MatrixRecallTaskGenerator(
            seed=42, matrix_size=matrix_size
        )

    # ── task loading ────────────────────────────────────────────────

    def load_tasks(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        num = limit or self.num_tasks
        if self.verbose:
            print(f"Generating {num} matrix recall tasks ({self.matrix_size}×{self.matrix_size})...")
        tasks = self.task_generator.generate(
            num_tasks=num,
            difficulty_distribution=self.difficulty_distribution,
        )
        dist = defaultdict(int)
        for t in tasks:
            dist[t["difficulty"]] += 1
        print(f"Difficulty distribution: {dict(dist)}")
        return tasks

    # ── prompt formatting ───────────────────────────────────────────

    def format_task(self, task: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        answer_hint = {
            "single": 'Return a single integer: {"answer": <int>, "confidence": 0.0-1.0}',
            "list":   'Return a JSON list of integers: {"answer": [<int>, ...], "confidence": 0.0-1.0}',
            "matrix": 'Return a 2-D list: {"answer": [[...], ...], "confidence": 0.0-1.0}',
        }[task["answer_type"]]

        ms = task.get("matrix_size", self.matrix_size)
        task_text = f"""Here is a {ms}x{ms} matrix of integers:

{task["matrix_text"]}

QUESTION:
{task["question"]}

INSTRUCTIONS:
- Carefully locate the requested value(s) in the matrix above.
- {answer_hint}
- Output ONLY the JSON object, no explanation or markdown."""

        context = {
            "benchmark_type": "matrix_recall",
            "task_type": "matrix_recall",
            "difficulty": task["difficulty"],
            "answer_type": task["answer_type"],
        }
        return task_text, context

    # ── response parsing ────────────────────────────────────────────

    def _parse_response(self, response_text: str, answer_type: str) -> Any:
        """Extract the answer payload from the agent's response."""
        if not response_text or not response_text.strip():
            return None

        text = response_text.strip()

        # Strip markdown code fences
        code_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if code_match:
            text = code_match.group(1).strip()

        # Strategy 1: JSON parse
        parsed = self._try_json(text)
        if parsed is not None:
            return parsed

        # Strategy 2: ast.literal_eval (handles Python dicts with single quotes)
        try:
            obj = ast.literal_eval(text)
            return self._extract_answer(obj, answer_type)
        except (ValueError, SyntaxError):
            pass

        # Strategy 3: find JSON-like substring
        for starter in ["{", "["]:
            idx = text.find(starter)
            if idx != -1:
                candidate = text[idx:]
                parsed = self._try_json(candidate)
                if parsed is not None:
                    return parsed

        # Strategy 4: bare integer for single-value questions
        if answer_type == "single":
            m = re.search(r"-?\d+", text)
            if m:
                return int(m.group(0))

        return None

    def _try_json(self, text: str) -> Any:
        try:
            obj = json.loads(text)
            return self._extract_answer(obj, answer_type=None)
        except (json.JSONDecodeError, ValueError):
            return None

    def _extract_answer(self, obj: Any, answer_type: Optional[str] = None) -> Any:
        if isinstance(obj, dict):
            for key in ("answer", "response", "result", "value"):
                if key in obj:
                    val = obj[key]
                    if isinstance(val, str):
                        try:
                            return json.loads(val)
                        except (json.JSONDecodeError, ValueError):
                            try:
                                return ast.literal_eval(val)
                            except (ValueError, SyntaxError):
                                pass
                    return val
        if isinstance(obj, (int, float)):
            return int(obj) if isinstance(obj, float) and obj == int(obj) else obj
        if isinstance(obj, (list, tuple)):
            return self._normalise_list(obj)
        return obj

    @staticmethod
    def _normalise_list(lst: Any) -> Any:
        """Recursively convert tuples→lists and floats→ints where lossless."""
        if isinstance(lst, (list, tuple)):
            return [MatrixRecallRunner._normalise_list(x) for x in lst]
        if isinstance(lst, float) and lst == int(lst):
            return int(lst)
        return lst

    # ── scoring ─────────────────────────────────────────────────────

    @staticmethod
    def check_answer(predicted: Any, expected: Any) -> bool:
        """Exact-match comparison after normalisation."""
        predicted = MatrixRecallRunner._normalise_list(predicted)
        expected = MatrixRecallRunner._normalise_list(expected)
        return predicted == expected

    # ── single-task processing ──────────────────────────────────────

    def _process_task(self, task: Dict[str, Any]) -> EvaluationResult:
        start_time = time.time()
        task_id = task["task_id"]

        task_text, context = self.format_task(task)

        with TraceCapture(
            task_id=task_id,
            agent_type=self.agent.__class__.__name__,
            model=self.agent.model,
            input_question=task["question"],
        ) as trace_ctx:
            response = self.agent.respond_to_task(task_text, context)
            latency = time.time() - start_time

            predicted = self._parse_response(response.response, task["answer_type"])
            is_correct = (
                self.check_answer(predicted, task["answer"])
                if predicted is not None
                else False
            )

            cost = 0.0
            meta = response.metadata if isinstance(response.metadata, dict) else {}
            if self.cost_tracker and meta:
                pt = meta.get("prompt_tokens", 0)
                ct = meta.get("completion_tokens", 0)
                if pt > 0:
                    with self._lock:
                        cost = self.cost_tracker.log_usage(
                            model=self.agent.model,
                            prompt_tokens=pt,
                            completion_tokens=ct,
                            task_id=task_id,
                            benchmark="matrix_recall",
                            agent_type=self.agent.__class__.__name__,
                        )

            trace_ctx.trace.final_output = (response.response or "")[:200]
            trace_ctx.trace.predicted = str(predicted)[:100]
            trace_ctx.trace.correct = str(task["answer"])[:100]
            trace_ctx.trace.match = is_correct
            trace_ctx.trace.confidence = response.confidence
            trace_ctx.trace.total_latency = latency
            trace_ctx.trace.total_cost = cost
            trace_ctx.trace.reasoning = response.reasoning or ""

            with self._lock:
                diff = task["difficulty"]
                self._total_count += 1
                if is_correct:
                    self._correct_count += 1
                self._difficulty_stats[diff]["total"] += 1
                if is_correct:
                    self._difficulty_stats[diff]["correct"] += 1

        return EvaluationResult(
            task_id=task_id,
            prompt=task["question"],
            agent_response=response.response,
            success=is_correct,
            score=1.0 if is_correct else 0.0,
            latency=latency,
            cost=cost,
            metadata={
                "difficulty": task["difficulty"],
                "answer_type": task["answer_type"],
                "expected": task["answer"],
                "predicted": predicted,
                "coordinates": task["coordinates"],
                "trace": trace_ctx.trace,
            },
        )

    # ── incremental save ────────────────────────────────────────────

    def _save_result_incremental(self, result: EvaluationResult):
        if not self._output_dir:
            return
        with self._lock:
            metadata = result.metadata if isinstance(result.metadata, dict) else {}
            task_dir = self._output_dir / result.task_id
            task_dir.mkdir(parents=True, exist_ok=True)

            trace = metadata.get("trace")
            if trace and isinstance(trace, QuestionTrace):
                trace_data = trace.to_dict()
            else:
                trace_data = {
                    "task_id": result.task_id,
                    "difficulty": metadata.get("difficulty"),
                    "answer_type": metadata.get("answer_type"),
                    "expected": metadata.get("expected"),
                    "predicted": metadata.get("predicted"),
                    "match": result.success,
                    "latency": round(result.latency, 2),
                    "cost": round(result.cost, 6),
                }
            with open(task_dir / "trace.json", "w") as f:
                json.dump(trace_data, f, indent=2, default=str)

            with open(self._output_dir / "results.jsonl", "a") as f:
                f.write(
                    json.dumps({
                        "task_id": result.task_id,
                        "difficulty": metadata.get("difficulty"),
                        "match": result.success,
                        "latency": round(result.latency, 2),
                        "cost": round(result.cost, 6),
                    })
                    + "\n"
                )

    # ── execution strategies ────────────────────────────────────────

    def _run_sequential(self, tasks: List[Dict[str, Any]]) -> List[EvaluationResult]:
        results: List[EvaluationResult] = []
        pbar = tqdm(
            tasks,
            desc="Running",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] Acc: {postfix}",
            postfix="---%",
        )
        for task in pbar:
            result = self._process_task(task)
            results.append(result)
            self._save_result_incremental(result)
            acc = (self._correct_count / self._total_count * 100) if self._total_count else 0
            pbar.set_postfix_str(f"{acc:.1f}%")
            if self.verbose:
                status = "PASS" if result.success else "FAIL"
                tqdm.write(f"  {result.task_id} [{task['difficulty']}] {status} ({result.latency:.1f}s)")
        return results

    def _run_concurrent(self, tasks: List[Dict[str, Any]]) -> List[EvaluationResult]:
        results: List[Optional[EvaluationResult]] = [None] * len(tasks)
        limiter = ThreadSafeAdaptiveLimiter(
            max_concurrency=self.concurrency,
            min_concurrency=1,
            backoff_factor=0.5,
            recovery_threshold=10,
        )

        def _process(task):
            wait = limiter.wait_if_needed()
            if wait > 0 and self.verbose:
                tqdm.write(f"  Rate-limit backoff: {wait:.1f}s")
            try:
                result = self._process_task(task)
                limiter.record_success()
                return result
            except Exception as e:
                if "rate" in str(e).lower() or "429" in str(e):
                    limiter.record_error(backoff_seconds=5.0)
                raise

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            future_map = {pool.submit(_process, t): i for i, t in enumerate(tasks)}
            pbar = tqdm(
                total=len(tasks),
                desc=f"Running ({self.concurrency}x)",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] Acc: {postfix}",
                postfix="---%",
            )
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    result = future.result()
                    results[idx] = result
                    self._save_result_incremental(result)
                    with self._lock:
                        acc = (self._correct_count / self._total_count * 100) if self._total_count else 0
                    pbar.set_postfix_str(f"{acc:.1f}%")
                except Exception as e:
                    results[idx] = EvaluationResult(
                        task_id=tasks[idx]["task_id"],
                        prompt="",
                        agent_response=f"Error: {e}",
                        success=False,
                        score=0.0,
                        latency=0.0,
                        cost=0.0,
                        metadata={"error": str(e)},
                    )
                pbar.update(1)
            pbar.close()
        return results  # type: ignore[return-value]

    # ── public entry point ──────────────────────────────────────────

    def run(
        self,
        limit: Optional[int] = None,
        save_results: bool = True,
    ) -> List[EvaluationResult]:
        tasks = self.load_tasks(limit)

        self._correct_count = 0
        self._total_count = 0
        self._difficulty_stats.clear()

        print(f"\nRunning Matrix Recall benchmark with {len(tasks)} tasks...")
        print(f"Agent: {self.agent.__class__.__name__}")
        print(f"Model: {self.agent.model}")
        print(f"Concurrency: {self.concurrency}")

        if save_results:
            if self.run_dir:
                base_dir = self.run_dir
            else:
                ts = time.strftime("%Y%m%d_%H%M%S")
                base_dir = Path("results") / "matrix_recall" / f"{self.agent.model}_{ts}"
            self._output_dir = base_dir / self.agent.__class__.__name__
            self._output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving to: {self._output_dir}/\n")

        if self.concurrency > 1:
            results = self._run_concurrent(tasks)
        else:
            results = self._run_sequential(tasks)

        accuracy = self._correct_count / self._total_count if self._total_count else 0.0
        print(f"\nFinal Accuracy: {accuracy * 100:.1f}% ({self._correct_count}/{self._total_count})")
        self._print_breakdown()

        if save_results:
            self._save_summary(results, accuracy)

        return results

    # ── reporting helpers ───────────────────────────────────────────

    def _print_breakdown(self):
        if not self._difficulty_stats:
            return
        print(f"\n{'=' * 70}")
        print("BREAKDOWN BY DIFFICULTY")
        print(f"{'=' * 70}")
        for diff in ("easy", "medium", "hard", "x-hard"):
            s = self._difficulty_stats.get(diff, {"correct": 0, "total": 0})
            if s["total"] > 0:
                acc = s["correct"] / s["total"] * 100
                print(f"  {diff.capitalize():10s}: {acc:5.1f}% ({s['correct']}/{s['total']})")

    def _save_summary(self, results: List[EvaluationResult], accuracy: float):
        if not self._output_dir:
            return
        total_latency = sum(r.latency for r in results if r.latency)
        total_cost = sum(r.cost for r in results if r.cost)
        diff_stats = {
            d: {
                "accuracy": s["correct"] / s["total"] if s["total"] else 0.0,
                "correct": s["correct"],
                "total": s["total"],
            }
            for d, s in self._difficulty_stats.items()
        }
        data = {
            "agent": self.agent.__class__.__name__,
            "model": self.agent.model,
            "benchmark": "Matrix Recall",
            "matrix_size": self.matrix_size,
            "accuracy": accuracy,
            "num_tasks": len(results),
            "total_latency_seconds": total_latency,
            "avg_latency_seconds": total_latency / len(results) if results else 0,
            "total_cost_usd": total_cost,
            "results_dir": str(self._output_dir),
            "breakdown_by_difficulty": diff_stats,
        }
        summary_file = self._output_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nSummary saved to: {summary_file}")
