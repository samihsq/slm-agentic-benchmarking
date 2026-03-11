"""
BIG-bench Lite Benchmark Runner.

Loads tasks from tasksource/bigbench on HuggingFace, formats them as
multiple-choice or free-text prompts, and scores with the appropriate
method per task (MC letter-match, exact string, or BLEU).

Default suite is the official 24-task BIG-bench Lite (BBL24).
Undersized tasks (fewer than 20 train examples) automatically combine
train + validation splits. Per-task accuracies are weighted equally when
computing the suite-level score.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from ....agents.base_agent import BaseAgent, BenchmarkResponse, EvaluationResult
from ....evaluation.cost_tracker import CostTracker
from ....utils.adaptive_limiter import ThreadSafeAdaptiveLimiter
from ....utils.trace import TraceCapture, QuestionTrace
from .task_sets import BIGBENCH_LITE_24, BIGBENCH_CORE_6, BIGBENCH_LITE_BY_NAME

# Default task config names (legacy 6-task set) for backwards compat.
DEFAULT_TASK_CONFIGS = BIGBENCH_CORE_6


class BigBenchRunner:
    """
    Runner for BIG-bench evaluation.

    Loads tasks from HuggingFace tasksource/bigbench, presents them as
    multiple-choice or free-text questions, and scores with the method
    appropriate for each task (mc / exact / bleu).

    For the BBL24 suite:
    - Standard tasks: up to `examples_per_task` (default 20) from train split.
    - Undersized tasks: all available from train + validation, no cap.
    - Final suite score: equal-weighted mean of per-task accuracy.
    """

    def __init__(
        self,
        agent: BaseAgent,
        cost_tracker: Optional[CostTracker] = None,
        verbose: bool = False,
        concurrency: int = 1,
        run_dir: Optional[Path] = None,
        task_configs: Optional[List[str]] = None,
        examples_per_task: Optional[int] = 20,
        suite: str = "bbl24",
        backend: str = "azure",
        architecture: str = "one_shot",
    ):
        self.agent = agent
        self.cost_tracker = cost_tracker
        self.verbose = verbose
        self.concurrency = concurrency
        self.run_dir = run_dir
        self.examples_per_task = examples_per_task
        self.suite = suite
        self.backend = backend
        self.architecture = architecture

        # Resolve task list from suite name or explicit list.
        if task_configs is not None:
            self.task_configs = task_configs
            self._task_meta: Dict[str, Dict[str, Any]] = {
                name: BIGBENCH_LITE_BY_NAME.get(name, {"name": name, "scoring": "mc", "undersized": False})
                for name in task_configs
            }
        elif suite == "bbl24":
            self.task_configs = [t["name"] for t in BIGBENCH_LITE_24]
            self._task_meta = {t["name"]: t for t in BIGBENCH_LITE_24}
        else:
            self.task_configs = list(BIGBENCH_CORE_6)
            self._task_meta = {}

        self._lock = Lock()
        self._correct_count = 0
        self._total_count = 0
        self._output_dir: Optional[Path] = None
        self._task_stats: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_tasks(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        try:
            from datasets import load_dataset  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "BigBench runner requires HuggingFace 'datasets'. Install with: pip install datasets"
            ) from e

        tasks: List[Dict[str, Any]] = []

        for config_name in self.task_configs:
            meta = self._task_meta.get(config_name, {"scoring": "mc", "undersized": False})
            scoring = meta.get("scoring", "mc")
            undersized = meta.get("undersized", False)

            rows = self._load_rows(load_dataset, config_name, undersized)

            # Apply per-task cap (skip cap for undersized tasks – use all available).
            if undersized:
                n_take = len(rows)
            elif limit is not None:
                n_take = min(limit // max(len(self.task_configs), 1), len(rows))
            elif self.examples_per_task is not None:
                n_take = min(self.examples_per_task, len(rows))
            else:
                n_take = min(50, len(rows))

            for i in range(n_take):
                row = rows[i]
                task = self._build_task_dict(config_name, row, scoring, i)
                tasks.append(task)

        if self.verbose:
            print(f"Loaded {len(tasks)} BIG-bench examples from {len(self.task_configs)} configs.")
        return tasks

    def _load_rows(self, load_dataset, config_name: str, undersized: bool) -> List[Dict[str, Any]]:
        """Load train rows; for undersized tasks also append validation rows."""
        try:
            ds_train = load_dataset(
                "tasksource/bigbench", config_name, split="train", trust_remote_code=True
            )
            rows = list(ds_train)
        except Exception as e:
            if self.verbose:
                print(f"  Skipping {config_name} (train): {e}")
            return []

        if undersized:
            try:
                ds_val = load_dataset(
                    "tasksource/bigbench", config_name, split="validation", trust_remote_code=True
                )
                rows = rows + list(ds_val)
            except Exception:
                pass  # validation split absent – use train only

        return rows

    def _build_task_dict(
        self,
        config_name: str,
        row: Dict[str, Any],
        scoring: str,
        position: int,
    ) -> Dict[str, Any]:
        inputs = row.get("inputs", "")
        targets = row.get("targets", [])
        mc_targets = row.get("multiple_choice_targets", [])
        mc_scores = row.get("multiple_choice_scores", [])
        idx = row.get("idx", position)

        # For non-MC tasks, targets holds the ground-truth string(s).
        correct_text = ""
        correct_idx = None

        if scoring == "mc":
            if not mc_targets and targets:
                mc_targets = list(targets) if isinstance(targets, (list, tuple)) else [targets]
                mc_scores = [1] if mc_targets else []
            for k, s in enumerate(mc_scores):
                if s == 1:
                    correct_idx = k
                    break
            if correct_idx is None and mc_targets:
                correct_idx = 0
            correct_text = mc_targets[correct_idx] if correct_idx is not None and mc_targets else ""
        else:
            # exact or bleu: ground truth is the first target string.
            if isinstance(targets, (list, tuple)) and targets:
                correct_text = str(targets[0])
            elif targets:
                correct_text = str(targets)

        return {
            "task_id": f"{config_name}_{idx}",
            "config": config_name,
            "scoring": scoring,
            "inputs": inputs,
            "targets": targets,
            "multiple_choice_targets": mc_targets,
            "multiple_choice_scores": mc_scores,
            "correct_idx": correct_idx,
            "correct_text": correct_text,
        }

    # ------------------------------------------------------------------
    # Task formatting
    # ------------------------------------------------------------------

    def format_task(self, task: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        inputs = task["inputs"]
        choices = task.get("multiple_choice_targets", [])
        scoring = task.get("scoring", "mc")
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(choices)]

        if scoring == "mc" and choices:
            choice_lines = [f"  {letters[i]}. {choice}" for i, choice in enumerate(choices)]
            prompt = (
                f"{inputs}\n\nChoices:\n{chr(10).join(choice_lines)}\n\n"
                "Respond with the letter of the correct answer (e.g. A, B, C). "
                'You may also output JSON: {"answer": "<letter>", "reasoning": "...", "confidence": 0.0-1.0}'
            )
        else:
            prompt = (
                f"{inputs}\n\n"
                "Respond with your answer as a short string. "
                'You may output JSON: {"answer": "<your answer>", "reasoning": "...", "confidence": 0.0-1.0}'
            )

        context = {
            "benchmark_type": "bigbench",
            "task_type": scoring,
            "config": task["config"],
            "max_completion_tokens": 512,
        }
        return prompt, context

    # ------------------------------------------------------------------
    # Answer parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_mc_answer(response_text: str, choice_letters: str) -> Optional[str]:
        if not response_text or not response_text.strip():
            return None
        text = response_text.strip()

        code_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if code_match:
            text = code_match.group(1).strip()
        for start in ["{", "["]:
            i = text.find(start)
            if i != -1:
                try:
                    obj = json.loads(text[i:].split("\n")[0].strip().rstrip(";"))
                    ans = obj.get("answer") or obj.get("response") or obj.get("choice")
                    if ans is not None:
                        ans = str(ans).strip().upper()
                        if ans in choice_letters:
                            return ans
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass

        for c in choice_letters:
            if re.search(rf"\b{c}\b", text, re.IGNORECASE):
                return c.upper()
        m = re.search(r"(?:answer|choice)\s*[:\s]*([A-Z])\b", text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        return None

    @staticmethod
    def _parse_free_answer(response_text: str) -> str:
        if not response_text:
            return ""
        text = response_text.strip()
        code_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if code_match:
            text = code_match.group(1).strip()
        for start in ["{", "["]:
            i = text.find(start)
            if i != -1:
                try:
                    obj = json.loads(text[i:].split("\n")[0].strip().rstrip(";"))
                    ans = obj.get("answer") or obj.get("response")
                    if ans is not None:
                        return str(ans).strip()
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass
        return text

    @staticmethod
    def _score_exact(predicted: str, correct: str) -> bool:
        return predicted.strip().lower() == correct.strip().lower()

    @staticmethod
    def _score_bleu(predicted: str, correct: str) -> float:
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # type: ignore
            reference = [correct.strip().lower().split()]
            hypothesis = predicted.strip().lower().split()
            if not hypothesis:
                return 0.0
            sf = SmoothingFunction().method1
            return sentence_bleu(reference, hypothesis, smoothing_function=sf)
        except Exception:
            return float(BigBenchRunner._score_exact(predicted, correct))

    def check_answer(self, predicted_raw: str, task: Dict[str, Any]) -> tuple[bool, float]:
        """Return (is_correct_bool, float_score) appropriate for the scoring method."""
        scoring = task.get("scoring", "mc")

        if scoring == "mc":
            choices = task.get("multiple_choice_targets", [])
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(choices)]
            predicted_letter = self._parse_mc_answer(predicted_raw, letters)
            correct_idx = task.get("correct_idx", 0)
            correct_letter = letters[correct_idx] if correct_idx is not None and correct_idx < len(letters) else None
            is_correct = bool(predicted_letter and predicted_letter.upper() == (correct_letter or "").upper())
            return is_correct, 1.0 if is_correct else 0.0

        predicted_str = self._parse_free_answer(predicted_raw)
        correct_str = task.get("correct_text", "")

        if scoring == "exact":
            is_correct = self._score_exact(predicted_str, correct_str)
            return is_correct, 1.0 if is_correct else 0.0

        if scoring == "bleu":
            score = self._score_bleu(predicted_str, correct_str)
            # treat BLEU >= 0.5 as "correct" for the pass/fail boolean
            return score >= 0.5, score

        return False, 0.0

    # ------------------------------------------------------------------
    # Per-task processing
    # ------------------------------------------------------------------

    def _process_task(self, task: Dict[str, Any]) -> EvaluationResult:
        start_time = time.time()
        task_id = task["task_id"]
        task_text, context = self.format_task(task)
        scoring = task.get("scoring", "mc")

        with TraceCapture(
            task_id=task_id,
            agent_type=self.agent.__class__.__name__,
            model=self.agent.model,
            input_question=task.get("inputs", "")[:500],
        ) as trace_ctx:
            response = self.agent.respond_to_task(task_text, context)
            latency = time.time() - start_time

            is_correct, float_score = self.check_answer(response.response or "", task)

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
                            benchmark="bigbench",
                            agent_type=self.agent.__class__.__name__,
                        )

            trace_ctx.trace.final_output = (response.response or "")[:200]
            trace_ctx.trace.predicted = (response.response or "")[:100]
            trace_ctx.trace.correct = task.get("correct_text", "")[:100]
            trace_ctx.trace.match = is_correct
            trace_ctx.trace.confidence = response.confidence
            trace_ctx.trace.total_latency = latency
            trace_ctx.trace.total_cost = cost
            trace_ctx.trace.reasoning = response.reasoning or ""
            trace_ctx.trace.timed_out = bool(meta.get("timed_out", False))

            with self._lock:
                self._total_count += 1
                if is_correct:
                    self._correct_count += 1
                cfg = task.get("config", "unknown")
                if cfg not in self._task_stats:
                    self._task_stats[cfg] = {
                        "correct": 0, "total": 0, "score_sum": 0.0, "scoring": scoring
                    }
                self._task_stats[cfg]["total"] += 1
                self._task_stats[cfg]["score_sum"] += float_score
                if is_correct:
                    self._task_stats[cfg]["correct"] += 1

        return EvaluationResult(
            task_id=task_id,
            prompt=task.get("inputs", ""),
            agent_response=response.response,
            success=is_correct,
            score=float_score,
            latency=latency,
            cost=cost,
            metadata={
                "config": task.get("config"),
                "scoring": scoring,
                "expected": task.get("correct_text"),
                "expected_idx": task.get("correct_idx"),
                "trace": trace_ctx.trace,
            },
        )

    # ------------------------------------------------------------------
    # Incremental saving
    # ------------------------------------------------------------------

    def _save_result_incremental(self, result: EvaluationResult) -> None:
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
                    "match": result.success,
                    "score": result.score,
                    "latency": round(result.latency or 0, 2),
                    "cost": round(result.cost or 0, 6),
                }
            with open(task_dir / "trace.json", "w") as f:
                json.dump(trace_data, f, indent=2, default=str)
            with open(self._output_dir / "results.jsonl", "a") as f:
                f.write(
                    json.dumps({
                        "task_id": result.task_id,
                        "config": metadata.get("config"),
                        "scoring": metadata.get("scoring"),
                        "match": result.success,
                        "score": result.score,
                        "latency": round(result.latency or 0, 2),
                        "cost": round(result.cost or 0, 6),
                        "timed_out": bool((metadata.get("trace") and getattr(metadata["trace"], "timed_out", False)) or False),
                    })
                    + "\n"
                )

    # ------------------------------------------------------------------
    # Execution paths
    # ------------------------------------------------------------------

    def _run_sequential(self, tasks: List[Dict[str, Any]]) -> List[EvaluationResult]:
        results: List[EvaluationResult] = []
        pbar = tqdm(
            tasks,
            desc="BigBench",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] Acc: {postfix}",
            postfix="---%",
        )
        for task in pbar:
            result = self._process_task(task)
            results.append(result)
            self._save_result_incremental(result)
            with self._lock:
                acc = (self._correct_count / self._total_count * 100) if self._total_count else 0
            pbar.set_postfix_str(f"{acc:.1f}%")
            if self.verbose:
                tqdm.write(f"  {result.task_id} {'PASS' if result.success else 'FAIL'} ({result.latency:.1f}s)")
        return results

    def _run_concurrent(self, tasks: List[Dict[str, Any]]) -> List[EvaluationResult]:
        results: List[Optional[EvaluationResult]] = [None] * len(tasks)
        limiter = ThreadSafeAdaptiveLimiter(
            max_concurrency=self.concurrency,
            min_concurrency=1,
            backoff_factor=0.5,
            recovery_threshold=10,
        )

        def _process(t: Dict[str, Any]) -> EvaluationResult:
            wait = limiter.wait_if_needed()
            if wait > 0 and self.verbose:
                tqdm.write(f"  Rate-limit backoff: {wait:.1f}s")
            try:
                r = self._process_task(t)
                limiter.record_success()
                return r
            except Exception as e:
                if "rate" in str(e).lower() or "429" in str(e):
                    limiter.record_error(backoff_seconds=5.0)
                raise

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            future_to_idx = {pool.submit(_process, t): i for i, t in enumerate(tasks)}
            pbar = tqdm(
                total=len(tasks),
                desc=f"BigBench ({self.concurrency}x)",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] Acc: {postfix}",
                postfix="---%",
            )
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
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

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        limit: Optional[int] = None,
        save_results: bool = True,
    ) -> List[EvaluationResult]:
        tasks = self.load_tasks(limit)
        self._correct_count = 0
        self._total_count = 0
        self._task_stats = {}

        print(f"\nRunning BIG-bench ({self.suite}) with {len(tasks)} examples across {len(self.task_configs)} tasks")
        print(f"Agent:        {self.agent.__class__.__name__}")
        print(f"Model:        {self.agent.model}")
        print(f"Backend:      {self.backend}")
        print(f"Architecture: {self.architecture}")
        print(f"Concurrency:  {self.concurrency}")

        if save_results:
            base_dir = (
                self.run_dir
                or Path("results") / "bigbench_lite" / self.backend / self.agent.model / self.architecture
                / time.strftime("%Y%m%d_%H%M%S")
            )
            self._output_dir = base_dir
            self._output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving to:    {self._output_dir}/\n")

        if self.concurrency > 1:
            results = self._run_concurrent(tasks)
        else:
            results = self._run_sequential(tasks)

        weighted_accuracy = self._compute_weighted_accuracy()
        raw_accuracy = self._correct_count / self._total_count if self._total_count else 0.0

        print(f"\nWeighted accuracy (equal task weight): {weighted_accuracy * 100:.1f}%")
        print(f"Raw accuracy (all examples):           {raw_accuracy * 100:.1f}% ({self._correct_count}/{self._total_count})")
        if self._task_stats:
            print("By task:")
            for cfg, st in self._task_stats.items():
                task_acc = st["correct"] / st["total"] * 100 if st["total"] else 0
                print(f"  {cfg:<48} {task_acc:5.1f}% ({st['correct']}/{st['total']}) [{st['scoring']}]")

        if save_results and self._output_dir:
            self._save_summary(results, raw_accuracy, weighted_accuracy)

        return results

    def _compute_weighted_accuracy(self) -> float:
        """Equal-weighted mean of per-task accuracy (each task counts as 1/N)."""
        if not self._task_stats:
            return 0.0
        per_task_accs = []
        for st in self._task_stats.values():
            if st["total"] > 0:
                scoring = st.get("scoring", "mc")
                if scoring == "bleu":
                    per_task_accs.append(st["score_sum"] / st["total"])
                else:
                    per_task_accs.append(st["correct"] / st["total"])
        return sum(per_task_accs) / len(per_task_accs) if per_task_accs else 0.0

    def _save_summary(
        self,
        results: List[EvaluationResult],
        raw_accuracy: float,
        weighted_accuracy: float,
    ) -> None:
        if not self._output_dir:
            return
        total_latency = sum(r.latency or 0 for r in results)
        total_cost = sum(r.cost or 0 for r in results)

        # Per-task breakdown with accuracy per task
        breakdown: Dict[str, Any] = {}
        for cfg, st in self._task_stats.items():
            task_meta = self._task_meta.get(cfg, {})
            scoring = st.get("scoring", "mc")
            if scoring == "bleu":
                task_score = st["score_sum"] / st["total"] if st["total"] else 0.0
            else:
                task_score = st["correct"] / st["total"] if st["total"] else 0.0
            breakdown[cfg] = {
                "accuracy": task_score,
                "correct": st["correct"],
                "total": st["total"],
                "scoring": scoring,
                "undersized": task_meta.get("undersized", False),
            }

        data = {
            "agent": self.agent.__class__.__name__,
            "model": self.agent.model,
            "backend": self.backend,
            "architecture": self.architecture,
            "benchmark": "bigbench",
            "suite": self.suite,
            "task_count": len(self.task_configs),
            "examples_per_task": self.examples_per_task,
            "num_examples_total": len(results),
            "raw_accuracy": raw_accuracy,
            "weighted_accuracy": weighted_accuracy,
            "total_latency_seconds": total_latency,
            "avg_latency_seconds": total_latency / len(results) if results else 0,
            "total_cost_usd": total_cost,
            "results_dir": str(self._output_dir),
            "task_configs": self.task_configs,
            "breakdown_by_task": breakdown,
        }
        with open(self._output_dir / "summary.json", "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nSummary saved to: {self._output_dir / 'summary.json'}")
