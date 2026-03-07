"""
Word Instruction Following Benchmark Runner.

Tests model's ability to follow 30 word-list transformation rules of
increasing complexity.  Each level is evaluated independently with the
ground-truth input (non-cascading), so every rule contributes to the
final score regardless of earlier failures.

By default sweeps list sizes 1–10, producing one rollout per size.
"""

import ast
import json
import re
import time
import copy
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import List, Dict, Any, Optional, Sequence

from tqdm import tqdm

from ....agents.base_agent import BaseAgent, BenchmarkResponse, EvaluationResult
from ....evaluation.cost_tracker import CostTracker

from .word_instruction_following import (
    LEVEL_RULES,
    gen_list,
    check_answer,
)

NUM_LEVELS = 30
DEFAULT_SIZES = list(range(1, 11))  # 1 through 10


class WordInstructionFollowingRunner:
    """
    Runner for Word-list Instruction Following evaluation.

    Each of the 30 levels is tested independently with ground-truth input
    (non-cascading).  By default sweeps list sizes 1–10, one rollout per
    size.  Results are reported both overall and broken down by list size.
    """

    def __init__(
        self,
        agent: BaseAgent,
        cost_tracker: Optional[CostTracker] = None,
        verbose: bool = False,
        concurrency: int = 1,
        run_dir: Optional[Path] = None,
        num_tasks: int = 1,
        list_size: Optional[int] = None,
        list_sizes: Optional[Sequence[int]] = None,
    ):
        """
        Args:
            list_size:  Single size (overrides list_sizes).
            list_sizes: Sequence of sizes to sweep. Default 1–10.
            num_tasks:  Rollouts per size (default 1).
        """
        self.agent = agent
        self.cost_tracker = cost_tracker
        self.verbose = verbose
        self.concurrency = concurrency
        self.run_dir = run_dir
        self.num_tasks = num_tasks

        if list_size is not None:
            self.list_sizes = [list_size]
        elif list_sizes is not None:
            self.list_sizes = list(list_sizes)
        else:
            self.list_sizes = DEFAULT_SIZES

        self._lock = Lock()
        self._completed_count = 0
        self._total_count = 0
        self._output_dir = None

        self._level_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        self._size_stats: Dict[int, List[int]] = defaultdict(list)

    def load_tasks(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        sizes = self.list_sizes
        rollouts_per_size = limit if limit else self.num_tasks

        if self.verbose:
            print(
                f"Generating tasks: sizes {sizes}, "
                f"{rollouts_per_size} rollout(s) per size "
                f"({len(sizes) * rollouts_per_size} total)"
            )

        tasks = []
        for size in sizes:
            for tid in range(rollouts_per_size):
                random.seed(tid * 1000 + size)
                current = gen_list(size)
                levels = []

                for lvl in range(1, NUM_LEVELS + 1):
                    name, text, fn = LEVEL_RULES[lvl]
                    inp = copy.deepcopy(current)
                    out = fn(inp)
                    levels.append({
                        "level": lvl,
                        "rule_name": name,
                        "rule_text": text,
                        "input_list": inp,
                        "target_list": out,
                    })
                    current = out

                tasks.append({
                    "task_id": f"wif_sz{size:02d}_r{tid:02d}",
                    "rollout_id": tid,
                    "levels": levels,
                    "initial_list_size": size,
                })

        if self.verbose:
            print(f"Generated {len(tasks)} tasks ({NUM_LEVELS} levels each)")

        return tasks

    def format_task(self, level_data: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        task_text = f"""LEVEL {level_data['level']}

RULE:
{level_data['rule_text']}

INPUT LIST:
{level_data['input_list']}

TASK:
Apply the rule exactly to the input list.

Return ONLY a JSON object in this exact format:
{{"list": <result>, "confidence": 0.0-1.0}}

Where <result> is the Python list/structure produced by the rule.
"""

        context = {
            "benchmark_type": "word_instruction_following",
            "task_type": "word_instruction_following",
            "level": level_data["level"],
            "rule_name": level_data["rule_name"],
        }

        return task_text, context

    # ── response parsing ────────────────────────────────────────────────

    def _parse_list_response(self, response_text: str) -> Any:
        if not response_text or not response_text.strip():
            return None

        text = response_text.strip()

        code_match = re.search(r'```(?:json|python)?\s*([\s\S]*?)\s*```', text)
        if code_match:
            text = code_match.group(1).strip()

        result = self._try_json_extract(text)
        if result is not None:
            return result

        result = self._try_literal_eval_extract(text)
        if result is not None:
            return result

        result = self._try_bracket_extract(text)
        if result is not None:
            return result

        if self.verbose:
            print(f"  Failed to parse list from: {text[:200]}...")
        return None

    def _try_json_extract(self, text: str) -> Any:
        try:
            parsed = json.loads(text)
            return self._extract_from_parsed(parsed)
        except (json.JSONDecodeError, ValueError):
            return None

    def _try_literal_eval_extract(self, text: str) -> Any:
        try:
            parsed = ast.literal_eval(text)
            return self._extract_from_parsed(parsed)
        except (ValueError, SyntaxError):
            return None

    def _extract_from_parsed(self, parsed: Any) -> Any:
        if isinstance(parsed, list):
            return parsed

        if isinstance(parsed, dict):
            for key in ("list", "result", "answer", "output", "response"):
                if key in parsed:
                    val = parsed[key]
                    if isinstance(val, (list, tuple)):
                        return self._tuples_to_lists(val)
                    if isinstance(val, str):
                        inner = self._try_parse_value(val)
                        if inner is not None:
                            return inner
            for key in ("answer", "response", "result"):
                if key in parsed and isinstance(parsed[key], dict):
                    inner = self._extract_from_parsed(parsed[key])
                    if inner is not None:
                        return inner

        return None

    def _try_bracket_extract(self, text: str) -> Any:
        start = text.find("[")
        if start == -1:
            return None

        depth = 0
        end = None
        for i in range(start, len(text)):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if end is None:
            return None

        return self._try_parse_value(text[start:end])

    def _try_parse_value(self, s: str) -> Any:
        s = s.strip()
        if not s:
            return None
        try:
            val = json.loads(s)
            if isinstance(val, list):
                return val
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            val = ast.literal_eval(s)
            if isinstance(val, (list, tuple)):
                return self._tuples_to_lists(val)
        except (ValueError, SyntaxError):
            pass
        return None

    @staticmethod
    def _tuples_to_lists(obj):
        if isinstance(obj, tuple):
            return [WordInstructionFollowingRunner._tuples_to_lists(x) for x in obj]
        if isinstance(obj, list):
            return [WordInstructionFollowingRunner._tuples_to_lists(x) for x in obj]
        return obj

    # ── task processing ─────────────────────────────────────────────────

    def _process_task(self, task: Dict[str, Any]) -> EvaluationResult:
        task_id = task["task_id"]
        rollout_id = task["rollout_id"]
        list_size = task["initial_list_size"]
        levels = task["levels"]

        levels_correct = 0
        level_results = []
        total_latency = 0
        total_cost = 0

        print(
            f"  {task_id}: starting ({len(levels)} levels, size={list_size})",
            flush=True,
        )

        for level_data in levels:
            level_num = level_data["level"]
            rule_name = level_data["rule_name"]

            task_text, context = self.format_task(level_data)
            start_time = time.time()

            try:
                response = self.agent.respond_to_task(task_text, context)
                latency = time.time() - start_time

                raw_text = (
                    (response.metadata or {}).get("raw_result")
                    or response.response
                )
                model_list = self._parse_list_response(raw_text)
                target_list = level_data["target_list"]

                target_norm = self._tuples_to_lists(target_list)
                model_norm = self._tuples_to_lists(model_list) if model_list is not None else None

                is_correct = model_norm is not None and check_answer(model_norm, target_norm)

                cost = 0
                if self.cost_tracker and response.metadata:
                    prompt_tokens = response.metadata.get("prompt_tokens", 0)
                    completion_tokens = response.metadata.get("completion_tokens", 0)
                    cost = self.cost_tracker.log_usage(
                        self.agent.model, prompt_tokens, completion_tokens,
                    )

                total_latency += latency
                total_cost += cost

                status = "PASS" if is_correct else "FAIL"
                parse_status = "parsed" if model_list is not None else "PARSE_ERROR"
                print(
                    f"    L{level_num:02d} {rule_name:<18s} {status}  "
                    f"({latency:.1f}s, {parse_status})",
                    flush=True,
                )

                level_results.append({
                    "level": level_num,
                    "rule_name": rule_name,
                    "passed": is_correct,
                    "latency": latency,
                    "cost": cost,
                    "model_list": model_norm,
                    "target_list": target_norm,
                })

                with self._lock:
                    self._level_stats[level_num]["total"] += 1
                    if is_correct:
                        self._level_stats[level_num]["correct"] += 1

                if is_correct:
                    levels_correct += 1

            except Exception as e:
                latency = time.time() - start_time
                total_latency += latency
                print(
                    f"    L{level_num:02d} {rule_name:<18s} ERROR ({latency:.1f}s): {e}",
                    flush=True,
                )
                level_results.append({
                    "level": level_num,
                    "rule_name": rule_name,
                    "passed": False,
                    "latency": latency,
                    "cost": 0,
                    "error": str(e),
                })

        if levels_correct == NUM_LEVELS:
            print(
                f"  {task_id}: PERFECT — {NUM_LEVELS}/{NUM_LEVELS}!",
                flush=True,
            )

        print(
            f"  {task_id} done: {levels_correct}/{NUM_LEVELS} correct, "
            f"{total_latency:.1f}s total",
            flush=True,
        )

        with self._lock:
            self._size_stats[list_size].append(levels_correct)

        score = levels_correct / NUM_LEVELS
        success = levels_correct >= 1

        result = EvaluationResult(
            task_id=task_id,
            prompt=f"Word instruction following rollout (size={list_size}, {len(levels)} levels)",
            agent_response=f"{levels_correct}/{NUM_LEVELS} correct",
            success=success,
            score=score,
            latency=total_latency,
            cost=total_cost,
            metadata={
                "rollout_id": rollout_id,
                "list_size": list_size,
                "levels_correct": levels_correct,
                "total_levels": NUM_LEVELS,
                "level_results": level_results,
                "initial_list_size": list_size,
            },
        )

        if self.run_dir:
            self._save_result_incremental(result)

        with self._lock:
            self._completed_count += 1

        return result

    # ── persistence ─────────────────────────────────────────────────────

    def _save_result_incremental(self, result: EvaluationResult):
        if not self._output_dir:
            return

        task_dir = self._output_dir / result.task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        with open(task_dir / "trace.json", "w") as f:
            json.dump({
                "task_id": result.task_id,
                "success": result.success,
                "score": result.score,
                "latency": result.latency,
                "cost": result.cost,
                "metadata": result.metadata,
                "prompt": result.prompt,
                "response": result.agent_response,
            }, f, indent=2, default=str)

        with open(self._output_dir / "results.jsonl", "a") as f:
            f.write(json.dumps({
                "task_id": result.task_id,
                "list_size": result.metadata.get("list_size"),
                "rollout_id": result.metadata.get("rollout_id"),
                "levels_correct": result.metadata.get("levels_correct"),
                "score": result.score,
                "success": result.success,
                "latency": result.latency,
                "cost": result.cost,
            }) + "\n")

    def _save_summary(self, results: List[EvaluationResult]):
        if not self._output_dir:
            return

        total = len(results)
        if total == 0:
            return

        mean_score = sum(r.score for r in results) / total
        mean_latency = sum(r.latency for r in results) / total
        total_cost = sum(r.cost or 0 for r in results)
        success_count = sum(1 for r in results if r.success)

        level_accuracy = {}
        for level_num in range(1, NUM_LEVELS + 1):
            stats = self._level_stats[level_num]
            if stats["total"] > 0:
                level_accuracy[f"level_{level_num}"] = {
                    "accuracy": stats["correct"] / stats["total"],
                    "correct": stats["correct"],
                    "total": stats["total"],
                }

        correct_dist = defaultdict(int)
        for r in results:
            correct_dist[r.metadata.get("levels_correct", 0)] += 1

        # Per-size breakdown
        size_breakdown = {}
        for size in sorted(self._size_stats.keys()):
            correct_list = self._size_stats[size]
            if not correct_list:
                size_breakdown[f"size_{size}"] = {
                    "mean_correct": 0,
                    "max_correct": 0,
                    "rollouts": 0,
                    "mean_score": 0,
                }
                continue
            size_breakdown[f"size_{size}"] = {
                "mean_correct": round(sum(correct_list) / len(correct_list), 2),
                "max_correct": max(correct_list),
                "rollouts": len(correct_list),
                "mean_score": round(sum(c / NUM_LEVELS for c in correct_list) / len(correct_list), 4),
            }

        summary = {
            "agent": self.agent.__class__.__name__,
            "model": self.agent.model,
            "list_sizes": self.list_sizes,
            "num_rollouts_per_size": self.num_tasks,
            "total_rollouts": total,
            "mean_score": round(mean_score, 4),
            "success_rate": round(success_count / total, 4),
            "mean_latency": round(mean_latency, 2),
            "total_cost": round(total_cost, 4),
            "by_size": size_breakdown,
            "level_accuracy": level_accuracy,
            "levels_correct_distribution": dict(correct_dist),
        }

        with open(self._output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Print size breakdown table
        print(f"\n  {'Size':<6} {'Correct':<16} {'Score':<8}")
        print(f"  {'-'*32}")
        for size in sorted(self._size_stats.keys()):
            info = size_breakdown[f"size_{size}"]
            print(
                f"  {size:<6} {info['mean_correct']:>6.1f} / {NUM_LEVELS:<9} "
                f"{info['mean_score']:.3f}"
            )

        if self.verbose:
            print(f"\nSummary saved to: {self._output_dir / 'summary.json'}")

    # ── execution ───────────────────────────────────────────────────────

    def _run_sequential(self, tasks: List[Dict[str, Any]]) -> List[EvaluationResult]:
        results = []
        for i, task in enumerate(tasks):
            print(f"\n[{i+1}/{len(tasks)}] {task['task_id']}")
            results.append(self._process_task(task))
        return results

    def _run_concurrent(self, tasks: List[Dict[str, Any]]) -> List[EvaluationResult]:
        results = []
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = {executor.submit(self._process_task, t): t for t in tasks}
            with tqdm(total=len(tasks), desc=f"Processing {self.agent.model}",
                      disable=not self.verbose) as pbar:
                for future in as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        print(f"Task failed: {e}")
                    pbar.update(1)
        return results

    def run(
        self, limit: Optional[int] = None, save_results: bool = True,
    ) -> List[EvaluationResult]:
        tasks = self.load_tasks(limit)

        if save_results:
            if self.run_dir:
                self._output_dir = self.run_dir / self.agent.__class__.__name__
            else:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                self._output_dir = (
                    Path("results") / "word_instruction_following"
                    / f"{self.agent.model}_{timestamp}"
                    / self.agent.__class__.__name__
                )

            self._output_dir.mkdir(parents=True, exist_ok=True)

            if self.verbose:
                print(f"Results will be saved to: {self._output_dir}")

        if self.concurrency > 1:
            results = self._run_concurrent(tasks)
        else:
            results = self._run_sequential(tasks)

        if save_results:
            self._save_summary(results)

        return results
