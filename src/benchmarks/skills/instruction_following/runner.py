"""
Instruction Following Benchmark Runner.

Tests model's ability to follow increasingly complex matrix transformation instructions
through 28 levels of difficulty.
"""

import ast
import json
import re
import time
import copy
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import List, Dict, Any, Optional

from tqdm import tqdm

from ....agents.base_agent import BaseAgent, BenchmarkResponse, EvaluationResult
from ....evaluation.cost_tracker import CostTracker
from ....utils.adaptive_limiter import ThreadSafeAdaptiveLimiter
from ....utils.trace import TraceCapture, QuestionTrace

from .matrix_instruction_following import (
    LEVEL_RULES,
    gen_matrix,
    check_answer,
)


class InstructionFollowingRunner:
    """
    Runner for Instruction Following evaluation.
    
    Tests the model's ability to follow matrix transformation rules across 28 levels
    of increasing difficulty. Each task is a rollout of 28 consecutive levels.
    """
    
    def __init__(
        self,
        agent: BaseAgent,
        cost_tracker: Optional[CostTracker] = None,
        verbose: bool = False,
        concurrency: int = 1,
        run_dir: Optional[Path] = None,
        num_tasks: int = 10,
        matrix_size: int = 4,
    ):
        """
        Initialize Instruction Following runner.
        
        Args:
            agent: Agent to evaluate
            cost_tracker: Optional cost tracker
            verbose: Enable verbose output
            concurrency: Number of concurrent requests
            run_dir: Optional directory to save results
            num_tasks: Number of task rollouts to generate
            matrix_size: Initial matrix size (default: 10x10)
        """
        self.agent = agent
        self.cost_tracker = cost_tracker
        self.verbose = verbose
        self.concurrency = concurrency
        self.run_dir = run_dir
        self.num_tasks = num_tasks
        self.matrix_size = matrix_size
        
        self._lock = Lock()
        self._completed_count = 0
        self._total_count = 0
        self._output_dir = None
        
        # Track metrics by level
        self._level_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    
    def load_tasks(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Generate instruction following tasks.
        
        Args:
            limit: Maximum number of task rollouts
        
        Returns:
            List of task rollouts, each containing 28 levels
        """
        num_rollouts = limit if limit else self.num_tasks
        
        if self.verbose:
            print(f"Generating {num_rollouts} instruction following task rollouts...")
        
        tasks = []
        for tid in range(num_rollouts):
            size = self.matrix_size
            current = gen_matrix(size)
            levels = []
            
            for lvl in range(1, 29):
                name, text, fn = LEVEL_RULES[lvl]
                inp = copy.deepcopy(current)
                out = fn(inp)
                levels.append({
                    "level": lvl,
                    "rule_name": name,
                    "rule_text": text,
                    "input_matrix": inp,
                    "target_matrix": out
                })
                current = out
                # Grow matrix every 3 levels (up to 6x6 max)
                if lvl % 3 == 0:
                    size = min(size + 1, 6)
            
            tasks.append({
                "task_id": f"if_rollout_{tid:04d}",
                "rollout_id": tid,
                "levels": levels,
                "initial_matrix_size": self.matrix_size,
            })
        
        if self.verbose:
            print(f"Generated {len(tasks)} task rollouts (28 levels each)")
        
        return tasks
    
    def format_task(self, level_data: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Format a single level as a task for the agent.
        
        Uses 'instruction_following' benchmark_type so the agent uses a
        task-specific system prompt that asks for a matrix (not the generic
        reasoning JSON wrapper that conflicts with the task format).
        """
        
        task_text = f"""LEVEL {level_data['level']}

RULE:
{level_data['rule_text']}

INPUT MATRIX:
{level_data['input_matrix']}

TASK:
Apply the rule exactly to the input matrix.

Return ONLY a JSON object in this exact format:
{{"matrix": [[...], [...], ...], "confidence": 0.0-1.0}}
"""
        
        context = {
            "benchmark_type": "instruction_following",
            "task_type": "instruction_following",
            "level": level_data['level'],
            "rule_name": level_data['rule_name'],
        }
        
        return task_text, context
    
    def _parse_matrix_response(self, response_text: str) -> Optional[List[List]]:
        """Parse matrix from agent response.
        
        Handles multiple response formats including when the generic system
        prompt wraps the matrix inside {"reasoning": ..., "answer": ...}:
        
        - Direct: {"matrix": [[...]], "confidence": 0.9}
        - Wrapped: {"answer": {"matrix": [[...]]}, ...}
        - Wrapped str: {"answer": "[[1,2],[3,4]]", ...}
        - Python dict: {'matrix': [[...]], 'confidence': 0.9}
        - Raw list: [[1, 2], [3, 4]]
        - Markdown code block: ```json\n{"matrix": [...]}\n```
        """
        if not response_text or not response_text.strip():
            return None
        
        text = response_text.strip()
        
        # Strip markdown code block wrappers
        code_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if code_match:
            text = code_match.group(1).strip()
        
        # Strategy 1: Try JSON parsing first (most reliable)
        matrix = self._try_extract_matrix_from_text(text)
        if matrix is not None:
            return matrix
        
        # Strategy 2: Try ast.literal_eval (handles Python dict with single quotes)
        try:
            parsed = ast.literal_eval(text)
            matrix = self._extract_matrix_from_parsed(parsed)
            if matrix is not None:
                return matrix
        except (ValueError, SyntaxError):
            pass
        
        # Strategy 3: Extract [[...]] from anywhere in the text
        if "[[" in text and "]]" in text:
            start = text.find("[[")
            end = text.rfind("]]") + 2
            matrix_str = text[start:end]
            try:
                matrix = ast.literal_eval(matrix_str)
                return self._normalize_matrix(matrix)
            except (ValueError, SyntaxError):
                pass
            try:
                return self._normalize_matrix(json.loads(matrix_str))
            except json.JSONDecodeError:
                pass
        
        if self.verbose:
            print(f"  Failed to parse matrix from: {text[:200]}...")
        return None
    
    def _try_extract_matrix_from_text(self, text: str) -> Optional[List[List]]:
        """Try to parse text as JSON and extract matrix from it."""
        try:
            parsed = json.loads(text)
            return self._extract_matrix_from_parsed(parsed)
        except (json.JSONDecodeError, ValueError):
            return None
    
    def _extract_matrix_from_parsed(self, parsed: Any) -> Optional[List[List]]:
        """Extract a matrix from a parsed dict/list, handling nested wrappers."""
        if isinstance(parsed, list):
            return self._normalize_matrix(parsed)
        
        if not isinstance(parsed, dict):
            return None
        
        # Direct: {"matrix": [[...]]}
        if "matrix" in parsed:
            val = parsed["matrix"]
            if isinstance(val, (list, tuple)):
                return self._normalize_matrix(val)
            # matrix might be a string representation
            if isinstance(val, str):
                return self._parse_matrix_string(val)
        
        # Wrapped: {"answer": {"matrix": [[...]]}} or {"answer": [[...]]} or {"answer": "[[...]]"}
        for key in ("answer", "response", "result", "output"):
            if key in parsed:
                val = parsed[key]
                if isinstance(val, dict) and "matrix" in val:
                    return self._normalize_matrix(val["matrix"])
                if isinstance(val, (list, tuple)):
                    return self._normalize_matrix(val)
                if isinstance(val, str):
                    # Try to parse the string value as matrix
                    inner = self._parse_matrix_string(val)
                    if inner is not None:
                        return inner
        
        return None
    
    def _parse_matrix_string(self, s: str) -> Optional[List[List]]:
        """Try to parse a string as a matrix."""
        s = s.strip()
        if not s:
            return None
        try:
            return self._normalize_matrix(json.loads(s))
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            return self._normalize_matrix(ast.literal_eval(s))
        except (ValueError, SyntaxError):
            pass
        # Try extracting [[...]] from within the string
        if "[[" in s and "]]" in s:
            start = s.find("[[")
            end = s.rfind("]]") + 2
            try:
                return self._normalize_matrix(ast.literal_eval(s[start:end]))
            except (ValueError, SyntaxError):
                pass
        return None
    
    def _normalize_matrix(self, matrix: Any) -> Optional[List[List]]:
        """Normalize matrix: convert tuples to lists recursively."""
        if not isinstance(matrix, (list, tuple)):
            return None
        result = []
        for row in matrix:
            if isinstance(row, (list, tuple)):
                result.append([
                    list(x) if isinstance(x, tuple) else x
                    for x in row
                ])
            else:
                return None  # Not a 2D structure
        return result
    
    def _process_task(self, task: Dict[str, Any]) -> EvaluationResult:
        """
        Process a single task rollout (28 levels).
        
        Returns:
            EvaluationResult with score = levels_passed / 28
        """
        task_id = task["task_id"]
        rollout_id = task["rollout_id"]
        levels = task["levels"]
        
        levels_passed = 0
        level_results = []
        total_latency = 0
        total_cost = 0
        
        print(f"  Rollout {rollout_id}: starting ({len(levels)} levels, {task['initial_matrix_size']}x{task['initial_matrix_size']} matrix)", flush=True)
        
        # Process each level sequentially — stops on first failure
        for level_idx, level_data in enumerate(levels):
            level_num = level_data["level"]
            rule_name = level_data["rule_name"]
            
            # Format task
            task_text, context = self.format_task(level_data)
            
            start_time = time.time()
            
            # Run agent
            try:
                response = self.agent.respond_to_task(task_text, context)
                latency = time.time() - start_time
                
                # Parse matrix from response
                model_matrix = self._parse_matrix_response(response.response)
                target_matrix = level_data["target_matrix"]
                
                # Check correctness
                if model_matrix is not None:
                    is_correct = check_answer(model_matrix, target_matrix)
                else:
                    is_correct = False
                
                # Track cost
                cost = 0
                if self.cost_tracker and response.metadata:
                    prompt_tokens = response.metadata.get("prompt_tokens", 0)
                    completion_tokens = response.metadata.get("completion_tokens", 0)
                    cost = self.cost_tracker.log_usage(
                        self.agent.model,
                        prompt_tokens,
                        completion_tokens,
                    )
                
                total_latency += latency
                total_cost += cost
                
                # Print per-level result immediately
                status = "PASS" if is_correct else "FAIL"
                parse_status = "parsed" if model_matrix is not None else "PARSE_ERROR"
                print(f"    L{level_num:02d} {rule_name:<14s} {status}  ({latency:.1f}s, {parse_status})", flush=True)
                
                level_results.append({
                    "level": level_num,
                    "rule_name": rule_name,
                    "passed": is_correct,
                    "latency": latency,
                    "cost": cost,
                    "model_matrix": model_matrix,
                    "target_matrix": target_matrix,
                })
                
                # Update level stats
                with self._lock:
                    self._level_stats[level_num]["total"] += 1
                    if is_correct:
                        self._level_stats[level_num]["correct"] += 1
                
                # Stop on first failure
                if is_correct:
                    levels_passed += 1
                else:
                    print(f"  Rollout {rollout_id}: stopped at level {level_num} ({levels_passed}/{level_num} passed)", flush=True)
                    break
            
            except Exception as e:
                latency = time.time() - start_time
                total_latency += latency
                print(f"    L{level_num:02d} {rule_name:<14s} ERROR ({latency:.1f}s): {e}", flush=True)
                print(f"  Rollout {rollout_id}: stopped at level {level_num} (error)", flush=True)
                break
        else:
            # Completed all 28 levels without failure
            print(f"  Rollout {rollout_id}: PERFECT — passed all 28 levels!", flush=True)
        
        print(f"  Rollout {rollout_id} done: {levels_passed}/28 levels, {total_latency:.1f}s total", flush=True)
        
        # Calculate rollout score
        score = levels_passed / 28.0
        success = levels_passed >= 1  # Success if passed at least level 1
        
        # Create evaluation result
        result = EvaluationResult(
            task_id=task_id,
            prompt=f"Instruction following rollout with {len(levels)} levels",
            agent_response=f"Passed {levels_passed}/28 levels",
            success=success,
            score=score,
            latency=total_latency,
            cost=total_cost,
            metadata={
                "rollout_id": rollout_id,
                "levels_passed": levels_passed,
                "total_levels": 28,
                "level_results": level_results,
                "initial_matrix_size": task["initial_matrix_size"],
            }
        )
        
        # Save result incrementally
        if self.run_dir:
            self._save_result_incremental(result)
        
        # Update progress
        with self._lock:
            self._completed_count += 1
        
        return result
    
    def _save_result_incremental(self, result: EvaluationResult):
        """Save a single result to disk."""
        if not self._output_dir:
            return
        
        # Create task directory
        task_dir = self._output_dir / result.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full trace
        trace_file = task_dir / "trace.json"
        with open(trace_file, 'w') as f:
            json.dump({
                "task_id": result.task_id,
                "success": result.success,
                "score": result.score,
                "latency": result.latency,
                "cost": result.cost,
                "metadata": result.metadata,
                "prompt": result.prompt,
                "response": result.agent_response,
            }, f, indent=2)
        
        # Append to results.jsonl
        results_file = self._output_dir / "results.jsonl"
        with open(results_file, 'a') as f:
            f.write(json.dumps({
                "task_id": result.task_id,
                "rollout_id": result.metadata.get("rollout_id"),
                "levels_passed": result.metadata.get("levels_passed"),
                "score": result.score,
                "success": result.success,
                "latency": result.latency,
                "cost": result.cost,
            }) + "\n")
    
    def _save_summary(self, results: List[EvaluationResult]):
        """Save summary statistics."""
        if not self._output_dir:
            return
        
        total = len(results)
        if total == 0:
            return
        
        # Overall metrics
        mean_score = sum(r.score for r in results) / total
        mean_latency = sum(r.latency for r in results) / total
        total_cost = sum(r.cost or 0 for r in results)
        success_count = sum(1 for r in results if r.success)
        
        # Level-specific accuracy
        level_accuracy = {}
        for level_num in range(1, 29):
            stats = self._level_stats[level_num]
            if stats["total"] > 0:
                level_accuracy[f"level_{level_num}"] = {
                    "accuracy": stats["correct"] / stats["total"],
                    "correct": stats["correct"],
                    "total": stats["total"],
                }
        
        # Distribution of levels passed
        levels_passed_dist = defaultdict(int)
        for r in results:
            levels_passed_dist[r.metadata.get("levels_passed", 0)] += 1
        
        summary = {
            "agent": self.agent.__class__.__name__,
            "model": self.agent.model,
            "num_rollouts": total,
            "mean_score": round(mean_score, 4),
            "success_rate": round(success_count / total, 4),
            "mean_latency": round(mean_latency, 2),
            "total_cost": round(total_cost, 4),
            "level_accuracy": level_accuracy,
            "levels_passed_distribution": dict(levels_passed_dist),
        }
        
        summary_file = self._output_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        if self.verbose:
            print(f"\nSummary saved to: {summary_file}")
    
    def _run_sequential(self, tasks: List[Dict[str, Any]]) -> List[EvaluationResult]:
        """Run tasks sequentially with per-level output."""
        results = []
        
        for i, task in enumerate(tasks):
            print(f"\n[{i+1}/{len(tasks)}] {task['task_id']}")
            result = self._process_task(task)
            results.append(result)
        
        return results
    
    def _run_concurrent(self, tasks: List[Dict[str, Any]]) -> List[EvaluationResult]:
        """Run tasks concurrently."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = {executor.submit(self._process_task, task): task for task in tasks}
            
            with tqdm(total=len(tasks), desc=f"Processing {self.agent.model}", disable=not self.verbose) as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                        pbar.update(1)
                    except Exception as e:
                        print(f"Task failed: {e}")
                        pbar.update(1)
        
        return results
    
    def run(self, limit: Optional[int] = None, save_results: bool = True) -> List[EvaluationResult]:
        """
        Run the instruction following benchmark.
        
        Args:
            limit: Maximum number of task rollouts (default: num_tasks from init)
            save_results: Whether to save results to disk
        
        Returns:
            List of EvaluationResult objects
        """
        # Load tasks
        tasks = self.load_tasks(limit)
        
        # Setup output directory
        if save_results:
            if self.run_dir:
                self._output_dir = self.run_dir / self.agent.__class__.__name__
            else:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                self._output_dir = Path("results") / "instruction_following" / f"{self.agent.model}_{timestamp}" / self.agent.__class__.__name__
            
            self._output_dir.mkdir(parents=True, exist_ok=True)
            
            if self.verbose:
                print(f"Results will be saved to: {self._output_dir}")
        
        # Run tasks
        if self.concurrency > 1:
            results = self._run_concurrent(tasks)
        else:
            results = self._run_sequential(tasks)
        
        # Save summary
        if save_results:
            self._save_summary(results)
        
        return results
