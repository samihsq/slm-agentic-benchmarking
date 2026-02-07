"""
BFCL (Berkeley Function Calling Leaderboard) Runner.

BFCL v3 evaluates function calling capabilities with:
- Simple function calls
- Parallel function calls  
- Nested function calls

This provides a standardized baseline for tool-calling evaluation.
Dataset: gorilla-llm/Berkeley-Function-Calling-Leaderboard (~2000 tasks)
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import List, Dict, Any, Optional, Iterator

from datasets import load_dataset
from tqdm import tqdm

from ....agents.base_agent import BaseAgent, BenchmarkResponse, EvaluationResult
from ....evaluation.cost_tracker import CostTracker
from ....utils.adaptive_limiter import ThreadSafeAdaptiveLimiter
from ....utils.trace import TraceCapture, QuestionTrace


class BFCLRunner:
    """
    Runner for BFCL v3 evaluation.
    
    BFCL tests function calling in various scenarios:
    - Single function calls
    - Multiple parallel calls
    - Nested/sequential calls
    """
    
    def __init__(
        self,
        agent: BaseAgent,
        cost_tracker: Optional[CostTracker] = None,
        verbose: bool = False,
        concurrency: int = 1,
        run_dir: Optional[Path] = None,
    ):
        """
        Initialize BFCL runner.
        
        Args:
            agent: Agent to evaluate
            cost_tracker: Optional cost tracker
            verbose: Enable verbose output
            concurrency: Number of concurrent requests
            run_dir: Optional directory to save results (for grouping runs)
        """
        self.agent = agent
        self.cost_tracker = cost_tracker
        self.verbose = verbose
        self.concurrency = concurrency
        self.run_dir = run_dir
        self._lock = Lock()
        self._success_count = 0
        self._total_count = 0
        self._output_dir = None
    
    def load_tasks(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load BFCL v3 tasks from HuggingFace (streaming - no full download).
        
        Args:
            limit: Maximum number of tasks
        
        Returns:
            List of task dicts
        """
        if self.verbose:
            print("Loading BFCL v3 dataset from HuggingFace (streaming)...")
        
        # Use streaming mode - pulls data on-demand without downloading full dataset
        ds = load_dataset('teddyyyy123/bfcl_v3', split='train', streaming=True)
        
        tasks = []
        for i, item in enumerate(ds):
            if limit and i >= limit:
                break
            
            # Parse task type from ID (e.g., "live_simple_0-0-0" -> "simple")
            task_id = item.get('id', f'bfcl_{i}')
            parts = task_id.split('_')
            task_type = parts[1] if len(parts) > 1 and parts[1] in ['simple', 'multiple', 'parallel'] else 'simple'
            
            # Parse function definitions
            functions = []
            func_data = item.get('function', '[]')
            if isinstance(func_data, str):
                try:
                    functions = json.loads(func_data)
                except json.JSONDecodeError:
                    functions = []
            elif isinstance(func_data, list):
                functions = func_data
            
            # Parse ground truth (expected calls)
            ground_truth = item.get('ground_truth', [])
            if isinstance(ground_truth, str):
                try:
                    ground_truth = json.loads(ground_truth)
                except json.JSONDecodeError:
                    ground_truth = []
            
            # Extract user question from chat_completion_input
            chat_input = item.get('chat_completion_input', [])
            if isinstance(chat_input, str):
                try:
                    chat_input = json.loads(chat_input)
                except json.JSONDecodeError:
                    chat_input = []
            
            user_message = ""
            for msg in chat_input:
                if isinstance(msg, dict) and msg.get('role') == 'user':
                    user_message = msg.get('content', '')
                    break
            
            tasks.append({
                "task_id": task_id,
                "type": task_type,
                "description": user_message,
                "functions": functions,
                "ground_truth": ground_truth,
                "raw_question": item.get('question', ''),
            })
        
        if self.verbose:
            print(f"Loaded {len(tasks)} BFCL tasks")
        
        return tasks
    
    def format_task(self, task: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Format task with function descriptions."""
        funcs_desc = "\n".join([
            f"- {f['name']}: {f['description']} (params: {f.get('parameters', {})})"
            for f in task.get("functions", [])
        ])
        
        task_text = f"{task['description']}\n\nAvailable functions:\n{funcs_desc}"
        
        context = {
            "benchmark_type": "tool_calling",
            "tools": task.get("functions", []),
            "task_type": task.get("type", "simple"),
        }
        
        return task_text, context
    
    def check_against_ground_truth(self, response: BenchmarkResponse, ground_truth: List[Dict]) -> tuple[bool, float]:
        """
        Check if response matches ground truth function calls.
        
        Ground truth format: [{"func_name": {"param1": [value1], "param2": [value2]}}]
        """
        if not ground_truth:
            return False, 0.0
        
        meta = response.metadata if isinstance(response.metadata, dict) else {}
        tool_calls = meta.get("tool_calls", [])
        response_text = response.response.lower()
        
        total_expected = len(ground_truth)
        matched = 0
        partial_scores = []
        
        for gt in ground_truth:
            if not isinstance(gt, dict):
                continue
            
            # Ground truth format: {"func_name": {"arg1": [val1], "arg2": [val2]}}
            for func_name, expected_args in gt.items():
                func_matched, func_score = self._check_single_call(
                    tool_calls, response_text, func_name, expected_args or {}
                )
                if func_matched:
                    matched += 1
                partial_scores.append(func_score)
        
        if total_expected == 0:
            return False, 0.0
        
        success = matched == total_expected
        avg_score = sum(partial_scores) / len(partial_scores) if partial_scores else 0.0
        
        return success, avg_score
    
    def _check_single_call(
        self, 
        tool_calls: List[Dict], 
        response_text: str, 
        func_name: str, 
        expected_args: Dict
    ) -> tuple[bool, float]:
        """Check if a single function call matches."""
        func_name_lower = func_name.lower()
        
        # Check tool_calls metadata first
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            
            tc_name = (tc.get("name", "") or tc.get("function", "")).lower()
            tc_args = tc.get("arguments", {}) or tc.get("parameters", {})
            
            if tc_name == func_name_lower:
                # Check arguments match
                if not expected_args:
                    return True, 1.0
                
                matches = 0
                for k, expected_vals in expected_args.items():
                    actual_val = tc_args.get(k)
                    # Ground truth values are in lists, e.g., [7890]
                    if isinstance(expected_vals, list):
                        if actual_val in expected_vals or str(actual_val) in [str(v) for v in expected_vals]:
                            matches += 1
                    elif str(actual_val).lower() == str(expected_vals).lower():
                        matches += 1
                
                score = matches / len(expected_args) if expected_args else 1.0
                return matches == len(expected_args), score
        
        # Try to extract function call from response text (strict parsing)
        # Look for patterns like: func_name(args) or func_name({...})
        import re
        
        # Pattern 1: function_name({...}) or function_name(...)
        call_pattern = rf'{re.escape(func_name)}\s*\(\s*[\{{\[]'
        if re.search(call_pattern, response_text, re.IGNORECASE):
            # Found a function call pattern, try to extract and validate args
            # Extract the arguments portion
            match = re.search(rf'{re.escape(func_name)}\s*\(\s*(\{{[^}}]+\}})', response_text, re.IGNORECASE)
            if match:
                try:
                    import ast
                    args_str = match.group(1).replace("'", '"')
                    parsed_args = json.loads(args_str)
                    
                    # Validate arguments
                    if expected_args:
                        matches = 0
                        for k, expected_vals in expected_args.items():
                            actual_val = parsed_args.get(k)
                            if isinstance(expected_vals, list):
                                if actual_val in expected_vals or str(actual_val) in [str(v) for v in expected_vals]:
                                    matches += 1
                            elif str(actual_val).lower() == str(expected_vals).lower():
                                matches += 1
                        score = matches / len(expected_args)
                        return matches == len(expected_args), score
                    return True, 0.5
                except:
                    pass
        
        # No valid function call found
        return False, 0.0
    
    def _process_task(self, task: Dict[str, Any]) -> EvaluationResult:
        """Process a single task (thread-safe)."""
        start_time = time.time()
        task_id = task["task_id"]
        
        # Format task
        task_text, context = self.format_task(task)
        
        # Use TraceCapture to record all internal agent calls
        with TraceCapture(
            task_id=task_id,
            agent_type=self.agent.__class__.__name__,
            model=self.agent.model,
            input_question=task_text,
        ) as trace_ctx:
            # Run agent
            response = self.agent.respond_to_task(task_text, context)
            
            latency = time.time() - start_time
            
            # Check function call against ground truth
            ground_truth = task.get("ground_truth", [])
            success, score = self.check_against_ground_truth(response, ground_truth)
            
            # Calculate cost (thread-safe)
            cost = 0.0
            meta = response.metadata if isinstance(response.metadata, dict) else {}
            if self.cost_tracker and meta:
                prompt_tokens = meta.get("prompt_tokens", 0)
                completion_tokens = meta.get("completion_tokens", 0)
                if prompt_tokens > 0:
                    with self._lock:
                        cost = self.cost_tracker.log_usage(
                            model=self.agent.model,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            task_id=task_id,
                            benchmark="bfcl",
                            agent_type=self.agent.__class__.__name__,
                        )
            
            # Update trace with results
            trace_ctx.trace.final_output = response.response
            trace_ctx.trace.predicted = str(meta.get("tool_calls", []))
            trace_ctx.trace.correct = str(ground_truth)
            trace_ctx.trace.match = success
            trace_ctx.trace.confidence = score
            trace_ctx.trace.total_latency = latency
            trace_ctx.trace.total_cost = cost
            trace_ctx.trace.reasoning = response.reasoning or ""
        
        return EvaluationResult(
            task_id=task_id,
            prompt=task_text,
            agent_response=response.response,
            success=success,
            score=score,
            latency=latency,
            cost=cost,
            metadata={
                "task_type": task.get("type", "simple"),
                "tool_calls": meta.get("tool_calls", []),
                "reasoning": response.reasoning,
                "ground_truth": ground_truth,
                "question": task.get("description", "")[:200],
                "trace": trace_ctx.trace,
            },
        )
    
    def _save_result_incremental(self, result: EvaluationResult):
        """Save a single result to per-question folder with full trace (thread-safe)."""
        if not self._output_dir:
            return
        
        with self._lock:
            metadata = result.metadata if isinstance(result.metadata, dict) else {}
            
            # Create per-question folder
            question_dir = self._output_dir / result.task_id.replace("/", "_")  # Sanitize task_id
            question_dir.mkdir(parents=True, exist_ok=True)
            
            # Get trace if available
            trace = metadata.get("trace")
            
            if trace and isinstance(trace, QuestionTrace):
                # Save full trace (includes all internal agent calls)
                trace_data = trace.to_dict()
                trace_file = question_dir / "trace.json"
                with open(trace_file, 'w') as f:
                    json.dump(trace_data, f, indent=2)
            else:
                # Fallback: save basic trace.json
                trace_data = {
                    "task_id": result.task_id,
                    "agent_type": self.agent.__class__.__name__,
                    "model": self.agent.model,
                    "input_question": result.prompt[:2000] if result.prompt else "",
                    "calls": [],
                    "final_output": result.agent_response if result.agent_response else "",
                    "predicted": metadata.get("tool_calls", []),
                    "correct": metadata.get("ground_truth", []),
                    "match": result.success,
                    "confidence": result.score,
                    "total_latency": round(result.latency, 2),
                    "total_cost": round(result.cost, 6),
                    "reasoning": metadata.get("reasoning", ""),
                    "task_type": metadata.get("task_type"),
                }
                
                # Include multi-agent steps if available
                for key in ["pipeline_steps", "discussion", "parallel_outputs"]:
                    if key in metadata:
                        trace_data["calls"] = [
                            {
                                "role": step.get("stage") or step.get("agent", "unknown"),
                                "input_prompt": step.get("input", ""),
                                "output_response": step.get("output", ""),
                            }
                            for step in metadata[key]
                        ]
                
                trace_file = question_dir / "trace.json"
                with open(trace_file, 'w') as f:
                    json.dump(trace_data, f, indent=2)
            
            # Also append to summary JSONL for quick access
            summary_line = {
                "task_id": result.task_id,
                "success": result.success,
                "score": round(result.score, 3),
                "latency": round(result.latency, 2),
                "cost": round(result.cost, 6),
                "task_type": metadata.get("task_type"),
            }
            with open(self._output_dir / "results.jsonl", "a") as f:
                f.write(json.dumps(summary_line) + "\n")

    def run(
        self,
        limit: Optional[int] = None,
        save_results: bool = True,
    ) -> List[EvaluationResult]:
        """
        Run BFCL evaluation.
        
        Args:
            limit: Maximum number of tasks
            save_results: Whether to save results
        
        Returns:
            List of evaluation results
        """
        tasks = self.load_tasks(limit)
        
        # Reset counters
        self._success_count = 0
        self._total_count = 0
        
        print(f"\nRunning BFCL v3 with {len(tasks)} tasks...")
        print(f"Agent: {self.agent.__class__.__name__}")
        print(f"Model: {self.agent.model}")
        print(f"Concurrency: {self.concurrency}")
        
        # Setup incremental saving with per-question folders
        if save_results:
            if self.run_dir:
                # Use provided run directory (for grouped runs)
                base_dir = self.run_dir
            else:
                # Create timestamped run folder
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                base_dir = Path("results") / "bfcl" / f"{self.agent.model}_{timestamp}"
            
            # Create agent-specific subfolder for per-question traces
            self._output_dir = base_dir / self.agent.__class__.__name__
            self._output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving to: {self._output_dir}/\n")
        
        if self.concurrency > 1:
            results = self._run_concurrent(tasks)
        else:
            results = self._run_sequential(tasks)
        
        success_rate = self._success_count / len(tasks) if tasks else 0.0
        print(f"\nFinal Success Rate: {success_rate * 100:.1f}% ({self._success_count}/{len(tasks)})")
        
        if save_results:
            self._save_summary(results, success_rate)
        
        return results
    
    def _run_sequential(self, tasks: List[Dict[str, Any]]) -> List[EvaluationResult]:
        """Run tasks sequentially."""
        results = []
        
        pbar = tqdm(
            tasks,
            desc="Running",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] Success: {postfix}',
            postfix="---%"
        )
        
        for task in pbar:
            result = self._process_task(task)
            results.append(result)
            
            self._total_count += 1
            if result.success:
                self._success_count += 1
            rate = (self._success_count / self._total_count * 100) if self._total_count > 0 else 0
            
            self._save_result_incremental(result)
            pbar.set_postfix_str(f"{rate:.1f}%")
            
            if self.verbose:
                status = "✓" if result.success else "✗"
                tqdm.write(f"  {result.task_id}: {status} ({result.latency:.1f}s)")
        
        return results
    
    def _run_concurrent(self, tasks: List[Dict[str, Any]]) -> List[EvaluationResult]:
        """Run tasks concurrently with adaptive rate limiting."""
        results = [None] * len(tasks)
        
        # Create adaptive rate limiter
        limiter = ThreadSafeAdaptiveLimiter(
            max_concurrency=self.concurrency,
            min_concurrency=1,
            backoff_factor=0.5,
            recovery_threshold=10,
        )
        
        def process_with_limiter(task):
            """Process task with rate limit handling."""
            wait_time = limiter.wait_if_needed()
            if wait_time > 0 and self.verbose:
                tqdm.write(f"  ⏳ Rate limit backoff: {wait_time:.1f}s")
            
            try:
                result = self._process_task(task)
                
                # Check for rate limit errors in result
                if result.metadata and "error" in str(result.metadata.get("reasoning", "")):
                    limiter.record_error(backoff_seconds=2.0)
                else:
                    limiter.record_success()
                
                return result
                
            except Exception as e:
                error_str = str(e).lower()
                if "rate" in error_str or "429" in error_str or "limit" in error_str:
                    backoff = limiter.record_error(backoff_seconds=5.0)
                    if self.verbose:
                        tqdm.write(f"  ⚠️ Rate limit hit, reducing to {limiter.current_concurrency}x, backoff {backoff:.1f}s")
                raise
        
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            future_to_idx = {
                executor.submit(process_with_limiter, task): i
                for i, task in enumerate(tasks)
            }
            
            pbar = tqdm(
                total=len(tasks),
                desc=f"Running ({self.concurrency}x)",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] Success: {postfix}',
                postfix="---%"
            )
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results[idx] = result
                    
                    with self._lock:
                        self._total_count += 1
                        if result.success:
                            self._success_count += 1
                        rate = (self._success_count / self._total_count * 100) if self._total_count > 0 else 0
                    
                    self._save_result_incremental(result)
                    
                    # Update progress bar with current concurrency
                    pbar.set_description(f"Running ({limiter.current_concurrency}x)")
                    pbar.set_postfix_str(f"{rate:.1f}%")
                    pbar.update(1)
                    
                    if self.verbose:
                        status = "✓" if result.success else "✗"
                        tqdm.write(f"  {result.task_id}: {status} ({result.latency:.1f}s)")
                        
                except Exception as e:
                    result = EvaluationResult(
                        task_id=tasks[idx]["task_id"],
                        prompt="",
                        agent_response=f"Error: {e}",
                        success=False,
                        score=0.0,
                        latency=0.0,
                        cost=0.0,
                        metadata={"error": str(e)},
                    )
                    results[idx] = result
                    self._save_result_incremental(result)
                    
                    with self._lock:
                        self._total_count += 1
                        rate = (self._success_count / self._total_count * 100) if self._total_count > 0 else 0
                    
                    pbar.set_description(f"Running ({limiter.current_concurrency}x)")
                    pbar.set_postfix_str(f"{rate:.1f}%")
                    pbar.update(1)
                    tqdm.write(f"  {tasks[idx]['task_id']}: ✗ Error: {e}")
            
            pbar.close()
        
        return results
    
    def _save_summary(self, results: List[EvaluationResult], success_rate: float):
        """Save final summary to JSON file."""
        if not self._output_dir:
            return
        
        summary_file = self._output_dir / "summary.json"
        
        total_latency = sum(r.latency for r in results if r.latency)
        total_cost = sum(r.cost for r in results if r.cost)
        
        data = {
            "agent": self.agent.__class__.__name__,
            "model": self.agent.model,
            "benchmark": "BFCL v3",
            "success_rate": success_rate,
            "num_tasks": len(results),
            "total_latency_seconds": total_latency,
            "avg_latency_seconds": total_latency / len(results) if results else 0,
            "total_cost_usd": total_cost,
            "results_dir": str(self._output_dir),
            "num_traces": len(list(self._output_dir.glob("*/trace.json"))),
        }
        
        with open(summary_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Summary saved to: {summary_file}")
