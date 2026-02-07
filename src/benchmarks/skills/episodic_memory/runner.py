"""
Episodic Memory (State Tracking) Benchmark Runner.

Tests long-context memory and entity state tracking capabilities.
Uses the Tulving Episodic Memory Benchmark (ICLR 2025).

Dataset: https://github.com/ahstat/episodic-memory-benchmark
Paper: https://arxiv.org/abs/2502.14802
"""

import json
import time
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

from .dataset_loader import EpisodicMemoryDataset
from .f1_evaluator import F1Evaluator


class EpisodicMemoryRunner:
    """
    Runner for Episodic Memory (State Tracking) evaluation.
    
    Tests the model's ability to track entity states and recall information
    from long narratives (10K-1M tokens).
    """
    
    # Question types for chronological awareness
    CHRONOLOGICAL_TYPES = [
        "time_chronological",
        "space_chronological",
        "content_chronological",
        "chronological_list",
        "time_latest",
        "space_latest",
        "latest",
    ]
    
    def __init__(
        self,
        agent: BaseAgent,
        cost_tracker: Optional[CostTracker] = None,
        verbose: bool = False,
        concurrency: int = 1,
        run_dir: Optional[Path] = None,
        num_chapters: int = 20,
        data_dir: str = "data/episodic_memory",
    ):
        """
        Initialize Episodic Memory runner.
        
        Args:
            agent: Agent to evaluate
            cost_tracker: Optional cost tracker
            verbose: Enable verbose output
            concurrency: Number of concurrent requests
            run_dir: Optional directory to save results (for grouping runs)
            num_chapters: Dataset size (20, 200, or 2000 chapters)
            data_dir: Directory for dataset storage
        """
        self.agent = agent
        self.cost_tracker = cost_tracker
        self.verbose = verbose
        self.concurrency = concurrency
        self.run_dir = run_dir
        self.num_chapters = num_chapters
        self.data_dir = data_dir
        
        self._lock = Lock()
        self._total_f1 = 0.0
        self._total_count = 0
        self._chronological_f1 = 0.0
        self._chronological_count = 0
        self._output_dir = None
        
        # Track metrics by question type
        self._type_stats = defaultdict(lambda: {"total_f1": 0.0, "count": 0})
        
        # Initialize evaluator and dataset loader
        self.evaluator = F1Evaluator(fuzzy_matching=True)
        self.dataset_loader = EpisodicMemoryDataset(data_dir=data_dir, verbose=verbose)
    
    def load_tasks(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Load episodic memory dataset.
        
        Args:
            limit: Maximum number of questions (None = all)
        
        Returns:
            Dictionary containing narrative, qa_pairs, and metadata
        """
        if self.verbose:
            print(f"Loading {self.num_chapters}-chapter dataset...")
        
        dataset = self.dataset_loader.load_dataset(
            num_chapters=self.num_chapters,
            force_download=False,
        )
        
        # Limit QA pairs if requested
        qa_pairs = dataset.get("qa_pairs", [])
        if limit and len(qa_pairs) > limit:
            qa_pairs = qa_pairs[:limit]
            dataset["qa_pairs"] = qa_pairs
        
        if self.verbose:
            print(f"Loaded {len(qa_pairs)} question-answer pairs")
            print(f"Narrative tokens: ~{dataset.get('num_tokens', 0):,}")
        
        return dataset
    
    def format_task(self, narrative: str, question: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Format a question with the full narrative context."""
        
        # Create task prompt with full narrative
        task_text = f"""You are given a long narrative document. Read it carefully and answer the question based ONLY on information in the narrative.

NARRATIVE:
{narrative}

QUESTION:
{question['question']}

Provide your answer as a list. Format your response as a JSON array of strings.
Example: ["item1", "item2", "item3"]

If the answer is a single item, still format as a list: ["item"]
If there is no answer in the narrative, return an empty list: []"""
        
        context = {
            "benchmark_type": "memory",
            "task_type": "episodic_memory",
            "question_type": question.get("type", "unknown"),
            "num_tokens": len(narrative) // 4,  # Rough estimate
        }
        
        return task_text, context
    
    def _process_task(self, task: Dict[str, Any], narrative: str) -> EvaluationResult:
        """Process a single question (thread-safe)."""
        start_time = time.time()
        task_id = task.get("id", f"q_{self._total_count}")
        
        # Format task
        task_text, context = self.format_task(narrative, task)
        
        # Use TraceCapture to record all internal agent calls
        with TraceCapture(
            task_id=task_id,
            agent_type=self.agent.__class__.__name__,
            model=self.agent.model,
            input_question=task_text[:1000] + "..." if len(task_text) > 1000 else task_text,
        ) as trace_ctx:
            # Run agent
            response = self.agent.respond_to_task(task_text, context)
            
            latency = time.time() - start_time
            
            # Evaluate using F1 score
            ground_truth = task.get("ground_truth", [])
            eval_result = self.evaluator.evaluate(response.response, ground_truth)
            
            f1_score = eval_result["f1"]
            precision = eval_result["precision"]
            recall = eval_result["recall"]
            
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
                            benchmark="episodic_memory",
                            agent_type=self.agent.__class__.__name__,
                        )
            
            # Update trace with results
            trace_ctx.trace.final_output = response.response
            trace_ctx.trace.predicted = str(eval_result["predicted_items"])
            trace_ctx.trace.correct = str(ground_truth)
            trace_ctx.trace.match = f1_score > 0.5  # Consider F1 > 0.5 as "match"
            trace_ctx.trace.confidence = f1_score
            trace_ctx.trace.total_latency = latency
            trace_ctx.trace.total_cost = cost
            trace_ctx.trace.reasoning = response.reasoning or ""
            
            # Update statistics (thread-safe)
            with self._lock:
                question_type = task.get("type", "unknown")
                
                self._total_f1 += f1_score
                self._total_count += 1
                
                # Track chronological awareness separately
                if question_type in self.CHRONOLOGICAL_TYPES:
                    self._chronological_f1 += f1_score
                    self._chronological_count += 1
                
                # Track by question type
                self._type_stats[question_type]["total_f1"] += f1_score
                self._type_stats[question_type]["count"] += 1
        
        return EvaluationResult(
            task_id=task_id,
            prompt=task_text[:1000] + "..." if len(task_text) > 1000 else task_text,
            agent_response=response.response,
            success=f1_score > 0.5,  # Threshold for "success"
            score=f1_score,
            latency=latency,
            cost=cost,
            metadata={
                "question": task.get("question", ""),
                "question_type": task.get("type", "unknown"),
                "ground_truth": ground_truth,
                "predicted_items": eval_result["predicted_items"],
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "reasoning": response.reasoning,
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
            question_dir = self._output_dir / result.task_id.replace("/", "_")
            question_dir.mkdir(parents=True, exist_ok=True)
            
            # Get trace if available
            trace = metadata.get("trace")
            
            if trace and isinstance(trace, QuestionTrace):
                # Save full trace
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
                    "question": metadata.get("question", "")[:500],
                    "question_type": metadata.get("question_type"),
                    "calls": [],
                    "final_output": result.agent_response[:500] if result.agent_response else "",
                    "predicted": metadata.get("predicted_items", []),
                    "correct": metadata.get("ground_truth", []),
                    "f1_score": metadata.get("f1_score", 0.0),
                    "precision": metadata.get("precision", 0.0),
                    "recall": metadata.get("recall", 0.0),
                    "total_latency": round(result.latency, 2),
                    "total_cost": round(result.cost, 6),
                    "reasoning": metadata.get("reasoning", ""),
                }
                
                trace_file = question_dir / "trace.json"
                with open(trace_file, 'w') as f:
                    json.dump(trace_data, f, indent=2)
            
            # Also append to summary JSONL for quick access
            summary_line = {
                "task_id": result.task_id,
                "question_type": metadata.get("question_type"),
                "f1_score": round(metadata.get("f1_score", 0.0), 3),
                "precision": round(metadata.get("precision", 0.0), 3),
                "recall": round(metadata.get("recall", 0.0), 3),
                "latency": round(result.latency, 2),
                "cost": round(result.cost, 6),
            }
            with open(self._output_dir / "results.jsonl", "a") as f:
                f.write(json.dumps(summary_line) + "\n")
    
    def run(
        self,
        limit: Optional[int] = None,
        save_results: bool = True,
    ) -> List[EvaluationResult]:
        """
        Run Episodic Memory evaluation.
        
        Args:
            limit: Maximum number of questions to evaluate
            save_results: Whether to save results
        
        Returns:
            List of evaluation results
        """
        # Load dataset
        dataset = self.load_tasks(limit)
        qa_pairs = dataset.get("qa_pairs", [])
        narrative = dataset.get("narrative", "")
        
        # Reset counters
        self._total_f1 = 0.0
        self._total_count = 0
        self._chronological_f1 = 0.0
        self._chronological_count = 0
        self._type_stats.clear()
        
        print(f"\nRunning Episodic Memory benchmark with {len(qa_pairs)} questions...")
        print(f"Dataset: {self.num_chapters} chapters (~{dataset.get('num_tokens', 0):,} tokens)")
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
                base_dir = Path("results") / "episodic_memory" / f"{self.agent.model}_{timestamp}"
            
            # Create agent-specific subfolder for per-question traces
            self._output_dir = base_dir / self.agent.__class__.__name__
            self._output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving to: {self._output_dir}/\n")
        
        if self.concurrency > 1:
            results = self._run_concurrent(qa_pairs, narrative)
        else:
            results = self._run_sequential(qa_pairs, narrative)
        
        simple_recall_score = self._total_f1 / self._total_count if self._total_count > 0 else 0.0
        chronological_score = self._chronological_f1 / self._chronological_count if self._chronological_count > 0 else 0.0
        
        print(f"\n{'='*70}")
        print(f"Simple Recall Score: {simple_recall_score:.3f}")
        print(f"Chronological Awareness Score: {chronological_score:.3f}")
        print(f"{'='*70}")
        
        # Print breakdown by question type
        self._print_breakdown()
        
        if save_results:
            self._save_summary(results, simple_recall_score, chronological_score)
        
        return results
    
    def _run_sequential(self, qa_pairs: List[Dict[str, Any]], narrative: str) -> List[EvaluationResult]:
        """Run questions sequentially."""
        results = []
        
        pbar = tqdm(
            qa_pairs,
            desc="Running",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] F1: {postfix}',
            postfix="---"
        )
        
        for qa in pbar:
            result = self._process_task(qa, narrative)
            results.append(result)
            
            avg_f1 = self._total_f1 / self._total_count if self._total_count > 0 else 0.0
            
            self._save_result_incremental(result)
            pbar.set_postfix_str(f"{avg_f1:.3f}")
            
            if self.verbose:
                tqdm.write(f"  {result.task_id}: F1={result.score:.3f} ({result.latency:.1f}s)")
        
        return results
    
    def _run_concurrent(self, qa_pairs: List[Dict[str, Any]], narrative: str) -> List[EvaluationResult]:
        """Run questions concurrently with adaptive rate limiting."""
        results = [None] * len(qa_pairs)
        
        # Create adaptive rate limiter
        limiter = ThreadSafeAdaptiveLimiter(
            max_concurrency=self.concurrency,
            min_concurrency=1,
            backoff_factor=0.5,
            recovery_threshold=10,
        )
        
        def process_with_limiter(qa):
            """Process question with rate limit handling."""
            wait_time = limiter.wait_if_needed()
            if wait_time > 0 and self.verbose:
                tqdm.write(f"  ⏳ Rate limit backoff: {wait_time:.1f}s")
            
            try:
                result = self._process_task(qa, narrative)
                limiter.record_success()
                return result
                
            except Exception as e:
                error_str = str(e).lower()
                if "rate" in error_str or "429" in error_str or "limit" in error_str:
                    backoff = limiter.record_error(backoff_seconds=5.0)
                    if self.verbose:
                        tqdm.write(f"  ⚠️ Rate limit hit, reducing to {limiter.current_concurrency}x")
                raise
        
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            future_to_idx = {
                executor.submit(process_with_limiter, qa): i
                for i, qa in enumerate(qa_pairs)
            }
            
            pbar = tqdm(
                total=len(qa_pairs),
                desc=f"Running ({self.concurrency}x)",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] F1: {postfix}',
                postfix="---"
            )
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results[idx] = result
                    
                    with self._lock:
                        avg_f1 = self._total_f1 / self._total_count if self._total_count > 0 else 0.0
                    
                    self._save_result_incremental(result)
                    
                    pbar.set_description(f"Running ({limiter.current_concurrency}x)")
                    pbar.set_postfix_str(f"{avg_f1:.3f}")
                    pbar.update(1)
                    
                    if self.verbose:
                        tqdm.write(f"  {result.task_id}: F1={result.score:.3f}")
                        
                except Exception as e:
                    result = EvaluationResult(
                        task_id=qa_pairs[idx].get("id", f"q_{idx}"),
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
                    
                    pbar.update(1)
                    tqdm.write(f"  {qa_pairs[idx].get('id', f'q_{idx}')}: ✗ Error: {e}")
            
            pbar.close()
        
        return results
    
    def _print_breakdown(self):
        """Print F1 scores breakdown by question type."""
        if not self._type_stats:
            return
        
        print(f"\n{'='*70}")
        print("BREAKDOWN BY QUESTION TYPE")
        print(f"{'='*70}")
        
        # Sort by count (most common first)
        sorted_types = sorted(
            self._type_stats.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        
        for q_type, stats in sorted_types[:15]:  # Top 15
            if stats['count'] > 0:
                avg_f1 = stats['total_f1'] / stats['count']
                is_chrono = " (chronological)" if q_type in self.CHRONOLOGICAL_TYPES else ""
                print(f"  {q_type:40s}{is_chrono:20s}: F1={avg_f1:.3f} (n={stats['count']})")
    
    def _save_summary(self, results: List[EvaluationResult], simple_recall: float, chronological: float):
        """Save final summary to JSON file."""
        if not self._output_dir:
            return
        
        summary_file = self._output_dir / "summary.json"
        
        total_latency = sum(r.latency for r in results if r.latency)
        total_cost = sum(r.cost for r in results if r.cost)
        
        # Convert type stats to regular dict
        type_stats = {
            q_type: {
                "avg_f1": stats['total_f1'] / stats['count'] if stats['count'] > 0 else 0.0,
                "count": stats['count'],
            }
            for q_type, stats in self._type_stats.items()
        }
        
        data = {
            "agent": self.agent.__class__.__name__,
            "model": self.agent.model,
            "benchmark": "Episodic Memory (State Tracking)",
            "num_chapters": self.num_chapters,
            "simple_recall_score": simple_recall,
            "chronological_awareness_score": chronological,
            "num_questions": len(results),
            "total_latency_seconds": total_latency,
            "avg_latency_seconds": total_latency / len(results) if results else 0,
            "total_cost_usd": total_cost,
            "results_dir": str(self._output_dir),
            "num_traces": len(list(self._output_dir.glob("*/trace.json"))),
            "breakdown_by_type": type_stats,
        }
        
        with open(summary_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nSummary saved to: {summary_file}")
