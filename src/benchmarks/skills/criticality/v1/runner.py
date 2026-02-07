"""
Criticality (Argument Quality) Benchmark Runner.

Tests a model's ability to assess argument quality through pairwise comparisons.
Uses the IBM Research Argument Quality Ranking 30k dataset.

Dataset: ibm-research/argument_quality_ranking_30k (~30k arguments)
Task: Compare two arguments on the same topic and identify which is stronger.
"""

import json
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import List, Dict, Any, Optional, Tuple

from datasets import load_dataset
from tqdm import tqdm

from .....agents.base_agent import BaseAgent, BenchmarkResponse, EvaluationResult
from .....evaluation.cost_tracker import CostTracker
from .....utils.adaptive_limiter import ThreadSafeAdaptiveLimiter
from .....utils.trace import TraceCapture, QuestionTrace


class CriticalityRunner:
    """
    Runner for Criticality (Argument Quality) evaluation.
    
    Tests the model's ability to assess argument quality by presenting
    pairs of arguments on the same topic and asking which is stronger.
    """
    
    def __init__(
        self,
        agent: BaseAgent,
        cost_tracker: Optional[CostTracker] = None,
        verbose: bool = False,
        concurrency: int = 1,
        run_dir: Optional[Path] = None,
        quality_score: str = "WA",
        min_quality_margin: float = 0.1,
    ):
        """
        Initialize Criticality runner.
        
        Args:
            agent: Agent to evaluate
            cost_tracker: Optional cost tracker
            verbose: Enable verbose output
            concurrency: Number of concurrent requests
            run_dir: Optional directory to save results (for grouping runs)
            quality_score: Which quality score to use ("WA" or "MACE-P")
            min_quality_margin: Minimum quality difference for pair generation
        """
        self.agent = agent
        self.cost_tracker = cost_tracker
        self.verbose = verbose
        self.concurrency = concurrency
        self.run_dir = run_dir
        self.quality_score = quality_score
        self.min_quality_margin = min_quality_margin
        self._lock = Lock()
        self._correct_count = 0
        self._total_count = 0
        self._output_dir = None
        
        # Track metrics by topic and margin
        self._topic_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        self._margin_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    
    def load_arguments(self, limit: Optional[int] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load arguments from HuggingFace and group by topic.
        
        Args:
            limit: Maximum number of arguments to load (will be used to generate ~limit/2 pairs)
        
        Returns:
            Dict mapping topic -> list of arguments
        """
        if self.verbose:
            print(f"Loading IBM Argument Quality dataset from HuggingFace...")
        
        # Use streaming mode to avoid downloading the full dataset
        ds = load_dataset(
            'ibm-research/argument_quality_ranking_30k',
            'argument_quality_ranking',
            split='train',
            streaming=True
        )
        
        # Group arguments by topic
        arguments_by_topic = defaultdict(list)
        loaded = 0
        
        for item in ds:
            if limit and loaded >= limit:
                break
            
            topic = item.get('topic', 'unknown')
            argument_text = item.get('argument', '')
            
            # Get quality score based on configuration
            if self.quality_score == "MACE-P":
                quality = item.get('MACE-P', 0.0)
            else:  # Default to WA
                quality = item.get('WA', 0.0)
            
            if argument_text and quality > 0:
                arguments_by_topic[topic].append({
                    'argument': argument_text,
                    'quality': float(quality),
                    'topic': topic,
                    'stance': item.get('stance_WA', 0),
                })
                loaded += 1
        
        if self.verbose:
            print(f"Loaded {loaded} arguments across {len(arguments_by_topic)} topics")
            print(f"Quality score used: {self.quality_score}")
        
        return dict(arguments_by_topic)
    
    def generate_pairs(
        self,
        arguments_by_topic: Dict[str, List[Dict[str, Any]]],
        num_pairs: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate pairwise comparison tasks.
        
        Args:
            arguments_by_topic: Arguments grouped by topic
            num_pairs: Target number of pairs to generate (None = all possible)
        
        Returns:
            List of pair comparison tasks
        """
        pairs = []
        
        for topic, arguments in arguments_by_topic.items():
            if len(arguments) < 2:
                continue
            
            # Sort by quality for efficient pairing
            sorted_args = sorted(arguments, key=lambda x: x['quality'])
            
            # Generate pairs with meaningful quality differences
            for i, arg1 in enumerate(sorted_args):
                for arg2 in sorted_args[i+1:]:
                    quality_diff = abs(arg2['quality'] - arg1['quality'])
                    
                    # Only include pairs with meaningful quality difference
                    if quality_diff >= self.min_quality_margin:
                        # Randomize which is shown as A vs B
                        if random.random() < 0.5:
                            arg_a, arg_b = arg1, arg2
                            correct_choice = "B" if arg2['quality'] > arg1['quality'] else "A"
                        else:
                            arg_a, arg_b = arg2, arg1
                            correct_choice = "A" if arg2['quality'] > arg1['quality'] else "B"
                        
                        pairs.append({
                            'pair_id': f"{topic}_{len(pairs)}",
                            'topic': topic,
                            'argument_a': arg_a['argument'],
                            'argument_b': arg_b['argument'],
                            'quality_a': arg_a['quality'],
                            'quality_b': arg_b['quality'],
                            'quality_diff': quality_diff,
                            'correct_choice': correct_choice,
                        })
                        
                        # Early exit if we have enough pairs
                        if num_pairs and len(pairs) >= num_pairs:
                            if self.verbose:
                                print(f"Generated {len(pairs)} pairwise comparison tasks")
                            return pairs
        
        # Shuffle to mix topics
        random.shuffle(pairs)
        
        # Limit to requested number
        if num_pairs:
            pairs = pairs[:num_pairs]
        
        if self.verbose:
            print(f"Generated {len(pairs)} pairwise comparison tasks")
            print(f"Min quality margin: {self.min_quality_margin}")
        
        return pairs
    
    def load_tasks(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load and prepare pairwise comparison tasks.
        
        Args:
            limit: Maximum number of comparison tasks
        
        Returns:
            List of pairwise comparison tasks
        """
        # Load arguments (load more than limit to ensure enough pairs)
        arg_limit = limit * 3 if limit else None
        arguments_by_topic = self.load_arguments(arg_limit)
        
        # Generate pairs
        pairs = self.generate_pairs(arguments_by_topic, num_pairs=limit)
        
        return pairs
    
    def format_task(self, pair: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Format a pairwise comparison task for the agent."""
        task_text = f"""Topic: {pair['topic']}

Compare these two arguments and determine which is stronger (more persuasive, better constructed, more compelling):

Argument A: {pair['argument_a']}

Argument B: {pair['argument_b']}

Which argument is stronger? Respond with either "A" or "B" and explain your reasoning."""
        
        context = {
            "benchmark_type": "reasoning",
            "task_type": "criticality",
            "topic": pair['topic'],
        }
        
        return task_text, context
    
    def extract_choice(self, response: str) -> Optional[str]:
        """Extract choice (A or B) from agent response."""
        response_upper = response.upper()
        
        # Look for explicit choice patterns
        patterns = [
            r'\b(A|B)\b',  # Single letter A or B
            r'ARGUMENT\s+(A|B)',
            r'CHOICE[:\s]+(A|B)',
            r'ANSWER[:\s]+(A|B)',
            r'STRONGER[:\s]+(A|B)',
        ]
        
        import re
        for pattern in patterns:
            match = re.search(pattern, response_upper)
            if match:
                choice = match.group(1) if match.lastindex else match.group(0)
                if choice in ['A', 'B']:
                    return choice
        
        # Check first 100 characters for A or B
        for char in response_upper[:100]:
            if char in ['A', 'B']:
                return char
        
        return None
    
    def check_answer(
        self,
        response: BenchmarkResponse,
        correct_choice: str,
        pair: Dict[str, Any]
    ) -> Tuple[bool, float]:
        """
        Check if the model's choice is correct.
        
        Returns:
            (is_correct, confidence)
        """
        predicted = self.extract_choice(response.response)
        
        if not predicted:
            return False, 0.0
        
        is_correct = (predicted == correct_choice)
        
        # Confidence based on quality difference (higher diff = easier task)
        quality_diff = pair['quality_diff']
        base_confidence = 1.0 if is_correct else 0.0
        
        # Adjust confidence based on task difficulty
        if quality_diff > 0.3:
            difficulty = "easy"
        elif quality_diff > 0.15:
            difficulty = "medium"
        else:
            difficulty = "hard"
        
        return is_correct, base_confidence
    
    def _process_task(self, pair: Dict[str, Any]) -> EvaluationResult:
        """Process a single pairwise comparison task (thread-safe)."""
        start_time = time.time()
        task_id = pair['pair_id']
        
        # Format task
        task_text, context = self.format_task(pair)
        
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
            
            # Extract predicted choice
            predicted_choice = self.extract_choice(response.response)
            
            # Check answer
            is_correct, confidence = self.check_answer(response, pair['correct_choice'], pair)
            
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
                            benchmark="criticality",
                            agent_type=self.agent.__class__.__name__,
                        )
            
            # Update trace with results
            trace_ctx.trace.final_output = response.response
            trace_ctx.trace.predicted = predicted_choice or ""
            trace_ctx.trace.correct = pair['correct_choice']
            trace_ctx.trace.match = is_correct
            trace_ctx.trace.confidence = confidence
            trace_ctx.trace.total_latency = latency
            trace_ctx.trace.total_cost = cost
            trace_ctx.trace.reasoning = response.reasoning or ""
            
            # Update topic and margin statistics (thread-safe)
            with self._lock:
                topic = pair['topic']
                self._topic_stats[topic]['total'] += 1
                if is_correct:
                    self._topic_stats[topic]['correct'] += 1
                
                # Bucket by quality margin
                quality_diff = pair['quality_diff']
                if quality_diff > 0.3:
                    margin_bucket = "easy (>0.3)"
                elif quality_diff > 0.15:
                    margin_bucket = "medium (0.15-0.3)"
                else:
                    margin_bucket = "hard (<0.15)"
                
                self._margin_stats[margin_bucket]['total'] += 1
                if is_correct:
                    self._margin_stats[margin_bucket]['correct'] += 1
        
        return EvaluationResult(
            task_id=task_id,
            prompt=task_text,
            agent_response=response.response,
            success=is_correct,
            score=confidence,
            latency=latency,
            cost=cost,
            metadata={
                'topic': pair['topic'],
                'predicted_choice': predicted_choice,
                'correct_choice': pair['correct_choice'],
                'quality_a': pair['quality_a'],
                'quality_b': pair['quality_b'],
                'quality_diff': pair['quality_diff'],
                'reasoning': response.reasoning,
                'trace': trace_ctx.trace,
            },
        )
    
    def _save_result_incremental(self, result: EvaluationResult):
        """Save a single result to per-task folder with full trace (thread-safe)."""
        if not self._output_dir:
            return
        
        with self._lock:
            metadata = result.metadata if isinstance(result.metadata, dict) else {}
            
            # Create per-task folder
            task_dir = self._output_dir / result.task_id.replace("/", "_")
            task_dir.mkdir(parents=True, exist_ok=True)
            
            # Get trace if available
            trace = metadata.get("trace")
            
            if trace and isinstance(trace, QuestionTrace):
                # Save full trace (includes all internal agent calls)
                trace_data = trace.to_dict()
                trace_file = task_dir / "trace.json"
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
                    "predicted": metadata.get("predicted_choice"),
                    "correct": metadata.get("correct_choice"),
                    "match": result.success,
                    "confidence": result.score,
                    "total_latency": round(result.latency, 2),
                    "total_cost": round(result.cost, 6),
                    "reasoning": metadata.get("reasoning", ""),
                    "topic": metadata.get("topic"),
                    "quality_diff": metadata.get("quality_diff"),
                }
                
                trace_file = task_dir / "trace.json"
                with open(trace_file, 'w') as f:
                    json.dump(trace_data, f, indent=2)
            
            # Also append to summary JSONL for quick access
            summary_line = {
                "task_id": result.task_id,
                "topic": metadata.get("topic"),
                "predicted": metadata.get("predicted_choice"),
                "correct": metadata.get("correct_choice"),
                "match": result.success,
                "quality_diff": round(metadata.get("quality_diff", 0), 3),
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
        Run Criticality evaluation.
        
        Args:
            limit: Maximum number of pairwise comparison tasks
            save_results: Whether to save results
        
        Returns:
            List of evaluation results
        """
        tasks = self.load_tasks(limit)
        
        # Reset counters
        self._correct_count = 0
        self._total_count = 0
        self._topic_stats.clear()
        self._margin_stats.clear()
        
        print(f"\nRunning Criticality benchmark with {len(tasks)} pairwise comparisons...")
        print(f"Agent: {self.agent.__class__.__name__}")
        print(f"Model: {self.agent.model}")
        print(f"Concurrency: {self.concurrency}")
        print(f"Quality score: {self.quality_score}")
        
        # Setup incremental saving with per-task folders
        if save_results:
            if self.run_dir:
                # Use provided run directory (for grouped runs)
                base_dir = self.run_dir
            else:
                # Create timestamped run folder
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                base_dir = Path("results") / "criticality" / f"{self.agent.model}_{timestamp}"
            
            # Create agent-specific subfolder for per-task traces
            self._output_dir = base_dir / self.agent.__class__.__name__
            self._output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving to: {self._output_dir}/\n")
        
        if self.concurrency > 1:
            results = self._run_concurrent(tasks)
        else:
            results = self._run_sequential(tasks)
        
        accuracy = self._correct_count / len(tasks) if tasks else 0.0
        print(f"\nFinal Accuracy: {accuracy * 100:.1f}% ({self._correct_count}/{len(tasks)})")
        
        # Print breakdown by topic and margin
        self._print_breakdown()
        
        if save_results:
            self._save_summary(results, accuracy)
        
        return results
    
    def _run_sequential(self, tasks: List[Dict[str, Any]]) -> List[EvaluationResult]:
        """Run tasks sequentially."""
        results = []
        
        pbar = tqdm(
            tasks,
            desc="Running",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] Acc: {postfix}',
            postfix="---%"
        )
        
        for task in pbar:
            result = self._process_task(task)
            results.append(result)
            
            self._total_count += 1
            if result.success:
                self._correct_count += 1
            acc = (self._correct_count / self._total_count * 100) if self._total_count > 0 else 0
            
            self._save_result_incremental(result)
            pbar.set_postfix_str(f"{acc:.1f}%")
            
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
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] Acc: {postfix}',
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
                            self._correct_count += 1
                        acc = (self._correct_count / self._total_count * 100) if self._total_count > 0 else 0
                    
                    self._save_result_incremental(result)
                    
                    # Update progress bar with current concurrency
                    pbar.set_description(f"Running ({limiter.current_concurrency}x)")
                    pbar.set_postfix_str(f"{acc:.1f}%")
                    pbar.update(1)
                    
                    if self.verbose:
                        status = "✓" if result.success else "✗"
                        tqdm.write(f"  {result.task_id}: {status} ({result.latency:.1f}s)")
                        
                except Exception as e:
                    result = EvaluationResult(
                        task_id=tasks[idx]["pair_id"],
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
                        acc = (self._correct_count / self._total_count * 100) if self._total_count > 0 else 0
                    
                    pbar.set_description(f"Running ({limiter.current_concurrency}x)")
                    pbar.set_postfix_str(f"{acc:.1f}%")
                    pbar.update(1)
                    tqdm.write(f"  {tasks[idx]['pair_id']}: ✗ Error: {e}")
            
            pbar.close()
        
        return results
    
    def _print_breakdown(self):
        """Print accuracy breakdown by topic and margin."""
        if not self._topic_stats and not self._margin_stats:
            return
        
        print("\n" + "=" * 70)
        print("BREAKDOWN BY DIFFICULTY (Quality Margin)")
        print("=" * 70)
        for margin, stats in sorted(self._margin_stats.items()):
            if stats['total'] > 0:
                acc = stats['correct'] / stats['total'] * 100
                print(f"  {margin:20s}: {acc:5.1f}% ({stats['correct']}/{stats['total']})")
        
        print("\n" + "=" * 70)
        print("BREAKDOWN BY TOPIC (Top 10)")
        print("=" * 70)
        # Sort by total count, show top 10
        sorted_topics = sorted(
            self._topic_stats.items(),
            key=lambda x: x[1]['total'],
            reverse=True
        )[:10]
        for topic, stats in sorted_topics:
            if stats['total'] > 0:
                acc = stats['correct'] / stats['total'] * 100
                topic_short = topic[:40] + "..." if len(topic) > 40 else topic
                print(f"  {topic_short:45s}: {acc:5.1f}% ({stats['correct']}/{stats['total']})")
    
    def _save_summary(self, results: List[EvaluationResult], accuracy: float):
        """Save final summary to JSON file."""
        if not self._output_dir:
            return
        
        summary_file = self._output_dir / "summary.json"
        
        total_latency = sum(r.latency for r in results if r.latency)
        total_cost = sum(r.cost for r in results if r.cost)
        
        # Convert defaultdict to regular dict for JSON serialization
        topic_stats = {
            topic: {
                "accuracy": stats['correct'] / stats['total'] if stats['total'] > 0 else 0,
                "correct": stats['correct'],
                "total": stats['total'],
            }
            for topic, stats in self._topic_stats.items()
        }
        
        margin_stats = {
            margin: {
                "accuracy": stats['correct'] / stats['total'] if stats['total'] > 0 else 0,
                "correct": stats['correct'],
                "total": stats['total'],
            }
            for margin, stats in self._margin_stats.items()
        }
        
        data = {
            "agent": self.agent.__class__.__name__,
            "model": self.agent.model,
            "benchmark": "Criticality (Argument Quality)",
            "quality_score_used": self.quality_score,
            "min_quality_margin": self.min_quality_margin,
            "accuracy": accuracy,
            "num_tasks": len(results),
            "total_latency_seconds": total_latency,
            "avg_latency_seconds": total_latency / len(results) if results else 0,
            "total_cost_usd": total_cost,
            "results_dir": str(self._output_dir),
            "num_traces": len(list(self._output_dir.glob("*/trace.json"))),
            "breakdown_by_topic": topic_stats,
            "breakdown_by_margin": margin_stats,
        }
        
        with open(summary_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nSummary saved to: {summary_file}")
