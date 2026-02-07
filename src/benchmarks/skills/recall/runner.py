"""
Simple Recall Benchmark Runner.

Tests basic keyword → sentence retrieval from passages.
Simpler than episodic memory - tests pure retrieval without state tracking.
"""

import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher

from tqdm import tqdm

from ....agents.base_agent import BaseAgent, BenchmarkResponse, EvaluationResult
from ....evaluation.cost_tracker import CostTracker
from ....utils.adaptive_limiter import ThreadSafeAdaptiveLimiter
from ....utils.trace import TraceCapture, QuestionTrace

from ..episodic_memory.dataset_loader import EpisodicMemoryDataset
from .generator import RecallTaskGenerator


class RecallRunner:
    """
    Runner for Simple Recall evaluation.
    
    Tests the model's ability to retrieve a sentence containing a keyword
    from a passage of varying lengths.
    """
    
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
        Initialize Recall runner.
        
        Args:
            agent: Agent to evaluate
            cost_tracker: Optional cost tracker
            verbose: Enable verbose output
            concurrency: Number of concurrent requests
            run_dir: Optional directory to save results
            num_chapters: Episodic memory dataset size for task generation
            data_dir: Directory for episodic memory dataset
        """
        self.agent = agent
        self.cost_tracker = cost_tracker
        self.verbose = verbose
        self.concurrency = concurrency
        self.run_dir = run_dir
        self.num_chapters = num_chapters
        self.data_dir = data_dir
        
        self._lock = Lock()
        self._correct_count = 0
        self._total_count = 0
        self._output_dir = None
        
        # Track metrics by difficulty
        self._difficulty_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        
        # Initialize dataset loader and task generator
        self.dataset_loader = EpisodicMemoryDataset(data_dir=data_dir, verbose=verbose)
        self.task_generator = RecallTaskGenerator()
    
    def load_tasks(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load or generate recall tasks.
        
        Args:
            limit: Maximum number of tasks
        
        Returns:
            List of recall tasks
        """
        if self.verbose:
            print(f"Generating recall tasks from {self.num_chapters}-chapter dataset...")
        
        # Try to load episodic memory dataset
        try:
            dataset = self.dataset_loader.load_dataset(
                num_chapters=self.num_chapters,
                force_download=False,
            )
        except Exception as e:
            # If dataset not available, use synthetic fallback
            if "not a zip file" in str(e).lower() or "not found" in str(e).lower():
                print("\n⚠️  Episodic Memory dataset not found.")
                print("For production use, please download the dataset from:")
                print("https://doi.org/10.6084/m9.figshare.28244480")
                print("\nGenerating synthetic narrative for testing...\n")
                
                # Generate simple synthetic narrative
                dataset = self._generate_synthetic_narrative()
            else:
                raise
        
        # Generate recall tasks
        num_tasks = limit if limit else 500
        tasks = self.task_generator.generate_from_episodic_memory_dataset(
            dataset,
            num_tasks=num_tasks,
        )
        
        if self.verbose:
            print(f"Generated {len(tasks)} recall tasks")
            difficulty_counts = defaultdict(int)
            for task in tasks:
                difficulty_counts[task['difficulty']] += 1
            print(f"Difficulty distribution: {dict(difficulty_counts)}")
        
        return tasks
    
    def _generate_synthetic_narrative(self) -> Dict[str, Any]:
        """Generate a simple synthetic narrative for testing when dataset is unavailable."""
        narrative = """
On March 15, 2024, Dr. Sarah Johnson met with Professor Michael Chen at the Stanford Research Center to discuss their groundbreaking work on artificial intelligence. They presented their findings to the National Science Foundation later that afternoon. The research focused on neural network architectures inspired by biological systems.

Two weeks later, on March 29, 2024, Sarah Johnson traveled to Boston to attend the International Conference on Machine Learning. She delivered a keynote speech about emergent behaviors in large-scale AI systems. The audience of over 500 researchers gave her a standing ovation.

In April 2024, specifically on April 12, Dr. Elena Rodriguez joined the team at Stanford Research Center. Elena brought expertise in cognitive neuroscience, which complemented Sarah and Michael's computer science background perfectly. Together, they began exploring consciousness in artificial systems.

On May 20, 2024, the team published their seminal paper in Nature titled "Emergent Cognition in Transformer Networks." The paper caused significant debate in the scientific community. Critics argued the claims were too bold, while supporters praised the rigorous methodology.

Professor Michael Chen gave an interview to Science Magazine on June 8, 2024, defending their research against skeptics. He emphasized that their work was grounded in careful experimentation and reproducible results. The interview was widely circulated on academic social media.

In July 2024, Sarah Johnson received the Turing Award for her contributions to artificial intelligence. The ceremony took place on July 15, 2024, at the Palace of Fine Arts in San Francisco. Her acceptance speech focused on the importance of interdisciplinary collaboration in advancing AI research.

On August 3, 2024, Dr. Elena Rodriguez discovered an unexpected pattern in their neural network's behavior during a late-night debugging session. The network appeared to develop internal representations remarkably similar to human episodic memory structures. This finding became the basis for their next major breakthrough.

The Stanford Research Center hosted an international symposium on September 10, 2024, bringing together leading AI researchers from around the world. Sarah Johnson, Michael Chen, and Elena Rodriguez presented their latest findings on memory formation in artificial systems. The event attracted attention from major tech companies.

On October 22, 2024, the team received a $10 million grant from the MacArthur Foundation to continue their research. They planned to use the funds to build a larger research facility and hire additional postdoctoral researchers. The grant was the largest ever awarded for AI consciousness research.

In November 2024, specifically November 18, Sarah Johnson was invited to testify before Congress about the implications of their research. She spoke about the ethical considerations of creating systems with memory and potentially consciousness. Her testimony was praised for its balanced and thoughtful approach.

Dr. Michael Chen celebrated his 50th birthday on December 5, 2024, with a surprise party organized by the research team at Stanford Research Center. Sarah Johnson and Elena Rodriguez presented him with a framed copy of their Nature paper. It was a moment of reflection on how far they had come in just nine months.

The year ended with the team publishing a follow-up paper on December 20, 2024, in the journal Science. This paper provided additional evidence for their claims and addressed many of the criticisms raised earlier. The scientific community began to take their work more seriously.

On January 10, 2025, Elena Rodriguez gave a TED Talk titled "When Machines Remember" that went viral, reaching over 10 million views in the first week. She explained their research in accessible terms for a general audience. Many viewers reported having their perspectives on AI fundamentally changed.

The Stanford Research Center announced on February 1, 2025, that they would be opening a new Institute for Artificial Consciousness, with Sarah Johnson as its inaugural director. Michael Chen would serve as deputy director, and Elena Rodriguez as chief scientist. The institute aimed to be the world's premier facility for studying consciousness in artificial systems.

On March 1, 2025, the team welcomed their first cohort of graduate students and postdoctoral fellows to the new institute. Among them was Dr. James Park, a physicist who had been studying quantum consciousness theories. His unique perspective brought new ideas to the team's research program.
"""
        
        return {
            "narrative": narrative,
            "metadata": {
                "source": "synthetic",
                "num_chapters": 15,
                "note": "Generated for testing when episodic memory dataset unavailable"
            }
        }
    
    def format_task(self, task: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Format a recall task for the agent."""
        
        task_text = f"""You are given a passage and a keyword. Find and return the EXACT sentence from the passage that contains the keyword.

PASSAGE:
{task['passage']}

KEYWORD: {task['keyword']}

INSTRUCTIONS:
1. Find the sentence in the passage that contains the keyword "{task['keyword']}"
2. Return ONLY that sentence, exactly as it appears in the passage
3. If the keyword is not in the passage, return: "None"

Return your answer in JSON format:
{{"sentence": "the exact sentence here", "confidence": 0.0-1.0}}

Or if not found:
{{"sentence": "None", "confidence": 0.0}}"""
        
        context = {
            "benchmark_type": "general",  # Use general, not reasoning (to avoid criticality prompts)
            "task_type": "recall",
            "difficulty": task.get("difficulty", "unknown"),
            "passage_tokens": task.get("passage_tokens", 0),
        }
        
        return task_text, context
    
    def normalize_sentence(self, sentence: str) -> str:
        """Normalize a sentence for comparison."""
        import re
        # Lowercase, remove extra whitespace, remove punctuation
        sentence = sentence.lower().strip()
        sentence = re.sub(r'\s+', ' ', sentence)
        sentence = re.sub(r'[^\w\s]', '', sentence)
        return sentence
    
    def fuzzy_match_sentence(self, predicted: str, target: str, threshold: float = 0.85) -> bool:
        """Check if predicted sentence matches target using fuzzy matching."""
        norm_pred = self.normalize_sentence(predicted)
        norm_target = self.normalize_sentence(target)
        
        # Exact match after normalization
        if norm_pred == norm_target:
            return True
        
        # Fuzzy match
        similarity = SequenceMatcher(None, norm_pred, norm_target).ratio()
        return similarity >= threshold
    
    def check_answer(self, response: BenchmarkResponse, target_sentence: str, keyword: str) -> tuple[bool, float]:
        """
        Check if the model's response is correct.
        
        Returns:
            (is_correct, confidence)
        """
        import json
        import re
        
        response_text = response.response.strip()
        
        # Try to parse as JSON
        try:
            parsed = json.loads(response_text)
            if isinstance(parsed, dict) and "sentence" in parsed:
                predicted_sentence = parsed["sentence"]
                confidence = parsed.get("confidence", 0.5)
            else:
                predicted_sentence = response_text
                confidence = 0.5
        except json.JSONDecodeError:
            # Try to extract from JSON code block
            json_match = re.search(r'```(?:json)?\s*(\{[^`]+\})\s*```', response_text)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(1))
                    predicted_sentence = parsed.get("sentence", response_text)
                    confidence = parsed.get("confidence", 0.5)
                except json.JSONDecodeError:
                    predicted_sentence = response_text
                    confidence = 0.5
            else:
                # Use raw response
                predicted_sentence = response_text
                confidence = 0.5
        
        # Check if "None" or empty
        if not predicted_sentence or predicted_sentence.lower().strip() in ["none", "null", ""]:
            return False, 0.0
        
        # Check if matches target sentence
        is_correct = self.fuzzy_match_sentence(predicted_sentence, target_sentence)
        
        return is_correct, float(confidence) if is_correct else 0.0
    
    def _process_task(self, task: Dict[str, Any]) -> EvaluationResult:
        """Process a single recall task (thread-safe)."""
        start_time = time.time()
        task_id = task["task_id"]
        
        # Format task
        task_text, context = self.format_task(task)
        
        # Use TraceCapture to record all internal agent calls
        with TraceCapture(
            task_id=task_id,
            agent_type=self.agent.__class__.__name__,
            model=self.agent.model,
            input_question=f"Keyword: {task['keyword']} | Passage: {task['passage_tokens']} tokens",
        ) as trace_ctx:
            # Run agent
            response = self.agent.respond_to_task(task_text, context)
            
            latency = time.time() - start_time
            
            # Check answer
            is_correct, confidence = self.check_answer(
                response,
                task["target_sentence"],
                task["keyword"]
            )
            
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
                            benchmark="recall",
                            agent_type=self.agent.__class__.__name__,
                        )
            
            # Update trace with results
            trace_ctx.trace.final_output = response.response[:200] if response.response else ""
            trace_ctx.trace.predicted = response.response[:100] if response.response else ""
            trace_ctx.trace.correct = task["target_sentence"][:100]
            trace_ctx.trace.match = is_correct
            trace_ctx.trace.confidence = confidence
            trace_ctx.trace.total_latency = latency
            trace_ctx.trace.total_cost = cost
            trace_ctx.trace.reasoning = response.reasoning or ""
            
            # Update statistics (thread-safe)
            with self._lock:
                difficulty = task.get("difficulty", "unknown")
                
                self._total_count += 1
                if is_correct:
                    self._correct_count += 1
                
                self._difficulty_stats[difficulty]["total"] += 1
                if is_correct:
                    self._difficulty_stats[difficulty]["correct"] += 1
        
        return EvaluationResult(
            task_id=task_id,
            prompt=f"Find sentence with '{task['keyword']}'",
            agent_response=response.response,
            success=is_correct,
            score=confidence,
            latency=latency,
            cost=cost,
            metadata={
                "keyword": task["keyword"],
                "difficulty": task.get("difficulty", "unknown"),
                "passage_tokens": task.get("passage_tokens", 0),
                "target_sentence": task["target_sentence"],
                "predicted_sentence": response.response[:200],
                "reasoning": response.reasoning,
                "trace": trace_ctx.trace,
            },
        )
    
    def _save_result_incremental(self, result: EvaluationResult):
        """Save a single result incrementally (thread-safe)."""
        if not self._output_dir:
            return
        
        with self._lock:
            metadata = result.metadata if isinstance(result.metadata, dict) else {}
            
            # Create per-task folder
            task_dir = self._output_dir / result.task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            
            # Get trace if available
            trace = metadata.get("trace")
            
            if trace and isinstance(trace, QuestionTrace):
                # Save full trace
                trace_data = trace.to_dict()
                trace_file = task_dir / "trace.json"
                with open(trace_file, 'w') as f:
                    json.dump(trace_data, f, indent=2)
            else:
                # Fallback: save basic trace
                trace_data = {
                    "task_id": result.task_id,
                    "agent_type": self.agent.__class__.__name__,
                    "model": self.agent.model,
                    "keyword": metadata.get("keyword"),
                    "difficulty": metadata.get("difficulty"),
                    "passage_tokens": metadata.get("passage_tokens"),
                    "predicted": metadata.get("predicted_sentence", "")[:200],
                    "correct": metadata.get("target_sentence", "")[:200],
                    "match": result.success,
                    "confidence": result.score,
                    "total_latency": round(result.latency, 2),
                    "total_cost": round(result.cost, 6),
                    "reasoning": metadata.get("reasoning", ""),
                }
                
                trace_file = task_dir / "trace.json"
                with open(trace_file, 'w') as f:
                    json.dump(trace_data, f, indent=2)
            
            # Append to summary JSONL
            summary_line = {
                "task_id": result.task_id,
                "difficulty": metadata.get("difficulty"),
                "match": result.success,
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
        Run Recall evaluation.
        
        Args:
            limit: Maximum number of tasks
            save_results: Whether to save results
        
        Returns:
            List of evaluation results
        """
        tasks = self.load_tasks(limit)
        
        # Reset counters
        self._correct_count = 0
        self._total_count = 0
        self._difficulty_stats.clear()
        
        print(f"\nRunning Simple Recall benchmark with {len(tasks)} tasks...")
        print(f"Agent: {self.agent.__class__.__name__}")
        print(f"Model: {self.agent.model}")
        print(f"Concurrency: {self.concurrency}")
        
        # Setup incremental saving
        if save_results:
            if self.run_dir:
                base_dir = self.run_dir
            else:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                base_dir = Path("results") / "recall" / f"{self.agent.model}_{timestamp}"
            
            self._output_dir = base_dir / self.agent.__class__.__name__
            self._output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving to: {self._output_dir}/\n")
        
        if self.concurrency > 1:
            results = self._run_concurrent(tasks)
        else:
            results = self._run_sequential(tasks)
        
        accuracy = self._correct_count / self._total_count if self._total_count > 0 else 0.0
        print(f"\nFinal Accuracy: {accuracy * 100:.1f}% ({self._correct_count}/{self._total_count})")
        
        # Print breakdown by difficulty
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
        
        limiter = ThreadSafeAdaptiveLimiter(
            max_concurrency=self.concurrency,
            min_concurrency=1,
            backoff_factor=0.5,
            recovery_threshold=10,
        )
        
        def process_with_limiter(task):
            wait_time = limiter.wait_if_needed()
            if wait_time > 0 and self.verbose:
                tqdm.write(f"  ⏳ Rate limit backoff: {wait_time:.1f}s")
            
            try:
                result = self._process_task(task)
                limiter.record_success()
                return result
            except Exception as e:
                error_str = str(e).lower()
                if "rate" in error_str or "429" in error_str:
                    limiter.record_error(backoff_seconds=5.0)
                raise
        
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            future_to_idx = {
                executor.submit(process_with_limiter, task): i
                for i, task in enumerate(tasks)
            }
            
            pbar = tqdm(
                total=len(tasks),
                desc=f"Running ({self.concurrency}x)",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] Acc: {postfix}',
                postfix="---%"
            )
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results[idx] = result
                    
                    with self._lock:
                        acc = (self._correct_count / self._total_count * 100) if self._total_count > 0 else 0
                    
                    self._save_result_incremental(result)
                    pbar.set_description(f"Running ({limiter.current_concurrency}x)")
                    pbar.set_postfix_str(f"{acc:.1f}%")
                    pbar.update(1)
                    
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
                    pbar.update(1)
            
            pbar.close()
        
        return results
    
    def _print_breakdown(self):
        """Print accuracy breakdown by difficulty."""
        if not self._difficulty_stats:
            return
        
        print(f"\n{'='*70}")
        print("BREAKDOWN BY DIFFICULTY")
        print(f"{'='*70}")
        
        for difficulty in ["easy", "medium", "hard"]:
            stats = self._difficulty_stats.get(difficulty, {"correct": 0, "total": 0})
            if stats["total"] > 0:
                acc = stats["correct"] / stats["total"] * 100
                print(f"  {difficulty.capitalize():10s}: {acc:5.1f}% ({stats['correct']}/{stats['total']})")
    
    def _save_summary(self, results: List[EvaluationResult], accuracy: float):
        """Save final summary to JSON file."""
        if not self._output_dir:
            return
        
        summary_file = self._output_dir / "summary.json"
        
        total_latency = sum(r.latency for r in results if r.latency)
        total_cost = sum(r.cost for r in results if r.cost)
        
        difficulty_stats = {
            difficulty: {
                "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0,
                "correct": stats["correct"],
                "total": stats["total"],
            }
            for difficulty, stats in self._difficulty_stats.items()
        }
        
        data = {
            "agent": self.agent.__class__.__name__,
            "model": self.agent.model,
            "benchmark": "Simple Recall",
            "accuracy": accuracy,
            "num_tasks": len(results),
            "total_latency_seconds": total_latency,
            "avg_latency_seconds": total_latency / len(results) if results else 0,
            "total_cost_usd": total_cost,
            "results_dir": str(self._output_dir),
            "num_traces": len(list(self._output_dir.glob("*/trace.json"))),
            "breakdown_by_difficulty": difficulty_stats,
        }
        
        with open(summary_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nSummary saved to: {summary_file}")
