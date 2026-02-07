"""
MedQA/MedMCQA Benchmark Runner.

Traditional medical question-answering benchmarks:
- MedQA: USMLE-style questions (1,273 test questions)
- MedMCQA: Indian medical MCQs (4,183 test questions)

These test medical knowledge rather than agentic capabilities.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import List, Dict, Any, Optional

from datasets import load_dataset
from tqdm import tqdm

from ....agents.base_agent import BaseAgent, EvaluationResult
from ....evaluation.cost_tracker import CostTracker
from ....utils.adaptive_limiter import ThreadSafeAdaptiveLimiter
from ....utils.trace import TraceCapture, QuestionTrace


class MedQARunner:
    """
    Runner for MedQA and MedMCQA benchmarks.

    These are traditional multiple-choice question benchmarks
    that test medical knowledge.
    """

    def __init__(
        self,
        agent: BaseAgent,
        cost_tracker: Optional[CostTracker] = None,
        verbose: bool = False,
        dataset: str = "medqa",  # "medqa" or "medmcqa"
        concurrency: int = 1,
        run_dir: Optional[Path] = None,
    ):
        """
        Initialize MedQA runner.

        Args:
            agent: Agent to evaluate
            cost_tracker: Optional cost tracker
            verbose: Enable verbose output
            dataset: Which dataset to use ("medqa" or "medmcqa")
            concurrency: Number of concurrent requests (default: 1)
            run_dir: Optional directory to save results (for grouping runs)
        """
        self.agent = agent
        self.cost_tracker = cost_tracker
        self.verbose = verbose
        self.dataset = dataset
        self.concurrency = concurrency
        self.run_dir = run_dir
        self._lock = Lock()  # For thread-safe updates
        self._correct_count = 0  # Track correct answers
        self._total_count = 0  # Track total processed
        self._output_dir = None  # For per-question folder saving

    def load_tasks(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load MedQA/MedMCQA tasks from HuggingFace.

        Args:
            limit: Maximum number of questions to load

        Returns:
            List of question dicts
        """
        if self.dataset == "medqa":
            return self._load_medqa(limit)
        elif self.dataset == "medmcqa":
            return self._load_medmcqa(limit)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

    def _load_medqa(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load MedQA dataset (USMLE-style, 1,273 test questions)."""
        if self.verbose:
            print("Loading MedQA dataset from HuggingFace...")

        # Load the test split
        dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")

        questions = []
        for i, item in enumerate(dataset):
            if limit and i >= limit:
                break

            # Get options - the dataset has 'options' as a dict with A, B, C, D keys
            options = item.get("options", {})
            if not options:
                # Fallback for older format with option_a, option_b, etc.
                option_keys = ["A", "B", "C", "D"]
                for j, key in enumerate(["option_a", "option_b", "option_c", "option_d"]):
                    if key in item:
                        options[option_keys[j]] = item[key]

            # Get the correct answer - use answer_idx if available (it's the letter)
            answer_key = item.get("answer_idx", "")
            if not answer_key:
                # Fallback: match answer text to options
                answer_text = item.get("answer", "")
                answer_key = "A"  # default
                for key, value in options.items():
                    if value == answer_text:
                        answer_key = key
                        break

            questions.append(
                {
                    "question_id": f"medqa_{i + 1:04d}",
                    "question": item["question"],
                    "options": options,
                    "answer": answer_key,
                    "explanation": None,
                }
            )

        if self.verbose:
            print(f"Loaded {len(questions)} MedQA questions")

        return questions

    def _load_medmcqa(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load MedMCQA dataset (Indian medical MCQs, 4,183 test questions)."""
        if self.verbose:
            print("Loading MedMCQA dataset from HuggingFace...")

        # Load the test split (validation has labels, test doesn't)
        dataset = load_dataset("openlifescienceai/medmcqa", split="validation")

        questions = []
        for i, item in enumerate(dataset):
            if limit and i >= limit:
                break

            options = {
                "A": item["opa"],
                "B": item["opb"],
                "C": item["opc"],
                "D": item["opd"],
            }

            # cop is 0-indexed (0=A, 1=B, 2=C, 3=D)
            answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}
            answer_key = answer_map.get(item["cop"], "A")

            questions.append(
                {
                    "question_id": f"medmcqa_{i + 1:04d}",
                    "question": item["question"],
                    "options": options,
                    "answer": answer_key,
                    "explanation": item.get("exp"),
                }
            )

        if self.verbose:
            print(f"Loaded {len(questions)} MedMCQA questions")

        return questions

    def format_question(self, question: Dict[str, Any]) -> str:
        """Format question with options for the agent."""
        options_str = "\n".join([f"{k}. {v}" for k, v in question["options"].items()])
        return f"{question['question']}\n\nOptions:\n{options_str}\n\nAnswer with the letter (A, B, C, or D) and brief explanation."

    def extract_answer_letter(self, response: str) -> Optional[str]:
        """Extract the predicted answer letter (A, B, C, or D) from response."""
        response_upper = response.upper()

        # Look for patterns like "A.", "Answer: A", "A -", etc.
        for letter in ["A", "B", "C", "D"]:
            # Check if letter appears at start or after common prefixes
            patterns = [
                f"{letter}.",
                f"{letter} -",
                f"answer is {letter}",
                f"answer: {letter}",
                f"option {letter}",
            ]
            for pattern in patterns:
                if pattern in response_upper[:100]:  # Check first 100 chars
                    return letter

        # Fallback: find first letter A-D in response
        for char in response_upper[:200]:
            if char in ["A", "B", "C", "D"]:
                return char

        return None

    def parse_answer(self, response: str, correct_answer: str) -> tuple[bool, float]:
        """
        Parse agent's answer and check correctness.

        Returns:
            (is_correct, confidence)
        """
        predicted = self.extract_answer_letter(response)
        correct = correct_answer.upper()

        if not predicted:
            return False, 0.0

        if predicted == correct:
            # Check if it's clearly stated (not just mentioned)
            response_upper = response.upper()
            if f"answer is {predicted}" in response_upper or f"{predicted}." in response_upper[:50]:
                return True, 1.0
            return True, 0.8

        return False, 0.0

    def _process_question(self, question: Dict[str, Any], idx: int, total: int) -> EvaluationResult:
        """Process a single question (thread-safe)."""
        start_time = time.time()
        task_id = question["question_id"]

        # Format question
        formatted_task = self.format_question(question)

        # Use TraceCapture to record all internal agent calls
        with TraceCapture(
            task_id=task_id,
            agent_type=self.agent.__class__.__name__,
            model=self.agent.model,
            input_question=formatted_task,
        ) as trace_ctx:
            # Run agent
            context = {"benchmark_type": "medical"}
            response = self.agent.respond_to_task(formatted_task, context)
            
            latency = time.time() - start_time

            # Extract predicted answer letter
            predicted_answer = self.extract_answer_letter(response.response)

            # Check answer
            is_correct, confidence = self.parse_answer(response.response, question["answer"])

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
                            benchmark=self.dataset,
                            agent_type=self.agent.__class__.__name__,
                        )
            
            # Update trace with results
            trace_ctx.trace.final_output = response.response
            trace_ctx.trace.predicted = predicted_answer or ""
            trace_ctx.trace.correct = question["answer"]
            trace_ctx.trace.match = is_correct
            trace_ctx.trace.confidence = confidence
            trace_ctx.trace.total_latency = latency
            trace_ctx.trace.total_cost = cost
            trace_ctx.trace.reasoning = response.reasoning or ""
            
            # Add agent-specific metadata (pipeline_steps, discussion, etc.)
            if meta:
                for key in ["pipeline_steps", "discussion", "parallel_outputs"]:
                    if key in meta:
                        # Record these as additional trace data
                        for step in meta[key]:
                            if isinstance(step, dict) and "output" in step:
                                # Already recorded via TraceCapture.record() in agents
                                pass

        result = EvaluationResult(
            task_id=task_id,
            prompt=formatted_task,
            agent_response=response.response,
            success=is_correct,
            score=confidence,
            latency=latency,
            cost=cost,
            metadata={
                "correct_answer": question["answer"],
                "predicted_answer": predicted_answer,
                "explanation": question.get("explanation"),
                "reasoning": response.reasoning,
                "trace": trace_ctx.trace,  # Include full trace
            },
        )

        return result

    def run(
        self,
        limit: Optional[int] = None,
        save_results: bool = True,
    ) -> List[EvaluationResult]:
        """
        Run MedQA evaluation.

        Args:
            limit: Maximum number of questions
            save_results: Whether to save results

        Returns:
            List of evaluation results
        """
        questions = self.load_tasks(limit)
        results = []

        # Reset counters
        self._correct_count = 0
        self._total_count = 0

        print(f"\nRunning {self.dataset.upper()} with {len(questions)} questions...")
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
                base_dir = Path("results") / self.dataset / f"{self.agent.model}_{timestamp}"
            
            # Create agent-specific subfolder for per-question traces
            self._output_dir = base_dir / self.agent.__class__.__name__
            self._output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving to: {self._output_dir}/\n")

        if self.concurrency > 1:
            # Concurrent execution
            results = self._run_concurrent(questions)
        else:
            # Sequential execution
            results = self._run_sequential(questions)

        correct = sum(1 for r in results if r.success)
        accuracy = correct / len(questions) if questions else 0.0
        print(f"\nFinal Accuracy: {accuracy * 100:.1f}% ({correct}/{len(questions)})")

        # Save final summary
        if save_results:
            self._save_summary(results, accuracy)

        return results

    def _save_result_incremental(self, result: EvaluationResult):
        """Save a single result to per-question folder with full trace (thread-safe)."""
        if not self._output_dir:
            return

        with self._lock:
            metadata = result.metadata or {}
            
            # Create per-question folder
            question_dir = self._output_dir / result.task_id
            question_dir.mkdir(parents=True, exist_ok=True)
            
            # Get trace if available
            trace = metadata.get("trace")
            
            if trace and isinstance(trace, QuestionTrace):
                # Save full trace (includes all internal agent calls)
                trace.save(self._output_dir)
            else:
                # Fallback: save basic trace.json for simple agents
                trace_data = {
                    "task_id": result.task_id,
                    "agent_type": self.agent.__class__.__name__,
                    "model": self.agent.model,
                    "input_question": result.prompt[:2000] if result.prompt else "",
                    "calls": [],  # Simple agents don't have internal calls
                    "final_output": result.agent_response if result.agent_response else "",
                    "predicted": metadata.get("predicted_answer"),
                    "correct": metadata.get("correct_answer"),
                    "match": result.success,
                    "confidence": result.score,
                    "total_latency": round(result.latency, 2),
                    "total_cost": round(result.cost, 6),
                    "reasoning": metadata.get("reasoning", ""),
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
                "predicted": metadata.get("predicted_answer"),
                "correct": metadata.get("correct_answer"),
                "match": result.success,
                "confidence": result.score,
                "latency": round(result.latency, 2),
                "cost": round(result.cost, 6),
            }
            with open(self._output_dir / "results.jsonl", "a") as f:
                f.write(json.dumps(summary_line) + "\n")

    def _run_sequential(self, questions: List[Dict[str, Any]]) -> List[EvaluationResult]:
        """Run questions sequentially."""
        results = []

        pbar = tqdm(
            questions,
            desc="Running",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] Acc: {postfix}",
            postfix="---%",
        )

        for i, question in enumerate(pbar):
            result = self._process_question(question, i, len(questions))
            results.append(result)

            # Update counters
            self._total_count += 1
            if result.success:
                self._correct_count += 1
            acc = (self._correct_count / self._total_count * 100) if self._total_count > 0 else 0

            # Save incrementally
            self._save_result_incremental(result)

            # Update progress bar
            pbar.set_postfix_str(f"{acc:.1f}%")

            if self.verbose:
                status = "✓" if result.success else "✗"
                tqdm.write(f"  {question['question_id']}: {status} ({result.latency:.1f}s)")

        return results

    def _run_concurrent(self, questions: List[Dict[str, Any]]) -> List[EvaluationResult]:
        """Run questions concurrently using thread pool with adaptive rate limiting."""
        results = [None] * len(questions)  # Preserve order

        # Create adaptive rate limiter
        limiter = ThreadSafeAdaptiveLimiter(
            max_concurrency=self.concurrency,
            min_concurrency=1,
            backoff_factor=0.5,
            recovery_threshold=10,
        )

        def process_with_limiter(question, idx):
            """Process question with rate limit handling."""
            # Wait if we're in backoff period
            wait_time = limiter.wait_if_needed()
            if wait_time > 0 and self.verbose:
                tqdm.write(f"  ⏳ Rate limit backoff: {wait_time:.1f}s")

            try:
                result = self._process_question(question, idx, len(questions))

                # Check if result indicates a rate limit error
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
                        tqdm.write(
                            f"  ⚠️ Rate limit hit, reducing to {limiter.current_concurrency}x, backoff {backoff:.1f}s"
                        )
                raise

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(process_with_limiter, q, i): i for i, q in enumerate(questions)
            }

            # Collect results with progress bar
            pbar = tqdm(
                total=len(questions),
                desc=f"Running ({self.concurrency}x)",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] Acc: {postfix}",
                postfix="---%",
            )

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results[idx] = result

                    # Update counters
                    with self._lock:
                        self._total_count += 1
                        if result.success:
                            self._correct_count += 1
                        acc = (
                            (self._correct_count / self._total_count * 100)
                            if self._total_count > 0
                            else 0
                        )

                    # Save incrementally
                    self._save_result_incremental(result)

                    # Update progress bar with current concurrency
                    pbar.set_description(f"Running ({limiter.current_concurrency}x)")
                    pbar.set_postfix_str(f"{acc:.1f}%")
                    pbar.update(1)

                    if self.verbose:
                        status = "✓" if result.success else "✗"
                        tqdm.write(f"  {result.task_id}: {status} ({result.latency:.1f}s)")

                except Exception as e:
                    # Create failed result
                    result = EvaluationResult(
                        task_id=questions[idx]["question_id"],
                        prompt=self.format_question(questions[idx]),
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
                        acc = (
                            (self._correct_count / self._total_count * 100)
                            if self._total_count > 0
                            else 0
                        )

                    pbar.set_postfix_str(f"{acc:.1f}%")
                    pbar.update(1)
                    tqdm.write(f"  {questions[idx]['question_id']}: ✗ Error: {e}")

            pbar.close()

        return results

    def _save_summary(self, results: List[EvaluationResult], accuracy: float):
        """Save final summary to JSON file."""
        if not self._output_dir:
            return

        # Summary file in the agent's directory
        summary_file = self._output_dir / "summary.json"

        total_latency = sum(r.latency for r in results if r.latency)
        total_cost = sum(r.cost for r in results if r.cost)

        data = {
            "agent": self.agent.__class__.__name__,
            "model": self.agent.model,
            "benchmark": self.dataset.upper(),
            "accuracy": accuracy,
            "num_questions": len(results),
            "total_latency_seconds": total_latency,
            "avg_latency_seconds": total_latency / len(results) if results else 0,
            "total_cost_usd": total_cost,
            "results_dir": str(self._output_dir),
            "num_traces": len(list(self._output_dir.glob("*/trace.json"))),
        }

        with open(summary_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Summary saved to: {summary_file}")
