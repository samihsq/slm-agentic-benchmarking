"""
Criticality v2 Runner: Logit-Based Argument Evaluation.

Uses logprobs to measure how strongly a model commits to its evaluation of
argument quality. Instead of just checking accuracy, we analyze the probability
distribution over candidate responses (A/B/C/D) to understand calibration,
discrimination, and robustness.

Supports:
  - Ollama models via OpenAI-compatible endpoint (DASD-4B, etc.)
  - Azure Foundry models (GPT-4o, Llama-3.3-70B) via OpenAI SDK
  - Fallback to accuracy-only for models without logprob support

See docs/criticality/PLAN.md for full design.
"""

import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Dict, List, Any, Optional

from openai import OpenAI
from tqdm import tqdm

from .task_generator import MCQTaskGenerator
from .logprob_utils import (
    LogprobExtractor,
    ChoiceLogprobs,
    CalibrationMetrics,
    compute_calibration_metrics,
)

try:
    from .sequence_scorer import SequenceScorer
    SEQUENCE_SCORER_AVAILABLE = True
except ImportError:
    SEQUENCE_SCORER_AVAILABLE = False
    SequenceScorer = None


# ── Model backend configuration ───────────────────────────────────────────

# Models that support logprobs and their API endpoints
OLLAMA_MODELS = {
    "hf.co/mradermacher/DASD-4B-Thinking-GGUF:Q4_K_M": {
        "thinking_model": True,
        "label": "DASD-4B",
    },
    "qwen3:0.6b": {
        "thinking_model": True,
        "label": "Qwen3-0.6B",
    },
    "falcon-h1:0.09b": {
        "thinking_model": False,
        "label": "Falcon-H1-90M",
    },
}

AZURE_LOGPROB_MODELS = {"gpt-4o", "llama-3.3-70b"}


class CriticalityV2Runner:
    """
    Runner for Criticality v2 (logprob-based argument evaluation).

    Supports two modes:
      - "sequence": Direct GGUF scoring via llama-cpp-python (works on any model)
      - "api": OpenAI-compatible API with logprob extraction (for hosted models)

    Phase 1: MCQ Logprob Probing
      - Present 4-option argument quality MCQs
      - Extract logprobs for choice tokens (A/B/C/D) or score sequences
      - Compare probability ranking to ground truth quality ranking

    Phase 2: Freeform Refutation Probing (stretch)
    Phase 3: Consistency Under Perturbation (stretch)
    """

    def __init__(
        self,
        model: str,
        model_path: Optional[str] = None,
        mode: Optional[str] = None,
        ollama_base_url: str = "http://10.27.102.240:11434",
        azure_api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        cost_tracker=None,
        verbose: bool = False,
        concurrency: int = 1,
        run_dir: Optional[Path] = None,
        quality_score: str = "WA",
        top_logprobs: int = 10,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,
    ):
        """
        Args:
            model: Model identifier (Ollama model name or Azure model name).
            model_path: Path to GGUF file for sequence mode (e.g., Ollama blob path).
            mode: "sequence" (default if model_path provided) or "api".
            ollama_base_url: Ollama server URL (for API mode).
            azure_api_key: Azure API key (for Azure models in API mode).
            azure_endpoint: Azure endpoint URL.
            cost_tracker: Optional cost tracker.
            verbose: Print progress info.
            concurrency: Number of concurrent requests.
            run_dir: Directory to save results.
            quality_score: Quality column to use ("WA" or "MACE-P").
            top_logprobs: Number of top logprobs to request (API mode only).
            n_ctx: Context window size (sequence mode only).
            n_gpu_layers: GPU layers to offload (sequence mode only, -1 = all).
        """
        self.model = model
        self.model_path = model_path
        self.verbose = verbose
        self.concurrency = concurrency
        self.run_dir = run_dir
        self.quality_score = quality_score
        self.top_logprobs = top_logprobs
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers

        self._lock = Lock()
        self._output_dir: Optional[Path] = None

        # Determine mode: sequence (local GGUF) or api (OpenAI-compatible endpoint)
        if mode is not None:
            self._mode = mode
        elif model_path is not None:
            self._mode = "sequence"
        else:
            self._mode = "api"

        if self._mode == "sequence" and not SEQUENCE_SCORER_AVAILABLE:
            raise ImportError(
                "llama-cpp-python is required for sequence mode. "
                "Install with: pip install llama-cpp-python"
            )
        
        if self._mode == "sequence" and not model_path:
            raise ValueError("model_path is required for sequence mode")

        # Backend setup
        self._is_ollama = model in OLLAMA_MODELS or ollama_base_url
        self._is_thinking = OLLAMA_MODELS.get(model, {}).get("thinking_model", False)
        self._model_label = OLLAMA_MODELS.get(model, {}).get("label", model)

        # Sequence scorer (lazy-loaded in run())
        self._scorer: Optional[SequenceScorer] = None

        # API client (only for API mode)
        if self._mode == "api":
            if model in AZURE_LOGPROB_MODELS and azure_api_key:
                self._client = OpenAI(
                    api_key=azure_api_key,
                    base_url=azure_endpoint,
                )
                self._is_ollama = False
            else:
                # Default to Ollama OpenAI-compatible endpoint
                self._client = OpenAI(
                    api_key="ollama",  # Ollama doesn't need a real key
                    base_url=f"{ollama_base_url}/v1",
                )

            # Logprob extractor (API mode only)
            self._extractor = LogprobExtractor(thinking_model=self._is_thinking)
        else:
            self._client = None
            self._extractor = None

        # Task generator
        self._task_gen = MCQTaskGenerator(
            quality_score=quality_score,
            verbose=verbose,
        )

        # Cost tracker
        self.cost_tracker = cost_tracker

        # Running stats
        self._results: List[Dict[str, Any]] = []
        self._top1_correct = 0
        self._total = 0

    # ── API calls ──────────────────────────────────────────────────────────

    def _call_model(self, prompt: str, system_prompt: str) -> Dict[str, Any]:
        """
        Call the model with logprobs enabled.

        Returns:
            Dict with keys: response, choice_logprobs, raw_response, latency, tokens
        """
        start = time.time()

        try:
            # Thinking models (e.g. DASD-4B) need room for their internal
            # reasoning chain before emitting the answer token.  Non-thinking
            # models just output a single choice letter directly.
            max_tokens = 300 if self._is_thinking else 1

            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                logprobs=True,
                top_logprobs=self.top_logprobs,
                max_tokens=max_tokens,
                temperature=0.0,  # Deterministic for logprob extraction
            )

            latency = time.time() - start

            # Extract the response text
            content = response.choices[0].message.content or ""

            # Extract choice logprobs
            choice_lps = self._extractor.extract_from_response(response)

            # Token usage
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0

            return {
                "content": content.strip(),
                "choice_logprobs": choice_lps,
                "latency": latency,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "raw_logprobs": self._serialize_logprobs(response),
            }

        except Exception as e:
            latency = time.time() - start
            if self.verbose:
                print(f"  API error: {e}")
            return {
                "content": "",
                "choice_logprobs": None,
                "latency": latency,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "error": str(e),
                "raw_logprobs": None,
            }

    def _serialize_logprobs(self, response) -> Optional[List[Dict]]:
        """Serialize raw logprobs from response for storage."""
        try:
            lp_data = response.choices[0].logprobs
            if lp_data is None or not lp_data.content:
                return None

            serialized = []
            for tok in lp_data.content:
                entry = {
                    "token": tok.token,
                    "logprob": tok.logprob,
                }
                if tok.top_logprobs:
                    entry["top_logprobs"] = [
                        {"token": alt.token, "logprob": alt.logprob}
                        for alt in tok.top_logprobs
                    ]
                serialized.append(entry)

            return serialized
        except (AttributeError, IndexError):
            return None

    # ── Task processing ────────────────────────────────────────────────────

    def _process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single MCQ task: score options or call API.

        Returns:
            Result dict with logprobs, metrics, and trace data.
        """
        task_id = task["task_id"]
        
        if self._mode == "sequence":
            # Sequence scoring mode
            start = time.time()
            
            # Build prefix for sequence scoring
            prefix = f'Topic: "{task["topic"]}". The strongest argument is: '
            
            # Extract option texts and labels
            options = [opt["argument"] for opt in task["options"]]
            labels = [opt["label"] for opt in task["options"]]
            
            try:
                # Score all options
                choice_lps = self._scorer.score_options(prefix, options, labels)
                latency = time.time() - start
                content = ""
                prompt_tokens = 0
                completion_tokens = 0
                error = None
            except Exception as e:
                latency = time.time() - start
                choice_lps = None
                content = ""
                prompt_tokens = 0
                completion_tokens = 0
                error = str(e)
                if self.verbose:
                    print(f"  Sequence scoring error: {e}")
        else:
            # API mode
            prompt = MCQTaskGenerator.format_prompt(task, system_constrained=True)
            system_prompt = MCQTaskGenerator.get_system_prompt()

            # Call model
            api_result = self._call_model(prompt, system_prompt)

            choice_lps = api_result["choice_logprobs"]
            content = api_result["content"]
            latency = api_result["latency"]
            prompt_tokens = api_result["prompt_tokens"]
            completion_tokens = api_result["completion_tokens"]
            error = api_result.get("error")

        # Determine predicted choice
        if choice_lps and choice_lps.top_choice:
            predicted = choice_lps.top_choice
        elif content and content.strip().upper() in {"A", "B", "C", "D"}:
            predicted = content.strip().upper()
        else:
            predicted = None

        # Check correctness
        correct_label = task["correct_label"]
        is_correct = predicted == correct_label

        # Quality spread for difficulty bucketing
        qualities = list(task["ground_truth_qualities"].values())
        quality_spread = max(qualities) - min(qualities) if qualities else 0.0

        # Build result
        result = {
            "task_id": task_id,
            "topic": task["topic"],
            "predicted": predicted,
            "correct_label": correct_label,
            "is_correct": is_correct,
            "choice_logprobs": choice_lps,
            "ground_truth_ranking": task["ground_truth_ranking"],
            "ground_truth_qualities": task["ground_truth_qualities"],
            "quality_spread": quality_spread,
            "content": content,
            "latency": latency,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "error": error,
            "raw_logprobs": None if self._mode == "sequence" else None,  # Only API mode has raw_logprobs
        }

        # Add logprob-specific metrics
        if choice_lps:
            result["margin"] = choice_lps.margin
            result["entropy"] = choice_lps.entropy
            result["probabilities"] = choice_lps.probabilities
            result["logprobs"] = choice_lps.logprobs
            result["extraction_source"] = choice_lps.source
            result["extraction_position"] = choice_lps.position
        else:
            result["margin"] = None
            result["entropy"] = None
            result["probabilities"] = {}
            result["logprobs"] = {}
            result["extraction_source"] = "none"
            result["extraction_position"] = -1

        # Update running stats (thread-safe)
        with self._lock:
            self._total += 1
            if is_correct:
                self._top1_correct += 1

        return result

    def _save_result(self, result: Dict[str, Any]):
        """Save a single result incrementally (thread-safe)."""
        if not self._output_dir:
            return

        with self._lock:
            # Per-task folder
            task_dir = self._output_dir / result["task_id"]
            task_dir.mkdir(parents=True, exist_ok=True)

            # Serialize ChoiceLogprobs
            trace = dict(result)
            if trace.get("choice_logprobs") is not None:
                cl = trace["choice_logprobs"]
                trace["choice_logprobs"] = {
                    "logprobs": cl.logprobs,
                    "probabilities": cl.probabilities,
                    "position": cl.position,
                    "source": cl.source,
                    "top_choice": cl.top_choice,
                    "margin": cl.margin,
                    "entropy": cl.entropy,
                }
            else:
                trace["choice_logprobs"] = None

            with open(task_dir / "trace.json", "w") as f:
                json.dump(trace, f, indent=2, default=str)

            # Append to summary JSONL
            summary_line = {
                "task_id": result["task_id"],
                "predicted": result["predicted"],
                "correct": result["correct_label"],
                "is_correct": result["is_correct"],
                "margin": result.get("margin"),
                "entropy": result.get("entropy"),
                "latency": round(result["latency"], 2),
                "extraction_source": result.get("extraction_source", ""),
            }
            with open(self._output_dir / "results.jsonl", "a") as f:
                f.write(json.dumps(summary_line) + "\n")

    # ── Main run ───────────────────────────────────────────────────────────

    def run(
        self,
        limit: Optional[int] = None,
        save_results: bool = True,
        include_shuffled: bool = False,
    ) -> Dict[str, Any]:
        """
        Run Criticality v2 evaluation (Phase 1: MCQ Logprob Probing).

        Args:
            limit: Maximum number of MCQ tasks.
            save_results: Whether to save per-task results.
            include_shuffled: If True, also run shuffled variants (Phase 3 preview).

        Returns:
            Dict with calibration metrics and task-level results.
        """
        # Load dataset and generate tasks
        num_tasks = limit or 200
        arg_limit = num_tasks * 5  # Load enough arguments
        self._task_gen.load_arguments(limit=arg_limit)
        tasks = self._task_gen.generate_tasks(num_tasks=num_tasks)

        if not tasks:
            print("No tasks generated. Check dataset availability.")
            return {"error": "No tasks generated"}

        # Optionally add shuffled variants
        if include_shuffled:
            shuffled = [self._task_gen.generate_shuffled_variant(t) for t in tasks[:20]]
            tasks.extend(shuffled)

        # Reset
        self._results = []
        self._top1_correct = 0
        self._total = 0

        # Load sequence scorer if in sequence mode
        if self._mode == "sequence":
            if self.verbose:
                print(f"Loading model from: {self.model_path}")
            self._scorer = SequenceScorer(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=self.verbose,
            )

        print(f"\n{'='*70}")
        print(f"Criticality v2 — Logprob-Based Argument Evaluation")
        print(f"{'='*70}")
        print(f"Model: {self._model_label} ({self.model})")
        print(f"Mode: {self._mode}")
        if self._mode == "sequence":
            print(f"Model path: {self.model_path}")
            print(f"Context size: {self.n_ctx}")
            print(f"GPU layers: {self.n_gpu_layers}")
        print(f"Tasks: {len(tasks)}")
        print(f"Quality score: {self.quality_score}")
        if self._mode == "api":
            print(f"Thinking model: {self._is_thinking}")
            print(f"Top logprobs: {self.top_logprobs}")
        print(f"Concurrency: {self.concurrency}")

        # Setup output directory
        if save_results:
            if self.run_dir:
                base_dir = self.run_dir
            else:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                base_dir = Path("results") / "criticality_v2" / f"{self._model_label}_{timestamp}"

            self._output_dir = base_dir
            self._output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving to: {self._output_dir}/")

        print()

        # Run tasks
        if self.concurrency > 1:
            results = self._run_concurrent(tasks)
        else:
            results = self._run_sequential(tasks)

        self._results = results

        # Compute calibration metrics
        metrics = compute_calibration_metrics(results)

        # Print summary
        self._print_summary(metrics, results)

        # Save final summary
        if save_results:
            self._save_summary(metrics, results)

        # Cleanup sequence scorer
        if self._mode == "sequence" and self._scorer:
            self._scorer.close()
            self._scorer = None

        return {
            "metrics": metrics.to_dict(),
            "num_tasks": len(results),
            "results": results,
        }

    def _run_sequential(self, tasks: List[Dict]) -> List[Dict]:
        """Run tasks sequentially with progress bar."""
        results = []

        pbar = tqdm(
            tasks,
            desc="Running",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] Acc: {postfix}",
            postfix="---%",
        )

        for task in pbar:
            result = self._process_task(task)
            results.append(result)
            self._save_result(result)

            acc = (self._top1_correct / self._total * 100) if self._total > 0 else 0
            pbar.set_postfix_str(f"{acc:.1f}%")

            if self.verbose:
                status = "+" if result["is_correct"] else "-"
                src = result.get("extraction_source", "?")
                margin = result.get("margin")
                margin_str = f"m={margin:.2f}" if margin is not None else "no-lp"
                tqdm.write(f"  {result['task_id']}: {status} pred={result['predicted']} [{src}] {margin_str}")

        return results

    def _run_concurrent(self, tasks: List[Dict]) -> List[Dict]:
        """Run tasks concurrently."""
        results = [None] * len(tasks)

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            future_to_idx = {
                executor.submit(self._process_task, task): i
                for i, task in enumerate(tasks)
            }

            pbar = tqdm(
                total=len(tasks),
                desc=f"Running ({self.concurrency}x)",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] Acc: {postfix}",
                postfix="---%",
            )

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results[idx] = result
                    self._save_result(result)
                except Exception as e:
                    results[idx] = {
                        "task_id": tasks[idx]["task_id"],
                        "error": str(e),
                        "is_correct": False,
                        "predicted": None,
                        "correct_label": tasks[idx]["correct_label"],
                        "latency": 0.0,
                    }

                acc = (self._top1_correct / self._total * 100) if self._total > 0 else 0
                pbar.set_postfix_str(f"{acc:.1f}%")
                pbar.update(1)

            pbar.close()

        return results

    # ── Summary ────────────────────────────────────────────────────────────

    def _print_summary(self, metrics: CalibrationMetrics, results: List[Dict]):
        """Print evaluation summary."""
        print(f"\n{'='*70}")
        print(f"CRITICALITY v2 RESULTS")
        print(f"{'='*70}")
        print(f"  Model:              {self._model_label}")
        print(f"  Tasks:              {metrics.num_tasks}")
        print(f"  Top-1 Accuracy:     {metrics.top1_accuracy*100:.1f}%")
        print(f"  Rank Correlation:   {metrics.rank_correlation:.4f}")
        print(f"  Calibration Error:  {metrics.calibration_error:.4f}")
        print(f"  Avg Margin:         {metrics.avg_margin:.4f}")
        print(f"  Avg Entropy:        {metrics.avg_entropy:.4f}")

        if metrics.easy_accuracy > 0 or metrics.hard_accuracy > 0:
            print(f"\n  Easy tasks acc:     {metrics.easy_accuracy*100:.1f}%")
            print(f"  Hard tasks acc:     {metrics.hard_accuracy*100:.1f}%")

        # Logprob extraction stats
        with_logprobs = sum(1 for r in results if r.get("choice_logprobs") is not None)
        without = len(results) - with_logprobs
        print(f"\n  Logprob extraction: {with_logprobs}/{len(results)} successful")
        if without > 0:
            print(f"  Missing logprobs:   {without} (fell back to content parsing)")

        # Source breakdown
        sources = defaultdict(int)
        for r in results:
            sources[r.get("extraction_source", "none")] += 1
        if sources:
            print(f"\n  Extraction sources:")
            for src, count in sorted(sources.items()):
                print(f"    {src}: {count}")

        print(f"{'='*70}")

    def _save_summary(self, metrics: CalibrationMetrics, results: List[Dict]):
        """Save final summary JSON."""
        if not self._output_dir:
            return

        total_latency = sum(r.get("latency", 0) for r in results)
        total_prompt_tokens = sum(r.get("prompt_tokens", 0) for r in results)
        total_completion_tokens = sum(r.get("completion_tokens", 0) for r in results)

        summary = {
            "model": self.model,
            "model_label": self._model_label,
            "benchmark": "Criticality v2 (Logprob-Based)",
            "quality_score": self.quality_score,
            "is_thinking_model": self._is_thinking,
            "top_logprobs_requested": self.top_logprobs,
            "metrics": metrics.to_dict(),
            "num_tasks": len(results),
            "total_latency_seconds": round(total_latency, 2),
            "avg_latency_seconds": round(total_latency / len(results), 2) if results else 0,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "results_dir": str(self._output_dir),
        }

        with open(self._output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nSummary saved to: {self._output_dir / 'summary.json'}")
