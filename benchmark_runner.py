#!/usr/bin/env python3
"""
Unified Benchmark Runner with Real-Time Telemetry

Run multiple models and benchmarks with automatic rate limiting and live progress tracking.

Usage:
    # Run phi-4 on medqa with all agents
    python benchmark_runner.py --models phi-4 --benchmarks medqa --agents all
    
    # Run multiple models on multiple benchmarks
    python benchmark_runner.py --models phi-4,gpt-4o,deepseek-v3.2 --benchmarks medqa,bfcl --agents oneshot,sequential
    
    # Run all available models on all benchmarks
    python benchmark_runner.py --models all --benchmarks all --agents all
    
    # Specify concurrency and limit
    python benchmark_runner.py --models phi-4 --benchmarks medqa -c 50 -n 100
"""

import argparse
import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import AVAILABLE_MODELS, get_llm_config
from src.agents import OneShotAgent, SequentialAgent, ConcurrentAgent, GroupChatAgent


# =============================================================================
# Configuration
# =============================================================================

BENCHMARKS = [
    "medqa",
    "bfcl",
    "criticality",
    "criticality_v2",
    "recall",
    "episodic_memory",
    "summarization",
]
AGENTS = {
    "oneshot": OneShotAgent,
    "sequential": SequentialAgent,
    "concurrent": ConcurrentAgent,
    "groupchat": GroupChatAgent,
}

DEFAULT_CONCURRENCY = {
    "phi-4": 100,
    "gpt-4o": 5,
    "deepseek-v3.2": 5,
    "mistral-large-3": 10,
    "mistral-small": 20,
    "ministral-3b": 50,
    "mistral-nemo": 20,
    "llama-3.3-70b": 10,
}


# =============================================================================
# Telemetry / Dashboard
# =============================================================================

@dataclass
class RunStats:
    """Statistics for a single run (model + benchmark + agent)."""
    model: str
    benchmark: str
    agent: str
    total: int = 0
    completed: int = 0
    correct: int = 0
    errors: int = 0
    rate_limits: int = 0
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    
    @property
    def accuracy(self) -> float:
        return self.correct / max(self.completed, 1)
    
    @property
    def error_rate(self) -> float:
        return self.errors / max(self.completed + self.errors, 1)
    
    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time
    
    @property
    def rate(self) -> float:
        elapsed = self.elapsed
        return self.completed / elapsed if elapsed > 0 else 0


class TelemetryDashboard:
    """Real-time telemetry dashboard for benchmark runs."""
    
    def __init__(self):
        self.runs: Dict[str, RunStats] = {}
        self.global_errors: List[str] = []
        self.lock = threading.Lock()
        self.running = False
        self._display_thread = None
    
    def add_run(self, model: str, benchmark: str, agent: str, total: int) -> str:
        """Add a new run to track."""
        key = f"{model}:{benchmark}:{agent}"
        with self.lock:
            self.runs[key] = RunStats(
                model=model,
                benchmark=benchmark,
                agent=agent,
                total=total,
            )
        return key
    
    def update(self, key: str, completed: int = 0, correct: int = 0, 
               errors: int = 0, rate_limits: int = 0):
        """Update run statistics."""
        with self.lock:
            if key in self.runs:
                stats = self.runs[key]
                stats.completed += completed
                stats.correct += correct
                stats.errors += errors
                stats.rate_limits += rate_limits
                stats.last_update = time.time()
    
    def log_error(self, message: str):
        """Log a global error."""
        with self.lock:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.global_errors.append(f"[{timestamp}] {message}")
            # Keep only last 10 errors
            if len(self.global_errors) > 10:
                self.global_errors = self.global_errors[-10:]
    
    def start_display(self, refresh_rate: float = 0.5):
        """Start the live display thread."""
        self.running = True
        self._display_thread = threading.Thread(target=self._display_loop, args=(refresh_rate,))
        self._display_thread.daemon = True
        self._display_thread.start()
    
    def stop_display(self):
        """Stop the live display."""
        self.running = False
        if self._display_thread:
            self._display_thread.join(timeout=1)
    
    def _display_loop(self, refresh_rate: float):
        """Background thread for display updates."""
        while self.running:
            self._render()
            time.sleep(refresh_rate)
    
    def _render(self):
        """Render the dashboard to terminal."""
        # Clear screen and move cursor to top
        print("\033[2J\033[H", end="")
        
        print("=" * 90)
        print("🚀 BENCHMARK RUNNER - LIVE TELEMETRY")
        print("=" * 90)
        
        with self.lock:
            if not self.runs:
                print("\nNo runs active yet...")
                return
            
            # Header
            print(f"\n{'Run':<35} {'Progress':>12} {'Accuracy':>10} {'Rate':>10} {'Errors':>8} {'Time':>8}")
            print("-" * 90)
            
            total_completed = 0
            total_correct = 0
            total_errors = 0
            total_tasks = 0
            
            for key, stats in self.runs.items():
                # Progress bar
                pct = stats.completed / max(stats.total, 1)
                bar_len = 10
                filled = int(bar_len * pct)
                bar = "█" * filled + "░" * (bar_len - filled)
                
                # Status indicator
                if stats.completed >= stats.total:
                    status = "✅"
                elif stats.errors > stats.completed * 0.1:
                    status = "⚠️"
                else:
                    status = "🔄"
                
                name = f"{stats.model}/{stats.benchmark}/{stats.agent}"
                if len(name) > 33:
                    name = name[:30] + "..."
                
                progress = f"{bar} {stats.completed}/{stats.total}"
                accuracy = f"{stats.accuracy*100:.1f}%"
                rate = f"{stats.rate:.1f}/s"
                errors = f"{stats.errors}"
                elapsed = f"{stats.elapsed:.0f}s"
                
                print(f"{status} {name:<33} {progress:>12} {accuracy:>10} {rate:>10} {errors:>8} {elapsed:>8}")
                
                total_completed += stats.completed
                total_correct += stats.correct
                total_errors += stats.errors
                total_tasks += stats.total
            
            # Summary
            print("-" * 90)
            overall_accuracy = total_correct / max(total_completed, 1) * 100
            overall_progress = total_completed / max(total_tasks, 1) * 100
            print(f"{'TOTAL':<35} {total_completed:>5}/{total_tasks:<6} {overall_accuracy:>9.1f}% {'':>10} {total_errors:>8}")
            
            # Recent errors
            if self.global_errors:
                print("\n📋 Recent Errors:")
                for err in self.global_errors[-5:]:
                    print(f"   {err[:80]}")
        
        print("\n" + "=" * 90)
        print("Press Ctrl+C to stop")
    
    def final_summary(self):
        """Print final summary after all runs complete."""
        print("\n" + "=" * 90)
        print("📊 FINAL RESULTS")
        print("=" * 90)
        
        with self.lock:
            for key, stats in self.runs.items():
                print(f"\n{stats.model} / {stats.benchmark} / {stats.agent}:")
                print(f"   Completed: {stats.completed}/{stats.total}")
                print(f"   Accuracy:  {stats.accuracy*100:.1f}%")
                print(f"   Errors:    {stats.errors} ({stats.error_rate*100:.1f}%)")
                print(f"   Duration:  {stats.elapsed:.1f}s")
                print(f"   Rate:      {stats.rate:.2f} tasks/sec")


# Global dashboard instance
dashboard = TelemetryDashboard()


# =============================================================================
# Benchmark Runners
# =============================================================================

def run_medqa(model: str, agent_type: str, concurrency: int, limit: int) -> Dict[str, Any]:
    """Run MedQA benchmark."""
    from src.benchmarks.archive.medical.medqa_runner import MedQARunner
    
    # Create agent
    agent_class = AGENTS[agent_type]
    agent = agent_class(model=model, verbose=False)
    
    # Track progress through dashboard
    key = dashboard.add_run(model, "medqa", agent_type, limit)
    
    # Create timestamped output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("results") / "medqa" / f"{model}_{timestamp}"
    
    # Create runner with concurrency
    runner = MedQARunner(
        agent=agent,
        verbose=False,
        dataset="medqa",
        concurrency=concurrency,
        run_dir=run_dir,
    )
    
    # Patch the _process_question method to update dashboard
    original_process = runner._process_question
    
    def tracked_process(question, idx, total):
        result = original_process(question, idx, total)
        # Check if correct from the result - EvaluationResult has .success attribute
        is_correct = getattr(result, 'success', False)
        dashboard.update(key, completed=1, correct=1 if is_correct else 0)
        return result
    
    runner._process_question = tracked_process
    
    # Run benchmark
    results = runner.run(limit=limit, save_results=True)
    
    # Calculate summary stats
    correct = sum(1 for r in results if r.success)
    
    return {
        "model": model,
        "benchmark": "medqa",
        "agent": agent_type,
        "num_tasks": len(results),
        "correct": correct,
        "accuracy": correct / len(results) if results else 0,
        "output_dir": str(run_dir),
    }


def run_bfcl(model: str, agent_type: str, concurrency: int, limit: int) -> Dict[str, Any]:
    """Run BFCL benchmark."""
    from src.benchmarks.archive.tool_calling.bfcl_runner import BFCLRunner
    
    # Create agent
    agent_class = AGENTS[agent_type]
    agent = agent_class(model=model, verbose=False)
    
    # Track progress through dashboard
    key = dashboard.add_run(model, "bfcl", agent_type, limit)
    
    # Create timestamped output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("results") / "bfcl" / f"{model}_{timestamp}"
    
    # Create runner with concurrency
    runner = BFCLRunner(
        agent=agent,
        verbose=False,
        concurrency=concurrency,
        run_dir=run_dir,
    )
    
    # Patch the _process_task method to update dashboard
    original_process = runner._process_task
    
    def tracked_process(task):
        result = original_process(task)
        is_correct = getattr(result, 'success', False)
        dashboard.update(key, completed=1, correct=1 if is_correct else 0)
        return result
    
    runner._process_task = tracked_process
    
    # Run benchmark
    results = runner.run(limit=limit, save_results=True)
    
    # Results is a dict with summary stats
    return {
        "model": model,
        "benchmark": "bfcl",
        "agent": agent_type,
        "num_tasks": results.get("num_tasks", 0),
        "correct": int(results.get("success_rate", 0) * results.get("num_tasks", 0)),
        "accuracy": results.get("success_rate", 0),
        "output_dir": str(run_dir),
    }


def run_criticality(model: str, agent_type: str, concurrency: int, limit: int) -> Dict[str, Any]:
    """Run Criticality (Argument Quality) benchmark."""
    from src.benchmarks.skills.criticality.v1.runner import CriticalityRunner
    
    # Create agent
    agent_class = AGENTS[agent_type]
    agent = agent_class(model=model, verbose=False)
    
    # Track progress through dashboard
    key = dashboard.add_run(model, "criticality", agent_type, limit)
    
    # Create timestamped output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("results") / "criticality" / f"{model}_{timestamp}"
    
    # Create runner with concurrency
    runner = CriticalityRunner(
        agent=agent,
        verbose=False,
        concurrency=concurrency,
        run_dir=run_dir,
    )
    
    # Patch the _process_task method to update dashboard
    original_process = runner._process_task
    
    def tracked_process(task):
        result = original_process(task)
        is_correct = getattr(result, 'success', False)
        dashboard.update(key, completed=1, correct=1 if is_correct else 0)
        return result
    
    runner._process_task = tracked_process
    
    # Run benchmark
    results = runner.run(limit=limit, save_results=True)
    
    # Calculate summary stats
    correct = sum(1 for r in results if r.success)
    
    return {
        "model": model,
        "benchmark": "criticality",
        "agent": agent_type,
        "num_tasks": len(results),
        "correct": correct,
        "accuracy": correct / len(results) if results else 0,
        "output_dir": str(run_dir),
    }


def run_recall(model: str, agent_type: str, concurrency: int, limit: int) -> Dict[str, Any]:
    """Run Simple Recall (Keyword-based sentence retrieval) benchmark."""
    from src.benchmarks.skills.recall.runner import RecallRunner
    
    # Create agent
    agent_class = AGENTS[agent_type]
    agent = agent_class(model=model, verbose=False)
    
    # Track progress through dashboard
    key = dashboard.add_run(model, "recall", agent_type, limit)
    
    # Create timestamped output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("results") / "recall" / f"{model}_{timestamp}"
    
    # Create runner with concurrency
    runner = RecallRunner(
        agent=agent,
        verbose=False,
        concurrency=concurrency,
        run_dir=run_dir,
        num_chapters=20,  # Use 20-chapter dataset (10K tokens)
    )
    
    # Patch the _process_task method to update dashboard
    original_process = runner._process_task
    
    def tracked_process(task):
        result = original_process(task)
        is_correct = getattr(result, 'success', False)
        dashboard.update(key, completed=1, correct=1 if is_correct else 0)
        return result
    
    runner._process_task = tracked_process
    
    # Run benchmark
    results = runner.run(limit=limit, save_results=True)
    
    # Calculate summary stats
    correct = sum(1 for r in results if r.success)
    
    return {
        "model": model,
        "benchmark": "recall",
        "agent": agent_type,
        "num_tasks": len(results),
        "correct": correct,
        "accuracy": correct / len(results) if results else 0,
        "output_dir": str(run_dir),
    }


def run_episodic_memory(model: str, agent_type: str, concurrency: int, limit: int) -> Dict[str, Any]:
    """Run Episodic Memory (State Tracking) benchmark."""
    from src.benchmarks.skills.episodic_memory.runner import EpisodicMemoryRunner
    
    # Create agent
    agent_class = AGENTS[agent_type]
    agent = agent_class(model=model, verbose=False)
    
    # Track progress through dashboard
    key = dashboard.add_run(model, "episodic_memory", agent_type, limit if limit else 100)
    
    # Create timestamped output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("results") / "episodic_memory" / f"{model}_{timestamp}"
    
    # Create runner with concurrency (default to 20 chapters dataset)
    runner = EpisodicMemoryRunner(
        agent=agent,
        verbose=False,
        concurrency=concurrency,
        run_dir=run_dir,
        num_chapters=20,  # Start with small dataset
    )
    
    # Patch the _process_task method to update dashboard
    original_process = runner._process_task
    
    def tracked_process(task, narrative):
        result = original_process(task, narrative)
        # Use F1 score > 0.5 as "correct"
        is_correct = result.score > 0.5 if hasattr(result, 'score') else False
        dashboard.update(key, completed=1, correct=1 if is_correct else 0)
        return result
    
    runner._process_task = tracked_process
    
    # Run benchmark
    results = runner.run(limit=limit, save_results=True)
    
    # Calculate summary stats (F1 scores)
    avg_f1 = sum(r.score for r in results if r.score) / len(results) if results else 0.0
    num_correct = sum(1 for r in results if r.score > 0.5)
    
    return {
        "model": model,
        "benchmark": "episodic_memory",
        "agent": agent_type,
        "num_tasks": len(results),
        "correct": num_correct,
        "avg_f1": avg_f1,
        "accuracy": num_correct / len(results) if results else 0,
        "output_dir": str(run_dir),
    }


def run_summarization(
    model: str,
    agent_type: str,
    concurrency: int,
    limit: int,
    *,
    metric: str = "rougeL",
    split: str = "validation",
    threshold: Optional[float] = None,
    bertscore_model_type: str = "microsoft/deberta-xlarge-mnli",
    bartscore_model: str = "facebook/bart-large-cnn",
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Run Summarization (XSum) benchmark."""
    from src.benchmarks.skills.summarization.runner import SummarizationRunner

    agent_class = AGENTS[agent_type]
    agent = agent_class(model=model, verbose=False)

    key = dashboard.add_run(model, "summarization", agent_type, limit)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("results") / "summarization" / f"{model}_{timestamp}"

    runner = SummarizationRunner(
        agent=agent,
        verbose=False,
        concurrency=concurrency,
        run_dir=run_dir,
        split=split,
        metric=metric,
        success_threshold=threshold,
        bertscore_model_type=bertscore_model_type,
        bartscore_model=bartscore_model,
        device=device,
    )

    original_process = runner._process_task

    def tracked_process(task):
        result = original_process(task)
        is_success = getattr(result, "success", False)
        dashboard.update(key, completed=1, correct=1 if is_success else 0)
        return result

    runner._process_task = tracked_process

    results = runner.run(limit=limit, save_results=True)

    avg_score = sum(r.score for r in results if r.score is not None) / len(results) if results else 0.0
    correct = sum(1 for r in results if r.success)

    return {
        "model": model,
        "benchmark": "summarization",
        "agent": agent_type,
        "num_tasks": len(results),
        "correct": correct,
        "avg_score": avg_score,
        "accuracy": correct / len(results) if results else 0,
        "metric": metric,
        "split": split,
        "output_dir": str(run_dir),
    }


def run_criticality_v2(model: str, agent_type: str, concurrency: int, limit: int, model_path: Optional[str] = None) -> Dict[str, Any]:
    """Run Criticality v2 (Logprob-Based) benchmark."""
    from src.benchmarks.skills.criticality.v2.runner import CriticalityV2Runner

    # Track progress through dashboard
    key = dashboard.add_run(model, "criticality_v2", agent_type, limit)

    # Create timestamped output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("results") / "criticality_v2" / f"{model}_{timestamp}"

    # Create runner (v2 uses its own OpenAI client, not an agent, or local GGUF scorer)
    runner = CriticalityV2Runner(
        model=model,
        model_path=model_path,
        verbose=False,
        concurrency=concurrency,
        run_dir=run_dir,
    )

    # Run benchmark
    output = runner.run(limit=limit, save_results=True)

    metrics = output.get("metrics", {})
    dashboard.update(key, completed=limit, correct=int(metrics.get("top1_accuracy", 0) * limit))

    return {
        "model": model,
        "benchmark": "criticality_v2",
        "agent": agent_type,
        "num_tasks": output.get("num_tasks", 0),
        "accuracy": metrics.get("top1_accuracy", 0),
        "rank_correlation": metrics.get("rank_correlation", 0),
        "calibration_error": metrics.get("calibration_error", 0),
        "output_dir": str(run_dir),
    }


BENCHMARK_RUNNERS = {
    "medqa": run_medqa,
    "bfcl": run_bfcl,
    "criticality": run_criticality,
    "criticality_v2": run_criticality_v2,
    "recall": run_recall,
    "episodic_memory": run_episodic_memory,
    "summarization": run_summarization,
}


# =============================================================================
# Main Runner
# =============================================================================

def get_concurrency(model: str, base_concurrency: Optional[int]) -> int:
    """Get appropriate concurrency for a model."""
    if base_concurrency:
        return base_concurrency
    return DEFAULT_CONCURRENCY.get(model, 10)


def run_single(
    model: str,
    benchmark: str,
    agent_type: str,
    concurrency: int,
    limit: int,
    model_path: Optional[str] = None,
    summarization_metric: str = "rougeL",
    summarization_split: str = "validation",
    summarization_threshold: Optional[float] = None,
    summarization_bertscore_model_type: str = "microsoft/deberta-xlarge-mnli",
    summarization_bartscore_model: str = "facebook/bart-large-cnn",
    summarization_device: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single benchmark configuration."""
    runner = BENCHMARK_RUNNERS.get(benchmark)
    if not runner:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    
    try:
        # criticality_v2 accepts model_path, others don't
        if benchmark == "criticality_v2":
            return runner(model, agent_type, concurrency, limit, model_path=model_path)
        elif benchmark == "summarization":
            return runner(
                model,
                agent_type,
                concurrency,
                limit,
                metric=summarization_metric,
                split=summarization_split,
                threshold=summarization_threshold,
                bertscore_model_type=summarization_bertscore_model_type,
                bartscore_model=summarization_bartscore_model,
                device=summarization_device,
            )
        else:
            return runner(model, agent_type, concurrency, limit)
    except Exception as e:
        dashboard.log_error(f"{model}/{benchmark}/{agent_type}: {str(e)[:50]}")
        return {
            "model": model,
            "benchmark": benchmark,
            "agent": agent_type,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Unified Benchmark Runner with Real-Time Telemetry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run phi-4 on medqa with all agents
    python benchmark_runner.py --models phi-4 --benchmarks medqa --agents all
    
    # Run multiple models on multiple benchmarks  
    python benchmark_runner.py --models phi-4,gpt-4o --benchmarks medqa,bfcl --agents oneshot
    
    # Run with specific concurrency and limit
    python benchmark_runner.py --models phi-4 --benchmarks medqa -c 50 -n 100
        """
    )
    
    parser.add_argument(
        "--models", "-m",
        type=str,
        required=True,
        help="Models to run (comma-separated, or 'all')"
    )
    parser.add_argument(
        "--benchmarks", "-b",
        type=str,
        required=True,
        help="Benchmarks to run (comma-separated, or 'all'): medqa, bfcl"
    )
    parser.add_argument(
        "--agents", "-a",
        type=str,
        default="all",
        help="Agents to run (comma-separated, or 'all'): oneshot, sequential, concurrent, groupchat"
    )
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=None,
        help="Concurrency level (auto-detected per model if not specified)"
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=100,
        help="Number of tasks per benchmark (default: 100)"
    )
    parser.add_argument(
        "--sequential-runs",
        action="store_true",
        help="Run configurations sequentially instead of in parallel"
    )
    parser.add_argument(
        "--parallel-runs", "-p",
        type=int,
        default=None,
        help="Max parallel run configurations (default: number of unique models)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results/combined",
        help="Output directory for combined_run_*.json (default: results/combined)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to GGUF model file for criticality_v2 sequence mode (e.g., Ollama blob path)"
    )

    # Summarization-specific options (only used when --benchmarks includes summarization)
    parser.add_argument(
        "--summarization-metric",
        type=str,
        default="rougeL",
        help="Summarization score metric: rougeL | bertscore | bartscore (default: rougeL)",
    )
    parser.add_argument(
        "--summarization-split",
        type=str,
        default="validation",
        help="XSum split to use: validation | test (default: validation)",
    )
    parser.add_argument(
        "--summarization-threshold",
        type=float,
        default=None,
        help="Override success threshold for summarization (default depends on metric)",
    )
    parser.add_argument(
        "--summarization-bertscore-model-type",
        type=str,
        default="microsoft/deberta-xlarge-mnli",
        help="HF model id for BERTScore (default: microsoft/deberta-xlarge-mnli)",
    )
    parser.add_argument(
        "--summarization-bartscore-model",
        type=str,
        default="facebook/bart-large-cnn",
        help="HF seq2seq model id for BARTScore (default: facebook/bart-large-cnn)",
    )
    parser.add_argument(
        "--summarization-device",
        type=str,
        default=None,
        help="Device for BERTScore/BARTScore (e.g. cpu, cuda). Default: auto",
    )
    
    args = parser.parse_args()
    
    # Parse models
    if args.models.lower() == "all":
        models = list(AVAILABLE_MODELS.keys())
    else:
        models = [m.strip() for m in args.models.split(",")]
    
    # Validate models
    for model in models:
        if model not in AVAILABLE_MODELS:
            print(f"❌ Unknown model: {model}")
            print(f"Available models: {', '.join(AVAILABLE_MODELS.keys())}")
            sys.exit(1)
    
    # Parse benchmarks
    if args.benchmarks.lower() == "all":
        benchmarks = BENCHMARKS
    else:
        benchmarks = [b.strip() for b in args.benchmarks.split(",")]
    
    # Validate benchmarks
    for benchmark in benchmarks:
        if benchmark not in BENCHMARK_RUNNERS:
            print(f"❌ Unknown benchmark: {benchmark}")
            print(f"Available benchmarks: {', '.join(BENCHMARK_RUNNERS.keys())}")
            sys.exit(1)
    
    # Parse agents
    if args.agents.lower() == "all":
        agents = list(AGENTS.keys())
    else:
        agents = [a.strip() for a in args.agents.split(",")]
    
    # Validate agents
    for agent in agents:
        if agent not in AGENTS:
            print(f"❌ Unknown agent: {agent}")
            print(f"Available agents: {', '.join(AGENTS.keys())}")
            sys.exit(1)
    
    # Build run configurations
    configs = []
    for model in models:
        for benchmark in benchmarks:
            for agent in agents:
                concurrency = get_concurrency(model, args.concurrency)
                configs.append({
                    "model": model,
                    "benchmark": benchmark,
                    "agent": agent,
                    "concurrency": concurrency,
                    "limit": args.limit,
                    "summarization_metric": args.summarization_metric,
                    "summarization_split": args.summarization_split,
                    "summarization_threshold": args.summarization_threshold,
                    "summarization_bertscore_model_type": args.summarization_bertscore_model_type,
                    "summarization_bartscore_model": args.summarization_bartscore_model,
                    "summarization_device": args.summarization_device,
                })
    
    print(f"📋 Running {len(configs)} configurations:")
    for cfg in configs:
        print(f"   • {cfg['model']} / {cfg['benchmark']} / {cfg['agent']} (c={cfg['concurrency']})")
    print()
    
    # Start telemetry dashboard
    dashboard.start_display()
    
    try:
        results = []
        
        if args.sequential_runs:
            # Run sequentially
            for cfg in configs:
                result = run_single(
                    cfg["model"],
                    cfg["benchmark"],
                    cfg["agent"],
                    cfg["concurrency"],
                    cfg["limit"],
                    model_path=args.model_path,
                    summarization_metric=cfg.get("summarization_metric", args.summarization_metric),
                    summarization_split=cfg.get("summarization_split", args.summarization_split),
                    summarization_threshold=cfg.get(
                        "summarization_threshold", args.summarization_threshold
                    ),
                    summarization_bertscore_model_type=cfg.get(
                        "summarization_bertscore_model_type", args.summarization_bertscore_model_type
                    ),
                    summarization_bartscore_model=cfg.get(
                        "summarization_bartscore_model", args.summarization_bartscore_model
                    ),
                    summarization_device=cfg.get("summarization_device", args.summarization_device),
                )
                results.append(result)
        else:
            # Run in parallel (one run per config, each run has its own concurrency)
            # Default: run as many parallel as there are unique models (separate rate limits)
            max_parallel = args.parallel_runs or len(models)
            print(f"🚀 Running {max_parallel} configurations in parallel\n")
            with ThreadPoolExecutor(max_workers=min(len(configs), max_parallel)) as executor:
                futures = {
                    executor.submit(
                        run_single,
                        cfg["model"],
                        cfg["benchmark"],
                        cfg["agent"],
                        cfg["concurrency"],
                        cfg["limit"],
                        args.model_path,
                        cfg.get("summarization_metric", args.summarization_metric),
                        cfg.get("summarization_split", args.summarization_split),
                        cfg.get("summarization_threshold", args.summarization_threshold),
                        cfg.get(
                            "summarization_bertscore_model_type",
                            args.summarization_bertscore_model_type,
                        ),
                        cfg.get("summarization_bartscore_model", args.summarization_bartscore_model),
                        cfg.get("summarization_device", args.summarization_device),
                    ): cfg
                    for cfg in configs
                }
                
                for future in as_completed(futures):
                    cfg = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        dashboard.log_error(f"{cfg['model']}/{cfg['benchmark']}: {e}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    
    finally:
        # Stop dashboard and show final summary
        dashboard.stop_display()
        time.sleep(0.5)  # Let display thread finish
        dashboard.final_summary()
        
        # Save combined results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(args.output_dir) / f"combined_run_{timestamp}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w") as f:
            json.dump({
                "timestamp": timestamp,
                "configs": configs,
                "results": [r for r in results if "error" not in r],
                "errors": [r for r in results if "error" in r],
            }, f, indent=2, default=str)
        
        print(f"\n📁 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
