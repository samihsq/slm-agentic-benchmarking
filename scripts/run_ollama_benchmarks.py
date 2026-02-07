#!/usr/bin/env python3
"""
Run all benchmarks on remote Ollama models.

Usage:
    python scripts/run_ollama_benchmarks.py
    python scripts/run_ollama_benchmarks.py --models dasd-4b --benchmarks medqa,recall --limit 50
    python scripts/run_ollama_benchmarks.py --concurrency 8 --limit 100
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.ollama_agent import OllamaAgent
from src.benchmarks import (
    MedQARunner,
    BFCLRunner,
    CriticalityRunner,
    RecallRunner,
    EpisodicMemoryRunner,
)
from src.config.azure_llm_config import OLLAMA_MODELS


OLLAMA_BASE_URL = "http://10.27.102.240:11434"

# All benchmarks to run
BENCHMARKS = ["medqa", "bfcl", "criticality", "recall", "episodic_memory"]

# Default concurrency per model (Ollama handles this differently than Azure --
# it queues requests and runs them based on available compute)
DEFAULT_CONCURRENCY = {
    "dasd-4b": 4,
    "falcon-h1-90m": 8,
    "qwen3-0.6b": 8,
}

# Default limits per benchmark
DEFAULT_LIMITS = {
    "medqa": 100,
    "bfcl": 100,
    "criticality": 100,
    "recall": 100,
    "episodic_memory": 50,
}


def create_agent(model_key: str, verbose: bool = False) -> OllamaAgent:
    """Create an OllamaAgent for the given model."""
    config = OLLAMA_MODELS[model_key]
    return OllamaAgent(
        model=config["model"],
        verbose=verbose,
        ollama_base_url=OLLAMA_BASE_URL,
        temperature=0.7,
        max_tokens=2048,
    )


def run_benchmark(
    model_key: str,
    benchmark: str,
    agent: OllamaAgent,
    concurrency: int,
    limit: int,
    run_dir: Path,
    verbose: bool,
):
    """Run a single benchmark for a single model."""
    runners = {
        "medqa": lambda: MedQARunner(
            agent, cost_tracker=None, verbose=verbose,
            dataset="medqa", concurrency=concurrency, run_dir=run_dir,
        ),
        "bfcl": lambda: BFCLRunner(
            agent, cost_tracker=None, verbose=verbose,
            concurrency=concurrency, run_dir=run_dir,
        ),
        "criticality": lambda: CriticalityRunner(
            agent, cost_tracker=None, verbose=verbose,
            concurrency=concurrency, run_dir=run_dir,
        ),
        "recall": lambda: RecallRunner(
            agent, cost_tracker=None, verbose=verbose,
            concurrency=concurrency, run_dir=run_dir, num_chapters=20,
        ),
        "episodic_memory": lambda: EpisodicMemoryRunner(
            agent, cost_tracker=None, verbose=verbose,
            concurrency=concurrency, run_dir=run_dir, num_chapters=20,
        ),
    }

    if benchmark not in runners:
        print(f"  Unknown benchmark: {benchmark}, skipping")
        return None

    runner = runners[benchmark]()
    results = runner.run(limit=limit, save_results=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Run benchmarks on Ollama models")
    parser.add_argument(
        "--models", default="all",
        help="Comma-separated model keys (dasd-4b,falcon-h1-90m,qwen3-0.6b) or 'all'",
    )
    parser.add_argument(
        "--benchmarks", default="all",
        help=f"Comma-separated benchmarks ({','.join(BENCHMARKS)}) or 'all'",
    )
    parser.add_argument(
        "--concurrency", "-c", type=int, default=None,
        help="Override concurrency (default: per-model setting)",
    )
    parser.add_argument(
        "--limit", "-n", type=int, default=None,
        help="Override limit per benchmark (default: per-benchmark setting)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    # Resolve models
    if args.models == "all":
        model_keys = list(OLLAMA_MODELS.keys())
    else:
        model_keys = [m.strip() for m in args.models.split(",")]
        for mk in model_keys:
            if mk not in OLLAMA_MODELS:
                print(f"Unknown model: {mk}. Available: {list(OLLAMA_MODELS.keys())}")
                sys.exit(1)

    # Resolve benchmarks
    if args.benchmarks == "all":
        benchmarks = BENCHMARKS
    else:
        benchmarks = [b.strip() for b in args.benchmarks.split(",")]

    # Verify Ollama is reachable
    import urllib.request
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/version", timeout=5) as resp:
            version = json.loads(resp.read().decode())["version"]
            print(f"Ollama v{version} at {OLLAMA_BASE_URL}")
    except Exception as e:
        print(f"Cannot reach Ollama at {OLLAMA_BASE_URL}: {e}")
        sys.exit(1)

    # Create top-level results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_base = Path("results") / "ollama" / timestamp

    print(f"\n{'=' * 70}")
    print(f"OLLAMA BENCHMARK RUN")
    print(f"{'=' * 70}")
    print(f"Models:      {', '.join(model_keys)}")
    print(f"Benchmarks:  {', '.join(benchmarks)}")
    print(f"Results dir: {results_base}")
    print(f"{'=' * 70}\n")

    all_summaries = []

    for model_key in model_keys:
        config = OLLAMA_MODELS[model_key]
        concurrency = args.concurrency or DEFAULT_CONCURRENCY.get(model_key, 4)

        print(f"\n{'─' * 70}")
        print(f"MODEL: {model_key} ({config['description']})")
        print(f"Concurrency: {concurrency}")
        print(f"{'─' * 70}")

        agent = create_agent(model_key, verbose=args.verbose)

        for benchmark in benchmarks:
            limit = args.limit or DEFAULT_LIMITS.get(benchmark, 100)

            print(f"\n  >>> {benchmark.upper()} (limit={limit}, concurrency={concurrency})")

            run_dir = results_base / benchmark / model_key
            run_dir.mkdir(parents=True, exist_ok=True)

            start = time.time()
            try:
                results = run_benchmark(
                    model_key, benchmark, agent, concurrency, limit, run_dir, args.verbose
                )
                elapsed = time.time() - start

                if results:
                    correct = sum(1 for r in results if r.success)
                    total = len(results)
                    accuracy = correct / total if total > 0 else 0

                    summary = {
                        "model": model_key,
                        "benchmark": benchmark,
                        "accuracy": accuracy,
                        "correct": correct,
                        "total": total,
                        "elapsed_seconds": round(elapsed, 1),
                        "avg_latency": round(elapsed / total, 2) if total else 0,
                    }
                    all_summaries.append(summary)

                    print(f"  <<< {benchmark.upper()}: {accuracy*100:.1f}% ({correct}/{total}) in {elapsed:.0f}s")

            except Exception as e:
                elapsed = time.time() - start
                print(f"  <<< {benchmark.upper()}: FAILED after {elapsed:.0f}s -- {e}")
                all_summaries.append({
                    "model": model_key,
                    "benchmark": benchmark,
                    "error": str(e),
                    "elapsed_seconds": round(elapsed, 1),
                })

    # Print final summary table
    print(f"\n\n{'=' * 70}")
    print("FINAL RESULTS")
    print(f"{'=' * 70}")
    print(f"{'Model':<20} {'Benchmark':<18} {'Accuracy':>10} {'N':>6} {'Time':>8}")
    print(f"{'─' * 70}")
    for s in all_summaries:
        if "error" in s:
            print(f"{s['model']:<20} {s['benchmark']:<18} {'ERROR':>10} {'':>6} {s['elapsed_seconds']:>7.0f}s")
        else:
            acc_str = f"{s['accuracy']*100:.1f}%"
            print(f"{s['model']:<20} {s['benchmark']:<18} {acc_str:>10} {s['total']:>6} {s['elapsed_seconds']:>7.0f}s")

    # Save combined summary
    summary_file = results_base / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
