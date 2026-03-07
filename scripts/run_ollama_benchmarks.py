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
from src.agents.ollama_agentic_agent import OllamaSequentialAgent, OllamaConcurrentAgent, OllamaGroupChatAgent
from src.benchmarks import (
    MedQARunner,
    BFCLRunner,
    CriticalityRunner,
    RecallRunner,
    MatrixRecallRunner,
    EpisodicMemoryRunner,
    InstructionFollowingRunner,
    WordInstructionFollowingRunner,
    SummarizationRunner,
    PlanningRunner,
    PlanBenchRunner,
    BigBenchRunner,
)
from src.benchmarks.skills.bigbench import DEFAULT_TASK_CONFIGS as BIGBENCH_DEFAULT_CONFIGS
from src.config.azure_llm_config import OLLAMA_MODELS


OLLAMA_BASE_URL = "http://localhost:11434"

# All benchmarks to run
BENCHMARKS = ["medqa", "bfcl", "criticality", "recall", "matrix_recall", "matrix_recall_xhard", "episodic_memory", "instruction_following", "word_instruction_following", "summarization", "planning", "plan_bench", "bigbench"]

# Default concurrency per model (Ollama handles this differently than Azure --
# it queues requests and runs them based on available compute)
DEFAULT_CONCURRENCY = {
    "dasd-4b": 8,
    "falcon-h1-90m": 8,
    "qwen3-0.6b": 8,
    "gemma3-1b": 8,
    "gemma3-4b": 8,
    "gemma3n-e2b": 8,
    "gemma3n-e4b": 8,
    "gpt-oss-20b": 4,
    "phi4-mini-reasoning-ollama": 8,
}

# Default limits per benchmark
DEFAULT_LIMITS = {
    "medqa": 100,
    "bfcl": 100,
    "criticality": 100,
    "recall": 100,
    "episodic_memory": 50,
    "instruction_following": 10,
    "word_instruction_following": 10,
    "matrix_recall": 50,
    "matrix_recall_xhard": 50,
    "summarization": 100,
    "planning": 50,
    "plan_bench": 5,
    "bigbench": 30,
}


AGENTIC_AGENT_TYPES = ["sequential", "concurrent", "groupchat"]
ALL_AGENT_TYPES = ["one_shot"] + AGENTIC_AGENT_TYPES


def create_agentic_agent(model_key: str, agent_type: str, verbose: bool = False):
    """Create an Ollama-backed agentic agent (sequential/concurrent/groupchat)."""
    cls = {
        "sequential": OllamaSequentialAgent,
        "concurrent": OllamaConcurrentAgent,
        "groupchat": OllamaGroupChatAgent,
    }[agent_type]
    return cls(model=model_key, verbose=verbose, ollama_base_url=OLLAMA_BASE_URL)


def create_agent(model_key: str, verbose: bool = False, benchmark: str = "") -> OllamaAgent:
    """Create an OllamaAgent for the given model."""
    config = OLLAMA_MODELS[model_key]
    # Word instruction following only needs short list outputs — disable thinking
    # and cap tokens to avoid models burning budget on CoT before answering.
    if benchmark in ("word_instruction_following", "bigbench"):
        return OllamaAgent(
            model=config["model"],
            verbose=verbose,
            ollama_base_url=OLLAMA_BASE_URL,
            temperature=0.1,
            max_tokens=512,
            think=False,
        )
    # No token limit for other benchmarks (planning, etc.); Ollama uses num_predict: -1
    return OllamaAgent(
        model=config["model"],
        verbose=verbose,
        ollama_base_url=OLLAMA_BASE_URL,
        temperature=0.7,
        max_tokens=None,
        think=True,
    )


def run_benchmark(
    model_key: str,
    benchmark: str,
    agent,
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
        "matrix_recall": lambda: MatrixRecallRunner(
            agent, cost_tracker=None, verbose=verbose,
            concurrency=concurrency, run_dir=run_dir, num_tasks=50,
        ),
        "matrix_recall_xhard": lambda: MatrixRecallRunner(
            agent, cost_tracker=None, verbose=verbose,
            concurrency=concurrency, run_dir=run_dir, num_tasks=50,
            difficulty_distribution={"x-hard": 1.0},
        ),
        "instruction_following": lambda: InstructionFollowingRunner(
            agent, cost_tracker=None, verbose=verbose,
            concurrency=concurrency, run_dir=run_dir, num_tasks=10, matrix_size=4,
        ),
        "word_instruction_following": lambda: WordInstructionFollowingRunner(
            agent, cost_tracker=None, verbose=verbose,
            concurrency=concurrency, run_dir=run_dir, num_tasks=10,
        ),
        "summarization": lambda: SummarizationRunner(
            agent, cost_tracker=None, verbose=verbose,
            concurrency=concurrency, run_dir=run_dir, split="validation",
        ),
        "planning": lambda: PlanningRunner(
            agent, cost_tracker=None, verbose=verbose,
            concurrency=concurrency, run_dir=run_dir, domain="all", language="en",
        ),
        "plan_bench": lambda: PlanBenchRunner(
            agent, task="t1", config="blocksworld", run_dir=run_dir,
            cost_tracker=None, verbose=verbose,
        ),
        "bigbench": lambda: BigBenchRunner(
            agent, cost_tracker=None, verbose=verbose,
            concurrency=concurrency, run_dir=run_dir,
            task_configs=BIGBENCH_DEFAULT_CONFIGS,
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
    parser.add_argument(
        "--agent-types", default="one_shot",
        help=f"Comma-separated agent types ({','.join(ALL_AGENT_TYPES)}) or 'all' (default: one_shot)",
    )
    parser.add_argument(
        "--plan-bench-tasks", default="t1",
        help="Tasks for plan_bench: comma-separated (t1,t2,...,t8_3) or 'all'",
    )
    parser.add_argument(
        "--plan-bench-configs", default="blocksworld",
        help="Configs for plan_bench: comma-separated or 'all' (blocksworld,blocksworld_3)",
    )
    args = parser.parse_args()

    # Resolve agent types
    if args.agent_types == "all":
        agent_types = ALL_AGENT_TYPES
    else:
        agent_types = [a.strip() for a in args.agent_types.split(",")]
        for at in agent_types:
            if at not in ALL_AGENT_TYPES:
                print(f"Unknown agent type: {at}. Available: {ALL_AGENT_TYPES}")
                sys.exit(1)

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
    print(f"Agent types: {', '.join(agent_types)}")
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

        _PB_ALL_TASKS = ["t1","t2","t3","t4","t5","t6","t7","t8_1","t8_2","t8_3"]
        _PB_ALL_CONFIGS = ["blocksworld", "blocksworld_3"]
        pb_tasks = _PB_ALL_TASKS if args.plan_bench_tasks == "all" else [t.strip() for t in args.plan_bench_tasks.split(",")]
        pb_configs = _PB_ALL_CONFIGS if args.plan_bench_configs == "all" else [c.strip() for c in args.plan_bench_configs.split(",")]

        for benchmark in benchmarks:
            limit = args.limit or DEFAULT_LIMITS.get(benchmark, 100)

            for agent_type in agent_types:
                if agent_type == "one_shot":
                    agent = create_agent(model_key, verbose=args.verbose, benchmark=benchmark)
                    agent_concurrency = concurrency
                else:
                    agent = create_agentic_agent(model_key, agent_type, verbose=args.verbose)
                    # Outer concurrency: multiple tasks run in parallel via ThreadPoolExecutor,
                    # each with its own Crew. Ollama queues requests internally.
                    agent_concurrency = concurrency

                if benchmark == "plan_bench":
                    if agent_type != "one_shot":
                        continue  # plan_bench only supports one_shot for Ollama
                    for pb_config in pb_configs:
                        for pb_task in pb_tasks:
                            bench_label = f"plan_bench:{pb_config}:{pb_task}"
                            print(f"\n  >>> {bench_label.upper()} (limit={limit}, concurrency={agent_concurrency})")
                            sub_run_dir = results_base / "plan_bench" / model_key / pb_config / pb_task
                            sub_run_dir.mkdir(parents=True, exist_ok=True)
                            start = time.time()
                            try:
                                runner = PlanBenchRunner(
                                    agent, task=pb_task, config=pb_config, run_dir=sub_run_dir,
                                    cost_tracker=None, verbose=args.verbose,
                                )
                                results = runner.run(limit=limit, save_results=True)
                                elapsed = time.time() - start
                                if results:
                                    correct = sum(1 for r in results if r.get("success", False))
                                    total = len(results)
                                    accuracy = correct / total if total > 0 else 0
                                    summary = {
                                        "model": model_key, "benchmark": bench_label,
                                        "agent_type": agent_type,
                                        "accuracy": accuracy, "correct": correct,
                                        "total": total, "elapsed_seconds": round(elapsed, 1),
                                        "avg_latency": round(elapsed / total, 2) if total else 0,
                                    }
                                    all_summaries.append(summary)
                                    print(f"  <<< {bench_label.upper()}: {accuracy*100:.1f}% ({correct}/{total}) in {elapsed:.0f}s")
                            except Exception as e:
                                elapsed = time.time() - start
                                print(f"  <<< {bench_label.upper()}: FAILED after {elapsed:.0f}s -- {e}")
                                all_summaries.append({
                                    "model": model_key, "benchmark": bench_label,
                                    "agent_type": agent_type,
                                    "error": str(e), "elapsed_seconds": round(elapsed, 1),
                                })
                    continue  # skip the generic run_benchmark path for plan_bench

                label = f"{benchmark} [{agent_type}]"
                print(f"\n  >>> {label.upper()} (limit={limit}, concurrency={agent_concurrency})")

                run_dir = results_base / benchmark / model_key / agent_type
                run_dir.mkdir(parents=True, exist_ok=True)

                start = time.time()
                try:
                    results = run_benchmark(
                        model_key, benchmark, agent, agent_concurrency, limit, run_dir, args.verbose
                    )
                    elapsed = time.time() - start

                    if results:
                        correct = sum(1 for r in results if (r.get("success", False) if isinstance(r, dict) else getattr(r, "success", False)))
                        total = len(results)
                        accuracy = correct / total if total > 0 else 0

                        summary = {
                            "model": model_key,
                            "benchmark": benchmark,
                            "agent_type": agent_type,
                            "accuracy": accuracy,
                            "correct": correct,
                            "total": total,
                            "elapsed_seconds": round(elapsed, 1),
                            "avg_latency": round(elapsed / total, 2) if total else 0,
                        }
                        all_summaries.append(summary)

                        print(f"  <<< {label.upper()}: {accuracy*100:.1f}% ({correct}/{total}) in {elapsed:.0f}s")

                except Exception as e:
                    elapsed = time.time() - start
                    print(f"  <<< {label.upper()}: FAILED after {elapsed:.0f}s -- {e}")
                    all_summaries.append({
                        "model": model_key,
                        "benchmark": benchmark,
                        "agent_type": agent_type,
                        "error": str(e),
                        "elapsed_seconds": round(elapsed, 1),
                    })

    # Print final summary table
    print(f"\n\n{'=' * 70}")
    print("FINAL RESULTS")
    print(f"{'=' * 70}")
    print(f"{'Model':<20} {'Benchmark':<18} {'Agent':<12} {'Accuracy':>10} {'N':>6} {'Time':>8}")
    print(f"{'─' * 76}")
    for s in all_summaries:
        agent_t = s.get("agent_type", "one_shot")
        if "error" in s:
            print(f"{s['model']:<20} {s['benchmark']:<18} {agent_t:<12} {'ERROR':>10} {'':>6} {s['elapsed_seconds']:>7.0f}s")
        else:
            acc_str = f"{s['accuracy']*100:.1f}%"
            print(f"{s['model']:<20} {s['benchmark']:<18} {agent_t:<12} {acc_str:>10} {s['total']:>6} {s['elapsed_seconds']:>7.0f}s")

    # Save combined summary
    summary_file = results_base / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
