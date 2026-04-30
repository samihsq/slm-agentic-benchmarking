#!/usr/bin/env python3
"""
Live integration test for the Planning (DeepPlanning) benchmark.

Runs PlanningRunner against Azure or Ollama backends and prints
per-domain success rates, mean composite score, and mean latency.

Usage:
  python scripts/test_planning_benchmark.py --backend azure --model phi-4 --limit 3
  python scripts/test_planning_benchmark.py --backend ollama --model dasd-4b --limit 2
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarks.skills.planning.runner import PlanningRunner


def main():
    parser = argparse.ArgumentParser(
        description="Run Planning benchmark (live Azure or Ollama)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="phi-4",
        help="Model name (Azure: e.g. phi-4, llama-3.3-70b; Ollama: e.g. dasd-4b)",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="one_shot",
        choices=["none", "one_shot", "sequential", "concurrent", "group_chat", "baseline"],
        help="Agent architecture (default: one_shot)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of tasks to run (default: 5)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="all",
        choices=["all", "travel", "shopping"],
        help="Task domain (default: all)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="azure",
        choices=["azure", "ollama"],
        help="Backend: azure or ollama (default: azure)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    try:
        if args.backend == "azure":
            from src.agents import (
                OneShotAgent,
                SequentialAgent,
                ConcurrentAgent,
                GroupChatAgent,
                BaselineAgent,
                get_baseline_agent,
            )
            agents = {
                "none": lambda: BaselineAgent(model=args.model, verbose=args.verbose),
                "one_shot": lambda: OneShotAgent(model=args.model, verbose=args.verbose),
                "sequential": lambda: SequentialAgent(model=args.model, verbose=args.verbose),
                "concurrent": lambda: ConcurrentAgent(model=args.model, verbose=args.verbose),
                "group_chat": lambda: GroupChatAgent(model=args.model, verbose=args.verbose),
                "baseline": lambda: get_baseline_agent(model=args.model, verbose=args.verbose),
            }
            agent = agents[args.agent]()
        else:
            from src.agents.ollama_agent import OllamaAgent
            from src.config.azure_llm_config import OLLAMA_MODELS
            OLLAMA_BASE_URL = "http://localhost:11434"
            if args.model not in OLLAMA_MODELS:
                print(f"Unknown Ollama model key: {args.model}")
                print(f"Available: {', '.join(OLLAMA_MODELS.keys())}")
                sys.exit(1)
            config = OLLAMA_MODELS[args.model]
            agent = OllamaAgent(
                model=config["model"],
                verbose=args.verbose,
                ollama_base_url=OLLAMA_BASE_URL,
                temperature=0.7,
                max_tokens=None,
                think=True,
            )

        run_dir = Path("results") / "planning" / f"{args.model}_{time.strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)

        runner = PlanningRunner(
            agent,
            cost_tracker=None,
            verbose=args.verbose,
            concurrency=1,
            run_dir=run_dir,
            domain=args.domain,
            language="en",
        )
        print(f"\nRunning Planning benchmark: backend={args.backend}, model={args.model}, agent={args.agent}, limit={args.limit}, domain={args.domain}")
        print(f"Results: {run_dir}\n")
        results = runner.run(limit=args.limit, save_results=True)

        if not results:
            print("No results returned.")
            sys.exit(1)

        # Aggregate
        total = len(results)
        success_count = sum(1 for r in results if r.success)
        mean_score = sum(r.score or 0 for r in results) / total
        mean_latency = sum(r.latency or 0 for r in results) / total
        travel_results = [r for r in results if (r.metadata or {}).get("domain") == "travel"]
        shopping_results = [r for r in results if (r.metadata or {}).get("domain") == "shopping"]

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"  Success rate:        {success_count}/{total} ({100 * success_count / total:.1f}%)")
        print(f"  Mean composite score: {mean_score:.4f}")
        print(f"  Mean latency:        {mean_latency:.2f}s")
        if travel_results:
            t_success = sum(1 for r in travel_results if r.success)
            print(f"  Travel success:      {t_success}/{len(travel_results)} ({100 * t_success / len(travel_results):.1f}%)")
        if shopping_results:
            s_success = sum(1 for r in shopping_results if r.success)
            print(f"  Shopping success:    {s_success}/{len(shopping_results)} ({100 * s_success / len(shopping_results):.1f}%)")
        print("=" * 60)
        print(f"\nSummary saved to: {run_dir / agent.__class__.__name__ / 'summary.json'}")
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
