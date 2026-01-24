#!/usr/bin/env python3
"""
Unified Benchmark Runner CLI.

Run benchmarks across multiple models and agent architectures with cost tracking.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agents import (
    OneShotAgent,
    SequentialAgent,
    ConcurrentAgent,
    GroupChatAgent,
    BaselineAgent,
    get_baseline_agent,
)
from src.benchmarks import (
    MedAgentBenchRunner,
    MedQARunner,
    MCPBenchRunner,
    BFCLRunner,
)
from src.evaluation import CostTracker, estimate_experiment_cost, calculate_metrics
from src.config import list_models, print_model_info


def get_agent(agent_type: str, model: str, verbose: bool = False):
    """Get agent instance by type."""
    agents = {
        "none": lambda: BaselineAgent(model=model, verbose=verbose),
        "one_shot": lambda: OneShotAgent(model=model, verbose=verbose),
        "sequential": lambda: SequentialAgent(model=model, verbose=verbose),
        "concurrent": lambda: ConcurrentAgent(model=model, verbose=verbose),
        "group_chat": lambda: GroupChatAgent(model=model, verbose=verbose),
        "baseline": lambda: get_baseline_agent(
            cost_efficient=(model == "gpt-4o-mini"),
            verbose=verbose
        ),
    }
    
    if agent_type not in agents:
        raise ValueError(
            f"Unknown agent type: {agent_type}. "
            f"Available: {list(agents.keys())}"
        )
    
    return agents[agent_type]()


def get_benchmark_runner(benchmark: str, agent, cost_tracker: Optional[CostTracker] = None, verbose: bool = False):
    """Get benchmark runner instance."""
    runners = {
        "medagent": lambda: MedAgentBenchRunner(agent, cost_tracker, verbose),
        "medqa": lambda: MedQARunner(agent, cost_tracker, verbose, dataset="medqa"),
        "medmcqa": lambda: MedQARunner(agent, cost_tracker, verbose, dataset="medmcqa"),
        "mcp": lambda: MCPBenchRunner(agent, cost_tracker, verbose),
        "bfcl": lambda: BFCLRunner(agent, cost_tracker, verbose),
    }
    
    if benchmark not in runners:
        raise ValueError(
            f"Unknown benchmark: {benchmark}. "
            f"Available: {list(runners.keys())}"
        )
    
    return runners[benchmark]()


def estimate_costs(args):
    """Estimate costs before running."""
    print("\n" + "=" * 70)
    print("COST ESTIMATION")
    print("=" * 70)
    
    # Determine models
    if args.models == "all":
        models = list(list_models(serverless_only=True).keys())
        models.extend(["glm-4.7-flash", "qwen3-30b-a3b", "lfm2.5-1.2b"])
    else:
        models = args.models.split(",")
    
    # Determine benchmarks
    benchmarks = {
        "medagent": 100,
        "medqa": 1273,
        "mcp": 250,
        "bfcl": 2000,
    }
    
    if args.benchmark != "all":
        benchmarks = {args.benchmark: benchmarks.get(args.benchmark, 100)}
    
    print(f"\nModels: {', '.join(models)}")
    print(f"Benchmarks: {', '.join(benchmarks.keys())}")
    print(f"\nEstimating costs...\n")
    
    costs = estimate_experiment_cost(
        models=models,
        benchmarks=benchmarks,
        avg_tokens_per_task=6000,
    )
    
    print("Estimated Costs:")
    print("-" * 70)
    for model, cost in costs.items():
        if model != "TOTAL":
            print(f"  {model:30s} ${cost:8.2f}")
    print("-" * 70)
    print(f"  {'TOTAL':30s} ${costs['TOTAL']:8.2f}")
    print("=" * 70 + "\n")
    
    return costs["TOTAL"]


def run_benchmark(args):
    """Run a single benchmark configuration."""
    # Initialize cost tracker
    cost_tracker = CostTracker(
        budget_limit=args.budget or 10000.0,
        alert_thresholds=[0.3, 0.6, 0.9],
        log_file="cost_tracking.json",
    )
    
    # Get agent
    agent = get_agent(args.agent, args.model, verbose=args.verbose)
    
    # Get benchmark runner
    runner = get_benchmark_runner(
        args.benchmark,
        agent,
        cost_tracker,
        verbose=args.verbose,
    )
    
    # Run benchmark
    print(f"\n{'=' * 70}")
    print(f"Running: {args.benchmark.upper()}")
    print(f"Model: {args.model}")
    print(f"Agent: {args.agent}")
    print(f"{'=' * 70}\n")
    
    results = runner.run(limit=args.limit, save_results=True)
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"Success Rate: {metrics.success_rate * 100:.1f}%")
    print(f"Average Confidence: {metrics.avg_confidence:.2f}")
    print(f"Average Latency: {metrics.avg_latency:.2f}s")
    print(f"Total Cost: ${metrics.total_cost:.2f}")
    print(f"{'=' * 70}\n")
    
    # Print cost summary
    cost_tracker.print_summary()
    
    return results, metrics


def compare_baseline(args):
    """Compare agentic vs non-agentic baseline."""
    print("\n" + "=" * 70)
    print("COMPARING AGENTIC VS NON-AGENTIC")
    print("=" * 70)
    
    cost_tracker = CostTracker(budget_limit=args.budget or 10000.0)
    
    # Run baseline (non-agentic)
    print("\n1. Running non-agentic baseline...")
    baseline_agent = get_baseline_agent(
        cost_efficient=(args.model == "gpt-4o-mini"),
        verbose=args.verbose,
    )
    baseline_runner = get_benchmark_runner(
        args.benchmark,
        baseline_agent,
        cost_tracker,
        verbose=args.verbose,
    )
    baseline_results = baseline_runner.run(limit=args.limit, save_results=True)
    baseline_metrics = calculate_metrics(baseline_results)
    
    # Run agentic
    print("\n2. Running agentic architecture...")
    agentic_agent = get_agent(args.agent, args.model, verbose=args.verbose)
    agentic_runner = get_benchmark_runner(
        args.benchmark,
        agentic_agent,
        cost_tracker,
        verbose=args.verbose,
    )
    agentic_results = agentic_runner.run(limit=args.limit, save_results=True)
    agentic_metrics = calculate_metrics(agentic_results)
    
    # Compare
    from src.evaluation.metrics import compare_metrics
    comparison = compare_metrics(baseline_metrics, agentic_metrics)
    
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"Baseline Success Rate: {baseline_metrics.success_rate * 100:.1f}%")
    print(f"Agentic Success Rate:  {agentic_metrics.success_rate * 100:.1f}%")
    print(f"Change: {comparison['success_rate_change']:+.1f}%")
    print(f"\nBaseline Cost: ${baseline_metrics.total_cost:.2f}")
    print(f"Agentic Cost:  ${agentic_metrics.total_cost:.2f}")
    print(f"Cost Ratio: {comparison['cost_ratio']:.2f}x")
    print("=" * 70 + "\n")
    
    cost_tracker.print_summary()


def main():
    parser = argparse.ArgumentParser(
        description="SLM Agentic Benchmarking Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Estimate costs before running
  python run_benchmark.py --estimate --models all --benchmark all

  # Run non-agentic baseline
  python run_benchmark.py --model gpt-4o-mini --agent baseline --benchmark medqa

  # Run agentic architecture
  python run_benchmark.py --model phi-4 --agent sequential --benchmark medagent

  # Compare agentic vs baseline
  python run_benchmark.py --compare-baseline --model phi-4 --agent sequential --benchmark medqa

  # Run with budget limit
  python run_benchmark.py --model phi-4 --agent sequential --benchmark medqa --budget 100
        """,
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="phi-4",
        help="Model to use (default: phi-4)",
    )
    
    parser.add_argument(
        "--agent",
        type=str,
        default="one_shot",
        choices=["none", "one_shot", "sequential", "concurrent", "group_chat", "baseline"],
        help="Agent architecture (default: one_shot)",
    )
    
    parser.add_argument(
        "--benchmark",
        type=str,
        default="medqa",
        choices=["medagent", "medqa", "medmcqa", "mcp", "bfcl", "all"],
        help="Benchmark to run (default: medqa)",
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tasks (for testing)",
    )
    
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Budget limit in USD (default: 10000)",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Estimate costs without running",
    )
    
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated list of models or 'all'",
    )
    
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Compare agentic vs non-agentic baseline",
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        print_model_info()
        return
    
    # Estimate costs if requested
    if args.estimate:
        estimate_costs(args)
        return
    
    # Compare baseline if requested
    if args.compare_baseline:
        compare_baseline(args)
        return
    
    # Run benchmark
    try:
        run_benchmark(args)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
