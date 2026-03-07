#!/usr/bin/env python3
"""
Unified Benchmark Runner CLI.

Run benchmarks across multiple models and agent architectures with cost tracking.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

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
    CriticalityRunner,
    CriticalityV2Runner,
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
        "baseline": lambda: get_baseline_agent(model=model, verbose=verbose),
    }
    
    if agent_type not in agents:
        raise ValueError(
            f"Unknown agent type: {agent_type}. "
            f"Available: {list(agents.keys())}"
        )
    
    return agents[agent_type]()


def get_benchmark_runner(benchmark: str, agent, cost_tracker: Optional[CostTracker] = None, verbose: bool = False, concurrency: int = 1, run_dir: Optional[Path] = None, model_path: Optional[str] = None, model_name: Optional[str] = None, list_size: Optional[int] = None, plan_bench_task: Optional[str] = None, plan_bench_config: Optional[str] = None, bigbench_tasks: Optional[List[str]] = None):
    """Get benchmark runner instance."""
    # For criticality_v2, resolve the model name from agent or direct arg
    _model = model_name or (agent.model if agent else "unknown")

    runners = {
        "medagent": lambda: MedAgentBenchRunner(agent, cost_tracker, verbose),
        "medqa": lambda: MedQARunner(agent, cost_tracker, verbose, dataset="medqa", concurrency=concurrency, run_dir=run_dir),
        "medmcqa": lambda: MedQARunner(agent, cost_tracker, verbose, dataset="medmcqa", concurrency=concurrency, run_dir=run_dir),
        "mcp": lambda: MCPBenchRunner(agent, cost_tracker, verbose),
        "bfcl": lambda: BFCLRunner(agent, cost_tracker, verbose, concurrency=concurrency, run_dir=run_dir),
        "criticality": lambda: CriticalityRunner(agent, cost_tracker, verbose, concurrency=concurrency, run_dir=run_dir),
        "criticality_v2": lambda: CriticalityV2Runner(model=_model, model_path=model_path, verbose=verbose, concurrency=concurrency, run_dir=run_dir),
        "recall": lambda: RecallRunner(agent, cost_tracker, verbose, concurrency=concurrency, run_dir=run_dir, num_chapters=20),
        "matrix_recall": lambda: MatrixRecallRunner(agent, cost_tracker, verbose, concurrency=concurrency, run_dir=run_dir, num_tasks=200),
        "matrix_recall_xhard": lambda: MatrixRecallRunner(agent, cost_tracker, verbose, concurrency=concurrency, run_dir=run_dir, num_tasks=200, difficulty_distribution={"x-hard": 1.0}),
        "episodic_memory": lambda: EpisodicMemoryRunner(agent, cost_tracker, verbose, concurrency=concurrency, run_dir=run_dir, num_chapters=20),
        "instruction_following": lambda: InstructionFollowingRunner(agent, cost_tracker, verbose, concurrency=concurrency, run_dir=run_dir, num_tasks=10, matrix_size=4),
        "word_instruction_following": lambda: WordInstructionFollowingRunner(agent, cost_tracker, verbose, concurrency=concurrency, run_dir=run_dir, num_tasks=10, list_size=list_size),
        "summarization": lambda: SummarizationRunner(agent, cost_tracker, verbose, concurrency=concurrency, run_dir=run_dir, split="validation"),
        "planning": lambda: PlanningRunner(agent, cost_tracker=cost_tracker, verbose=verbose, concurrency=concurrency, run_dir=run_dir, domain="all", language="en"),
        "plan_bench": lambda: PlanBenchRunner(agent, task=plan_bench_task or "t1", config=plan_bench_config or "blocksworld", run_dir=run_dir, cost_tracker=cost_tracker, verbose=verbose),
        "bigbench": lambda: BigBenchRunner(agent, cost_tracker=cost_tracker, verbose=verbose, concurrency=concurrency, run_dir=run_dir, task_configs=[x.strip() for x in (bigbench_tasks or "").split(",") if x.strip()] if bigbench_tasks else None),
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
        "criticality": 1000,
        "recall": 500,
        "matrix_recall": 200,
        "episodic_memory": 100,
        "instruction_following": 10,
        "word_instruction_following": 10,
        "summarization": 500,
        "planning": 240,
        "plan_bench": 50,
        "bigbench": 300,
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
    
    model_path = getattr(args, 'model_path', None)

    # criticality_v2 with --model-path uses sequence mode (no agent needed)
    if args.benchmark == "criticality_v2" and model_path:
        agent = None
    else:
        agent = get_agent(args.agent, args.model, verbose=args.verbose)
    
    # Get benchmark runner
    runner = get_benchmark_runner(
        args.benchmark,
        agent,
        cost_tracker,
        verbose=args.verbose,
        concurrency=args.concurrency,
        model_path=model_path,
        model_name=args.model,
        list_size=getattr(args, 'list_size', 10),
        plan_bench_task=getattr(args, 'plan_bench_task', None),
        plan_bench_config=getattr(args, 'plan_bench_config', None),
        bigbench_tasks=getattr(args, 'bigbench_tasks', None),
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
    if metrics.num_evaluated is not None and metrics.num_evaluated == 0 and metrics.num_tasks > 0:
        print(f"Success Rate: n/a (0/{metrics.num_tasks} evaluated)")
    elif metrics.num_evaluated is not None and metrics.num_evaluated < metrics.num_tasks:
        print(
            f"Success Rate: {(metrics.success_rate or 0.0) * 100:.1f}% "
            f"({metrics.num_evaluated}/{metrics.num_tasks} evaluated)"
        )
    else:
        print(f"Success Rate: {(metrics.success_rate or 0.0) * 100:.1f}%")
    print(f"Average Confidence: {metrics.avg_confidence:.2f}")
    print(f"Average Latency: {metrics.avg_latency:.2f}s")
    print(f"Total Cost: ${metrics.total_cost:.2f}")
    print(f"{'=' * 70}\n")
    
    # Print cost summary
    cost_tracker.print_summary()
    
    return results, metrics


def find_latest_run_dir(benchmark: str, model: str) -> Optional[Path]:
    """Find the most recent run directory for a given benchmark/model."""
    results_dir = Path("results") / benchmark
    if not results_dir.exists():
        return None
    
    # Find directories matching pattern: {model}_{timestamp}
    matching_dirs = []
    for d in results_dir.iterdir():
        if d.is_dir() and d.name.startswith(f"{model}_"):
            matching_dirs.append(d)
    
    if not matching_dirs:
        return None
    
    # Sort by name (timestamp) and return most recent
    matching_dirs.sort(key=lambda x: x.name, reverse=True)
    return matching_dirs[0]


def get_completed_agents(run_dir: Path) -> set:
    """Get set of agent types that have completed (have summary.json in their folder)."""
    completed = set()
    if not run_dir.exists():
        return completed
    
    # Map folder names to agent types
    agent_name_map = {
        "BaselineAgent": "none",
        "OneShotAgent": "one_shot", 
        "SequentialAgent": "sequential",
        "ConcurrentAgent": "concurrent",
        "GroupChatAgent": "group_chat",
    }
    
    # Check for agent folders with summary.json (new structure)
    for agent_dir in run_dir.iterdir():
        if agent_dir.is_dir() and agent_dir.name in agent_name_map:
            summary_file = agent_dir / "summary.json"
            if summary_file.exists():
                completed.add(agent_name_map[agent_dir.name])
    
    # Also check for old structure (*.summary.json files)
    for summary_file in run_dir.glob("*.summary.json"):
        agent_name = summary_file.stem.replace(".summary", "")
        if agent_name in agent_name_map:
            completed.add(agent_name_map[agent_name])
    
    return completed


def run_all_agents(args):
    """Run benchmark with all agent types, saving all results in one folder."""
    
    # Handle resume: use existing run_dir or create new one
    if getattr(args, 'resume', None):
        if args.resume == "latest":
            run_dir = find_latest_run_dir(args.benchmark, args.model)
            if not run_dir:
                print(f"No previous run found for {args.model}/{args.benchmark}")
                print("Starting new run...")
                run_dir = None
        else:
            run_dir = Path(args.resume)
            if not run_dir.exists():
                print(f"Resume directory not found: {run_dir}")
                return {}
    else:
        run_dir = None
    
    # Create new run directory if not resuming
    if run_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = Path("results") / args.benchmark / f"{args.model}_{timestamp}"
    else:
        # Extract timestamp from existing run_dir name (e.g., "phi-4_20260128_001234")
        dir_name = run_dir.name
        parts = dir_name.split("_")
        timestamp = "_".join(parts[-2:]) if len(parts) >= 2 else time.strftime("%Y%m%d_%H%M%S")
    
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Agentic types only by default, add baseline if --with-baseline
    agent_types = ["one_shot", "sequential", "concurrent", "group_chat"]
    if getattr(args, 'with_baseline', False):
        agent_types = ["none"] + agent_types  # Add baseline first
    
    # Check which agents already completed (for resume)
    completed_agents = get_completed_agents(run_dir)
    pending_agents = [a for a in agent_types if a not in completed_agents]
    
    print("\n" + "=" * 70)
    if completed_agents:
        print("RESUMING RUN - ALL AGENT CONFIGURATIONS")
    else:
        print("RUNNING ALL AGENT CONFIGURATIONS")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Limit: {args.limit or 'all'}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Output: {run_dir}/")
    if completed_agents:
        print(f"Completed: {', '.join(sorted(completed_agents))} ✓")
        print(f"Remaining: {', '.join(pending_agents)}")
    else:
        print(f"Agents: {', '.join(agent_types)}")
    print("=" * 70)
    
    if not pending_agents:
        print("\n✅ All agents already completed!")
        return {}
    
    cost_tracker = CostTracker(
        budget_limit=args.budget or 10000.0,
        alert_thresholds=[0.3, 0.6, 0.9],
        log_file="cost_tracking.json",
    )
    
    all_results = {}
    
    for agent_type in pending_agents:
        print(f"\n{'=' * 70}")
        print(f"AGENT: {agent_type.upper()}")
        print(f"{'=' * 70}")
        
        try:
            agent = get_agent(agent_type, args.model, verbose=args.verbose)
            
            runner = get_benchmark_runner(
                args.benchmark,
                agent,
                cost_tracker,
                verbose=args.verbose,
                concurrency=args.concurrency,
                run_dir=run_dir,  # All agents save to same folder
                model_path=getattr(args, 'model_path', None),
                list_size=getattr(args, 'list_size', 10),
                plan_bench_task=getattr(args, 'plan_bench_task', None),
                plan_bench_config=getattr(args, 'plan_bench_config', None),
                bigbench_tasks=getattr(args, 'bigbench_tasks', None),
            )
            
            results = runner.run(limit=args.limit, save_results=True)
            metrics = calculate_metrics(results)
            
            all_results[agent_type] = {
                "success_rate": metrics.success_rate,
                "avg_latency": metrics.avg_latency,
                "total_cost": metrics.total_cost,
                "num_tasks": metrics.num_tasks,
                "num_evaluated": metrics.num_evaluated,
            }
            
            if metrics.num_evaluated is not None and metrics.num_evaluated == 0 and metrics.num_tasks > 0:
                print(f"\n  Success Rate: n/a (0/{metrics.num_tasks} evaluated)")
            elif metrics.num_evaluated is not None and metrics.num_evaluated < metrics.num_tasks:
                print(
                    f"\n  Success Rate: {(metrics.success_rate or 0.0) * 100:.1f}% "
                    f"({metrics.num_evaluated}/{metrics.num_tasks} evaluated)"
                )
            else:
                print(f"\n  Success Rate: {(metrics.success_rate or 0.0) * 100:.1f}%")
            print(f"  Avg Latency: {metrics.avg_latency:.2f}s")
            print(f"  Cost: ${metrics.total_cost:.4f}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[agent_type] = {"error": str(e)}
    
    # Save run summary
    import json
    summary_file = run_dir / "run_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "model": args.model,
            "benchmark": args.benchmark,
            "limit": args.limit,
            "concurrency": args.concurrency,
            "timestamp": timestamp,
            "results": all_results,
        }, f, indent=2)
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY - ALL AGENTS")
    print("=" * 70)
    print(f"\nResults saved to: {run_dir}/")
    print(f"\n{'Agent':<15} {'Success %':>10} {'Latency':>10} {'Cost':>10}")
    print("-" * 50)
    
    for agent_type, result in all_results.items():
        if "error" in result:
            print(f"{agent_type:<15} {'ERROR':>10}")
        else:
            sr = result.get("success_rate")
            success_txt = "   n/a" if sr is None else f"{sr*100:>8.1f}%"
            print(f"{agent_type:<15} {success_txt:>10} {result['avg_latency']:>9.1f}s ${result['total_cost']:>8.4f}")
    
    print("=" * 70)
    cost_tracker.print_summary()
    
    return all_results


def dry_run(args):
    """Run all agent types with 1 task each to verify setup."""
    print("\n" + "=" * 70)
    print("DRY RUN - Testing all agent configurations")
    print("=" * 70)
    
    # Agentic types only by default
    agent_types = ["one_shot", "sequential", "concurrent", "group_chat"]
    if getattr(args, 'with_baseline', False):
        agent_types = ["none"] + agent_types
    benchmark = args.benchmark if args.benchmark != "all" else "medqa"
    
    results = {}
    
    for agent_type in agent_types:
        print(f"\n{'─' * 70}")
        print(f"Testing: {agent_type} agent with {args.model}")
        print(f"{'─' * 70}")
        
        try:
            # Get agent
            agent = get_agent(agent_type, args.model, verbose=args.verbose)
            
            # Get benchmark runner with limit=1
            runner = get_benchmark_runner(
                benchmark,
                agent,
                cost_tracker=None,
                verbose=args.verbose,
                concurrency=1,
                model_path=getattr(args, 'model_path', None),
                plan_bench_task=getattr(args, 'plan_bench_task', None),
                plan_bench_config=getattr(args, 'plan_bench_config', None),
                bigbench_tasks=getattr(args, 'bigbench_tasks', None),
            )
            
            # Run single task
            task_results = runner.run(limit=1, save_results=False)
            
            if task_results:
                r = task_results[0]
                success = r.success if hasattr(r, 'success') else False
                latency = r.latency if hasattr(r, 'latency') else 0
                results[agent_type] = {
                    "status": "✓ PASS" if success else "○ RAN",
                    "latency": latency,
                    "success": success,
                }
                print(f"  Result: {'✓ Correct' if success else '✗ Incorrect'}")
                print(f"  Latency: {latency:.2f}s")
            else:
                results[agent_type] = {"status": "✗ NO RESULT", "latency": 0, "success": False}
                
        except Exception as e:
            results[agent_type] = {"status": f"✗ ERROR", "error": str(e)[:100], "success": False}
            print(f"  Error: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("DRY RUN SUMMARY")
    print("=" * 70)
    print(f"\nModel: {args.model}")
    print(f"Benchmark: {benchmark}")
    print()
    
    for agent_type, result in results.items():
        status = result["status"]
        latency = result.get("latency", 0)
        error = result.get("error", "")
        if error:
            print(f"  {agent_type:15s} {status} - {error}")
        else:
            print(f"  {agent_type:15s} {status} ({latency:.1f}s)")
    
    passed = sum(1 for r in results.values() if "ERROR" not in r["status"])
    print(f"\n  {passed}/{len(agent_types)} agents working")
    print("=" * 70 + "\n")
    
    return results


def run_all_models(args):
    """Run a benchmark across all serverless models concurrently."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    models = list(list_models(serverless_only=True).keys())

    # Default exclusions: deepseek models are too slow for concurrent sweeps
    DEFAULT_EXCLUDE = {"deepseek-r1", "deepseek-v3", "deepseek-v3.2"}
    exclude_str = getattr(args, "exclude_models", None)
    if exclude_str:
        exclude = set(exclude_str.split(","))
    else:
        exclude = DEFAULT_EXCLUDE
    models = [m for m in models if m not in exclude]

    agent_type = args.agent
    benchmark = args.benchmark

    print("\n" + "=" * 70)
    print("RUNNING ALL MODELS (concurrent)")
    print("=" * 70)
    print(f"Benchmark: {benchmark}")
    print(f"Agent: {agent_type}")
    print(f"Models ({len(models)}): {', '.join(models)}")
    print(f"Limit: {args.limit or 'all'}")
    print(f"Model concurrency: {len(models)} (one thread per model)")
    print("=" * 70)

    cost_tracker = CostTracker(
        budget_limit=args.budget or 10000.0,
        alert_thresholds=[0.3, 0.6, 0.9],
        log_file="cost_tracking.json",
    )

    all_results = {}

    def _run_one_model(model_name: str):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = Path("results") / benchmark / f"{model_name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            agent = get_agent(agent_type, model_name, verbose=args.verbose)
            runner = get_benchmark_runner(
                benchmark, agent, cost_tracker,
                verbose=args.verbose,
                concurrency=1,
                run_dir=run_dir,
                model_path=getattr(args, "model_path", None),
                model_name=model_name,
                list_size=getattr(args, "list_size", 10),
                plan_bench_task=getattr(args, "plan_bench_task", None),
                plan_bench_config=getattr(args, "plan_bench_config", None),
                bigbench_tasks=getattr(args, "bigbench_tasks", None),
            )
            results = runner.run(limit=args.limit, save_results=True)
            metrics = calculate_metrics(results)
            avg_score = round(
                sum(((r.get("score") if isinstance(r, dict) else getattr(r, "score", 0)) or 0) for r in results)
                / max(len(results), 1),
                4,
            )
            return model_name, {
                "success_rate": metrics.success_rate,
                "num_evaluated": metrics.num_evaluated,
                "avg_score": avg_score,
                "avg_latency": metrics.avg_latency,
                "total_cost": metrics.total_cost,
                "num_tasks": metrics.num_tasks,
                "run_dir": str(run_dir),
            }
        except Exception as e:
            print(f"\n  [{model_name}] ERROR: {e}")
            return model_name, {"error": str(e)}

    # plan_bench uses chdir + vendored imports; run sequentially to avoid import/cwd races
    max_workers = 1 if benchmark == "plan_bench" else len(models)
    if max_workers == 1 and len(models) > 1:
        print(f"Note: plan_bench runs one model at a time (sequential).\n")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_one_model, m): m for m in models}
        for future in as_completed(futures):
            model_name, result = future.result()
            all_results[model_name] = result
            if "error" not in result:
                print(
                    f"\n  [{model_name}] done — "
                    f"score={result['avg_score']:.3f}  "
                    f"success={(result.get('success_rate') or 0.0)*100:.0f}%  "
                    f"latency={result['avg_latency']:.1f}s  "
                    f"cost=${result['total_cost']:.4f}"
                )

    # Save cross-model summary
    import json as _json
    summary_path = Path("results") / benchmark / "all_models_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        _json.dump({
            "benchmark": benchmark,
            "agent": agent_type,
            "limit": args.limit,
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "results": all_results,
        }, f, indent=2)

    # Print final table
    print("\n" + "=" * 70)
    print("ALL MODELS — FINAL SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<28} {'Score':>8} {'Success':>9} {'Latency':>9} {'Cost':>10}")
    print("-" * 68)

    for model_name in models:
        r = all_results.get(model_name, {})
        if "error" in r:
            print(f"{model_name:<28} {'ERROR':>8}   {r['error'][:30]}")
        else:
            print(
                f"{model_name:<28} {r['avg_score']:>8.3f} "
                f"{((r.get('success_rate') if isinstance(r, dict) else 0.0) or 0.0)*100:>8.1f}% "
                f"{r['avg_latency']:>8.1f}s "
                f"${r['total_cost']:>8.4f}"
            )

    print("-" * 68)
    print(f"\nSummary saved to: {summary_path}")
    print("=" * 70)
    cost_tracker.print_summary()

    return all_results


def compare_baseline(args):
    """Compare agentic vs non-agentic baseline."""
    print("\n" + "=" * 70)
    print("COMPARING AGENTIC VS NON-AGENTIC")
    print("=" * 70)
    
    cost_tracker = CostTracker(budget_limit=args.budget or 10000.0)
    
    # Run baseline (non-agentic)
    print("\n1. Running non-agentic baseline...")
    baseline_agent = get_baseline_agent(model="llama-3.3-70b", verbose=args.verbose)
    baseline_runner = get_benchmark_runner(
        args.benchmark,
        baseline_agent,
        cost_tracker,
        verbose=args.verbose,
        model_path=getattr(args, 'model_path', None),
        plan_bench_task=getattr(args, 'plan_bench_task', None),
        plan_bench_config=getattr(args, 'plan_bench_config', None),
        bigbench_tasks=getattr(args, 'bigbench_tasks', None),
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
        model_path=getattr(args, 'model_path', None),
        plan_bench_task=getattr(args, 'plan_bench_task', None),
        plan_bench_config=getattr(args, 'plan_bench_config', None),
        bigbench_tasks=getattr(args, 'bigbench_tasks', None),
    )
    agentic_results = agentic_runner.run(limit=args.limit, save_results=True)
    agentic_metrics = calculate_metrics(agentic_results)
    
    # Compare
    from src.evaluation.metrics import compare_metrics
    comparison = compare_metrics(baseline_metrics, agentic_metrics)
    
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"Baseline Success Rate: {(baseline_metrics.success_rate or 0.0) * 100:.1f}%")
    print(f"Agentic Success Rate:  {(agentic_metrics.success_rate or 0.0) * 100:.1f}%")
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

  # Run non-agentic baseline (uses llama-3.3-70b for comparison)
  python run_benchmark.py --model phi-4 --agent baseline --benchmark medqa

  # Run agentic architecture
  python run_benchmark.py --model phi-4 --agent sequential --benchmark medagent

  # Compare agentic vs baseline
  python run_benchmark.py --compare-baseline --model phi-4 --agent sequential --benchmark medqa

  # Run across all models concurrently
  python run_benchmark.py --all-models --benchmark word_instruction_following --agent one_shot --limit 5

  # Same but skip expensive models
  python run_benchmark.py --all-models --benchmark word_instruction_following --agent one_shot --limit 5 --exclude-models gpt-4o,mistral-large-3,deepseek-r1

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
        choices=["medagent", "medqa", "medmcqa", "mcp", "bfcl", "criticality", "criticality_v2", "recall", "matrix_recall", "matrix_recall_xhard", "episodic_memory", "instruction_following", "word_instruction_following", "summarization", "planning", "plan_bench", "bigbench", "all"],
        help="Benchmark to run (default: medqa)",
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tasks (for testing)",
    )
    
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=1,
        help="Number of concurrent requests (default: 1)",
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to GGUF model file for criticality_v2 sequence mode (e.g., Ollama blob path)",
    )
    
    parser.add_argument(
        "--list-size",
        type=int,
        default=None,
        help="Fixed list size for word_instruction_following. If omitted, sweeps sizes 1–10.",
    )
    parser.add_argument(
        "--plan-bench-task",
        type=str,
        default="t1",
        help="PlanBench task (t1–t8_3). Default: t1.",
    )
    parser.add_argument(
        "--plan-bench-config",
        type=str,
        default="blocksworld",
        help="PlanBench config/domain (e.g. blocksworld, depots, logistics). Default: blocksworld.",
    )
    parser.add_argument(
        "--bigbench-tasks",
        type=str,
        default=None,
        metavar="CONFIG1,CONFIG2,...",
        help="Comma-separated BIG-bench task configs (e.g. logical_deduction,navigate). Default: 6-task set.",
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
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all agent types with 1 task each to verify setup",
    )
    
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run benchmark across all serverless models concurrently. Use --exclude-models to skip specific models.",
    )
    
    parser.add_argument(
        "--exclude-models",
        type=str,
        default=None,
        help="Comma-separated models to skip when using --all-models (e.g., 'gpt-4o,mistral-large-3')",
    )
    
    parser.add_argument(
        "--all-agents",
        action="store_true",
        help="Run benchmark with all agentic types (one_shot, sequential, concurrent, group_chat). Use --with-baseline to also include baseline.",
    )
    
    parser.add_argument(
        "--with-baseline",
        action="store_true",
        help="Include baseline (non-agentic) agent when using --all-agents",
    )
    
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        metavar="RUN_DIR",
        help="Resume a previous run. Use without value for latest run, or specify run directory path.",
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
    
    # Dry run if requested
    if args.dry_run:
        dry_run(args)
        return
    
    # Run all models if requested
    if args.all_models:
        run_all_models(args)
        return
    
    # Run all agents if requested
    if args.all_agents:
        run_all_agents(args)
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
