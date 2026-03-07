#!/usr/bin/env python3
"""
Run BIG-bench comparison: one-shot (all models), skill-routed agentic, and
uniform gpt-oss-20b agentic. Prints a comparison table by task × condition.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any

# Repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.agents import OneShotAgent
from src.agents.skill_routed_agent import (
    SkillRoutedSequentialAgent,
    SkillRoutedConcurrentAgent,
    SkillRoutedGroupChatAgent,
    build_role_models_from_skills,
    SEQUENTIAL_ROLE_TO_SKILL,
    CONCURRENT_ROLE_TO_SKILL,
    GROUPCHAT_ROLE_TO_SKILL,
)
from src.benchmarks import BigBenchRunner
from src.benchmarks.skills.bigbench import DEFAULT_TASK_CONFIGS
from src.evaluation import CostTracker, calculate_metrics

def _get_best_models():
    """Load get_best_models_per_skill from scripts/get_best_models_per_skill.py."""
    import importlib.util
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    spec = importlib.util.spec_from_file_location(
        "get_best_models_per_skill",
        script_dir / "get_best_models_per_skill.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_best_models_per_skill(results_root=repo_root / "results", default_model="phi-4", azure_only=True)


def run_one_condition(
    condition_name: str,
    agent,
    runner: BigBenchRunner,
    limit: Optional[int],
) -> Dict[str, Any]:
    """Run one (agent, runner) and return accuracy and per-config breakdown."""
    results = runner.run(limit=limit, save_results=True)
    metrics = calculate_metrics(results)
    breakdown = {}
    if hasattr(runner, "_task_stats"):
        breakdown = runner._task_stats
    return {
        "condition": condition_name,
        "accuracy": (metrics.success_rate or 0.0) * 100,
        "num_tasks": len(results),
        "total_cost": metrics.total_cost,
        "avg_latency": metrics.avg_latency,
        "breakdown_by_config": breakdown,
    }


def main():
    parser = argparse.ArgumentParser(description="BIG-bench: one-shot, skill-routed, and uniform gpt-oss-20b comparison")
    parser.add_argument("--limit", type=int, default=30, help="Max tasks per run (default: 30)")
    parser.add_argument("--models", type=str, default="phi-4,gpt-oss-20b", help="Comma-separated models for one-shot sweep")
    parser.add_argument("--results-dir", type=Path, default=Path("results/bigbench"), help="Base dir for results")
    parser.add_argument("--uniform-model", type=str, default="gpt-oss-20b", help="Model for uniform agentic baseline (default: gpt-oss-20b; use an Azure key if get_llm does not support Ollama)")
    parser.add_argument("--exclude-skill-models", type=str, default="gpt-4o,mistral-large-3,deepseek-r1,deepseek-v3,deepseek-v3.2", metavar="M1,M2", help="Comma-separated model keys to exclude from skill-routed selection (default: rate-limited models)")
    parser.add_argument("--skip-one-shot", action="store_true", help="Skip one-shot sweep")
    parser.add_argument("--skip-skill-routed", action="store_true", help="Skip skill-routed agentic runs")
    parser.add_argument("--skip-uniform", action="store_true", help="Skip uniform agentic runs")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_dir = args.results_dir / f"comparison_{timestamp}"
    base_dir.mkdir(parents=True, exist_ok=True)

    cost_tracker = CostTracker(budget_limit=10000.0, alert_thresholds=[0.3, 0.6, 0.9], log_file="cost_tracking.json")
    task_configs = DEFAULT_TASK_CONFIGS
    one_shot_models = [m.strip() for m in args.models.split(",") if m.strip()]

    all_results: List[Dict[str, Any]] = []

    # 1) One-shot sweep
    if not args.skip_one_shot:
        print("\n" + "=" * 70)
        print("ONE-SHOT (per model)")
        print("=" * 70)
        for model in one_shot_models:
            run_dir = base_dir / "one_shot" / model
            run_dir.mkdir(parents=True, exist_ok=True)
            agent = OneShotAgent(model=model, verbose=args.verbose)
            runner = BigBenchRunner(agent, cost_tracker=cost_tracker, verbose=args.verbose, concurrency=1, run_dir=run_dir, task_configs=task_configs)
            row = run_one_condition(f"one_shot_{model}", agent, runner, args.limit)
            all_results.append(row)
            print(f"  {model}: {row['accuracy']:.1f}% ({row['num_tasks']} tasks)")

    # 2) Skill-routed agentic (best model per role from existing results)
    if not args.skip_skill_routed:
        print("\n" + "=" * 70)
        print("SKILL-ROUTED AGENTIC (best model per skill)")
        print("=" * 70)
        exclude = {m.strip() for m in (args.exclude_skill_models or "").split(",") if m.strip()}
        skill_to_model_raw = _get_best_models()
        skill_to_model = {
            skill: (model if model not in exclude else "phi-4")
            for skill, model in skill_to_model_raw.items()
        }
        print(f"  Skill→model: {skill_to_model}")
        for agent_cls, role_to_skill, name in [
            (SkillRoutedSequentialAgent, SEQUENTIAL_ROLE_TO_SKILL, "skill_routed_sequential"),
            (SkillRoutedConcurrentAgent, CONCURRENT_ROLE_TO_SKILL, "skill_routed_concurrent"),
            (SkillRoutedGroupChatAgent, GROUPCHAT_ROLE_TO_SKILL, "skill_routed_group_chat"),
        ]:
            role_models = build_role_models_from_skills(skill_to_model, role_to_skill, default_model="phi-4")
            run_dir = base_dir / name
            run_dir.mkdir(parents=True, exist_ok=True)
            agent = agent_cls(model="phi-4", verbose=args.verbose, role_models=role_models)
            runner = BigBenchRunner(agent, cost_tracker=cost_tracker, verbose=args.verbose, concurrency=1, run_dir=run_dir, task_configs=task_configs)
            row = run_one_condition(name, agent, runner, args.limit)
            all_results.append(row)
            print(f"  {name}: {row['accuracy']:.1f}% ({row['num_tasks']} tasks)")

    # 3) Uniform agentic (all roles use same model, e.g. gpt-oss-20b)
    if not args.skip_uniform:
        uniform_model = args.uniform_model
        print("\n" + "=" * 70)
        print(f"UNIFORM AGENTIC ({uniform_model})")
        print("=" * 70)
        for agent_cls, name in [
            (SkillRoutedSequentialAgent, "uniform_sequential"),
            (SkillRoutedConcurrentAgent, "uniform_concurrent"),
            (SkillRoutedGroupChatAgent, "uniform_group_chat"),
        ]:
            run_dir = base_dir / name
            run_dir.mkdir(parents=True, exist_ok=True)
            agent = agent_cls(model=uniform_model, verbose=args.verbose, role_models={})
            runner = BigBenchRunner(agent, cost_tracker=cost_tracker, verbose=args.verbose, concurrency=1, run_dir=run_dir, task_configs=task_configs)
            row = run_one_condition(name, agent, runner, args.limit)
            all_results.append(row)
            print(f"  {name}: {row['accuracy']:.1f}% ({row['num_tasks']} tasks)")

    # Comparison table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE (accuracy %)")
    print("=" * 70)
    print(f"\n{'Condition':<35} {'Accuracy':>10} {'Tasks':>8} {'Cost':>10}")
    print("-" * 68)
    for row in all_results:
        print(f"{row['condition']:<35} {row['accuracy']:>9.1f}% {row['num_tasks']:>8} ${row['total_cost']:>8.4f}")
    print("-" * 68)
    print(f"\nResults saved under: {base_dir}")

    summary_path = base_dir / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"timestamp": timestamp, "limit": args.limit, "task_configs": task_configs, "results": all_results}, f, indent=2)
    print(f"Summary: {summary_path}")
    cost_tracker.print_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
