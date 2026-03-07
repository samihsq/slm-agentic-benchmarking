#!/usr/bin/env python3
"""
PlanBench live integration test: run pipeline for one task/config with a real model.
Use RUN_PLANBENCH_API=1 to enable (e.g. in CI). Skips evaluation if VAL is not set.

Usage:
  RUN_PLANBENCH_API=1 python scripts/test_plan_bench_live.py --task t1 --config blocksworld --limit 2 --model gpt-4o
  python scripts/test_plan_bench_live.py  # skips if RUN_PLANBENCH_API not set
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents import OneShotAgent
from src.benchmarks.skills.plan_bench.runner import PlanBenchRunner


def main():
    parser = argparse.ArgumentParser(description="PlanBench live test (real API)")
    parser.add_argument("--task", default="t1", help="PlanBench task (default: t1)")
    parser.add_argument("--config", default="blocksworld", help="Config/domain (default: blocksworld)")
    parser.add_argument("--limit", type=int, default=2, help="Number of instances (default: 2)")
    parser.add_argument("--model", default="gpt-4o", help="Model name (default: gpt-4o)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if os.environ.get("RUN_PLANBENCH_API") != "1":
        print("Skipping live test (set RUN_PLANBENCH_API=1 to run)")
        return 0

    agent = OneShotAgent(model=args.model, verbose=args.verbose)
    runner = PlanBenchRunner(
        agent,
        task=args.task,
        config=args.config,
        limit=args.limit,
        verbose=args.verbose,
    )
    results = runner.run(limit=args.limit, save_results=True)

    if not results:
        print("No results returned")
        return 1
    for r in results:
        if not isinstance(r.get("llm_raw_response"), str):
            print("Invalid result shape:", r)
            return 1
        if len((r.get("llm_raw_response") or "").strip()) == 0:
            print("Empty response for instance_id:", r.get("instance_id"))
            return 1
    print(f"OK: {len(results)} instances, non-empty responses")
    if os.environ.get("VAL"):
        correct = sum(1 for r in results if r.get("llm_correct"))
        print(f"Evaluated (VAL set): {correct}/{len(results)} correct")
    else:
        print("Evaluation skipped (VAL not set)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
