#!/usr/bin/env python3
"""
Run PlanBench on Azure AI models.

Usage:
    python scripts/run_azure_planbench.py
    python scripts/run_azure_planbench.py --models phi-4-mini,phi-4-mini-reasoning
    python scripts/run_azure_planbench.py --tasks all --configs all --limit 100
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.agents.baseline_agent import BaselineAgent
from src.benchmarks import PlanBenchRunner
from src.config.azure_llm_config import AVAILABLE_MODELS


# Azure models confirmed to handle high call volumes without daily rate limits.
# phi-4-mini and phi-4-mini-reasoning are excluded — they hit a 150 calls/day
# free-tier cap (Azure AI Foundry serverless Phi small models).
AZURE_PLANBENCH_MODELS = {
    "ministral-3b": "Ministral-3B (3B)",
    "mistral-nemo": "Mistral Nemo (12B)",
    "phi-4":        "Phi-4 (14B)",
}

_PB_ALL_TASKS   = ["t1","t2","t3","t4","t5","t6","t7","t8_1","t8_2","t8_3"]
_PB_ALL_CONFIGS = ["blocksworld", "blocksworld_3"]


def main():
    parser = argparse.ArgumentParser(description="Run PlanBench on Azure models")
    parser.add_argument(
        "--models", default="all",
        help=f"Comma-separated model keys or 'all'. Available: {list(AZURE_PLANBENCH_MODELS)}",
    )
    parser.add_argument(
        "--tasks", default="all",
        help="Task IDs: comma-separated (t1,t2,...) or 'all'",
    )
    parser.add_argument(
        "--configs", default="all",
        help="Configs: comma-separated or 'all' (blocksworld,blocksworld_3)",
    )
    parser.add_argument("--limit", "-n", type=int, default=100, help="Instances per task")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Resolve models
    if args.models == "all":
        model_keys = list(AZURE_PLANBENCH_MODELS.keys())
    else:
        model_keys = [m.strip() for m in args.models.split(",")]
        for mk in model_keys:
            if mk not in AZURE_PLANBENCH_MODELS and mk not in AVAILABLE_MODELS:
                print(f"Unknown model: {mk}. Available: {list(AZURE_PLANBENCH_MODELS)}")
                sys.exit(1)

    tasks   = _PB_ALL_TASKS   if args.tasks   == "all" else [t.strip() for t in args.tasks.split(",")]
    configs = _PB_ALL_CONFIGS if args.configs == "all" else [c.strip() for c in args.configs.split(",")]

    # Quick connectivity check
    import os, urllib.request
    key  = os.environ.get("AZURE_API_KEY", "")
    base = os.environ.get("AZURE_AI_ENDPOINT", "<AZURE_ENDPOINT>")
    try:
        req = urllib.request.Request(
            base.rstrip("/") + "/models",
            headers={"Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=8) as r:
            n_models = len(json.loads(r.read()).get("data", []))
        print(f"Azure endpoint reachable — {n_models} models available")
    except Exception as e:
        print(f"Cannot reach Azure endpoint: {e}")
        sys.exit(1)

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_base = Path("results") / "azure_planbench" / timestamp

    print(f"\n{'=' * 70}")
    print("AZURE PLANBENCH RUN")
    print(f"{'=' * 70}")
    print(f"Models:   {', '.join(model_keys)}")
    print(f"Tasks:    {', '.join(tasks)}")
    print(f"Configs:  {', '.join(configs)}")
    print(f"Limit:    {args.limit} per task")
    print(f"Results:  {results_base}")
    print(f"{'=' * 70}\n")

    all_summaries = []

    for model_key in model_keys:
        desc = AZURE_PLANBENCH_MODELS.get(model_key, model_key)
        print(f"\n{'─' * 70}")
        print(f"MODEL: {model_key} ({desc})")
        print(f"{'─' * 70}")

        agent = BaselineAgent(model=model_key, verbose=args.verbose)

        for pb_config in configs:
            for pb_task in tasks:
                bench_label = f"plan_bench:{pb_config}:{pb_task}"
                print(f"\n  >>> {bench_label.upper()} (limit={args.limit})")
                sub_run_dir = results_base / model_key / pb_config / pb_task
                sub_run_dir.mkdir(parents=True, exist_ok=True)

                start = time.time()
                try:
                    runner = PlanBenchRunner(
                        agent, task=pb_task, config=pb_config, run_dir=sub_run_dir,
                        cost_tracker=None, verbose=args.verbose,
                    )
                    results = runner.run(limit=args.limit, save_results=True)
                    elapsed = time.time() - start

                    if results:
                        correct = sum(1 for r in results if r.get("success", False))
                        total   = len(results)
                        accuracy = correct / total if total > 0 else 0
                        summary = {
                            "model": model_key, "benchmark": bench_label,
                            "accuracy": accuracy, "correct": correct,
                            "total": total, "elapsed_seconds": round(elapsed, 1),
                            "avg_latency": round(elapsed / total, 2) if total else 0,
                        }
                        all_summaries.append(summary)
                        print(f"  <<< {bench_label.upper()}: {accuracy*100:.1f}% ({correct}/{total}) in {elapsed:.0f}s")
                    else:
                        print(f"  <<< {bench_label.upper()}: no results in {elapsed:.0f}s")

                except Exception as e:
                    elapsed = time.time() - start
                    print(f"  <<< {bench_label.upper()}: FAILED after {elapsed:.0f}s -- {e}")
                    all_summaries.append({
                        "model": model_key, "benchmark": bench_label,
                        "error": str(e), "elapsed_seconds": round(elapsed, 1),
                    })

    # Final summary
    print(f"\n\n{'=' * 70}")
    print("FINAL RESULTS")
    print(f"{'=' * 70}")
    print(f"{'Model':<26} {'Benchmark':<32} {'Accuracy':>10} {'N':>5} {'Time':>8}")
    print(f"{'─' * 70}")
    for s in all_summaries:
        if "error" in s:
            print(f"{s['model']:<26} {s['benchmark']:<32} {'ERROR':>10} {'':>5} {s['elapsed_seconds']:>7.0f}s")
        else:
            acc_str = f"{s['accuracy']*100:.1f}%"
            print(f"{s['model']:<26} {s['benchmark']:<32} {acc_str:>10} {s['total']:>5} {s['elapsed_seconds']:>7.0f}s")

    summary_file = results_base / "summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
