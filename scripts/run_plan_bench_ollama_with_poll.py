#!/usr/bin/env python3
"""
Run PlanBench on all Ollama models in the background and poll progress every minute.
Reports progress and valid results (num_tasks > 0) until the run completes.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent


def find_latest_ollama_run():
    """Return the most recent results/ollama/<timestamp> path, or None."""
    ollama_dir = ROOT / "results" / "ollama"
    if not ollama_dir.exists():
        return None
    subdirs = [p for p in ollama_dir.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    return max(subdirs, key=lambda p: p.stat().st_mtime)


def poll_plan_bench_results(run_dir: Path):
    """Collect plan_bench summaries from run_dir/plan_bench/<model>/OllamaAgent/summary.json."""
    plan_dir = run_dir / "plan_bench"
    if not plan_dir.exists():
        return {}
    out = {}
    for model_dir in plan_dir.iterdir():
        if not model_dir.is_dir():
            continue
        summary_file = model_dir / "OllamaAgent" / "summary.json"
        if not summary_file.exists():
            out[model_dir.name] = {"status": "running", "num_tasks": None, "success_rate": None}
            continue
        try:
            data = json.loads(summary_file.read_text())
            out[model_dir.name] = {
                "status": "done",
                "num_tasks": data.get("num_tasks", 0),
                "success_rate": data.get("success_rate", 0),
                "total_correct": data.get("total_correct", 0),
            }
        except Exception as e:
            out[model_dir.name] = {"status": "error", "error": str(e)}
    return out


def main():
    limit = 5  # Keep run bounded; increase for full sweep
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_ollama_benchmarks.py"),
        "--benchmarks", "plan_bench",
        "--limit", str(limit),
    ]
    print(f"Starting: {' '.join(cmd)}")
    print("Polling every 60s. Valid = num_tasks > 0.\n")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    last_run_dir = None
    while True:
        time.sleep(60)
        if proc.poll() is not None:
            break
        run_dir = find_latest_ollama_run()
        if run_dir and run_dir != last_run_dir:
            last_run_dir = run_dir
        if not run_dir:
            print("[poll] No results/ollama run dir yet")
            continue
        results = poll_plan_bench_results(run_dir)
        if not results:
            print(f"[poll] {run_dir.name} — no plan_bench results yet")
            continue
        valid = sum(1 for r in results.values() if isinstance(r.get("num_tasks"), int) and r.get("num_tasks", 0) > 0)
        print(f"[poll] {run_dir.name} — models: {len(results)} | valid (num_tasks>0): {valid}/{len(results)}")
        for model, r in sorted(results.items()):
            if r.get("status") == "done":
                n = r.get("num_tasks") or 0
                sr = r.get("success_rate") or 0
                ok = "✓" if n > 0 else "○"
                print(f"       {ok} {model}: num_tasks={n} success_rate={sr:.2%}")
            else:
                print(f"       … {model}: {r.get('status', 'running')}")

    # Final poll and summary
    run_dir = find_latest_ollama_run()
    if run_dir:
        results = poll_plan_bench_results(run_dir)
        valid = sum(1 for r in results.values() if isinstance(r.get("num_tasks"), int) and r.get("num_tasks", 0) > 0)
        print(f"\nDone. Run dir: {run_dir}")
        print(f"Models: {len(results)} | Valid results (num_tasks>0): {valid}/{len(results)}")
        for model, r in sorted(results.items()):
            if r.get("status") == "done":
                n = r.get("num_tasks") or 0
                sr = r.get("success_rate") or 0
                print(f"  {model}: num_tasks={n} success_rate={sr:.2%}")
    if proc.returncode != 0:
        print(f"\nProcess exited with code {proc.returncode}")
        sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
