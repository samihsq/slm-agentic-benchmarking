#!/usr/bin/env python3
"""
Run benchmark in background and poll results so you can fix in real time if issues arise.

Usage:
  python scripts/run_with_monitor.py --runner azure --benchmarks summarization -n 5 --poll 10
  python scripts/run_with_monitor.py --runner ollama --benchmarks summarization -n 5 --poll 15
  python scripts/run_with_monitor.py --poll-only --results-dir summarization,ollama
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = ROOT / "results"


def poll_results(results_dirs, poll_interval):
    state = {"summaries": [], "counts": {}, "errors": []}
    for base in results_dirs:
        if not base.exists():
            continue
        for path in base.rglob("summary.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                state["summaries"].append({"path": str(path.relative_to(ROOT)), "data": data})
            except Exception as e:
                state["errors"].append(f"{path}: {e}")
        for path in base.rglob("results.jsonl"):
            try:
                lines = sum(1 for _ in path.open(encoding="utf-8") if _.strip())
                state["counts"][str(path.relative_to(ROOT))] = lines
            except Exception as e:
                state["errors"].append(f"{path}: {e}")
    return state


def print_poll(state, prefix="  "):
    for s in state.get("summaries", [])[-10:]:
        d = s["data"]
        model = d.get("model", "?")
        num = d.get("num_tasks", 0)
        rate = d.get("success_rate")
        avg = d.get("avg_score")
        pct = f"{float(rate)*100:.1f}%" if rate is not None else "-"
        score = f"avg={float(avg):.3f}" if avg is not None else ""
        print(f"{prefix}{model}: {num} tasks, {pct} success {score}".strip())
    for k, v in list(state.get("counts", {}).items())[-15:]:
        print(f"{prefix}  {k}: {v} lines")
    for e in state.get("errors", []):
        print(f"{prefix}  WARN {e}")


def run_azure(args):
    cmd = [sys.executable, str(ROOT / "benchmark_runner.py"), "--models", args.models,
           "--benchmarks", args.benchmarks, "--agents", args.agents, "-n", str(args.limit),
           "-c", str(args.concurrency)]
    if "summarization" in (args.benchmarks or "").split(","):
        cmd += ["--summarization-split", getattr(args, "summarization_split", "validation")]
    return subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=dict(os.environ))


def run_ollama(args):
    cmd = [sys.executable, str(ROOT / "scripts" / "run_ollama_benchmarks.py"), "--models", args.models,
           "--benchmarks", args.benchmarks, "-n", str(args.limit), "-c", str(args.concurrency)]
    if getattr(args, "verbose", False):
        cmd.append("--verbose")
    return subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=dict(os.environ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", choices=["azure", "ollama"], default="azure")
    ap.add_argument("--benchmarks", default="summarization")
    ap.add_argument("--models", default="all")
    ap.add_argument("--agents", default="oneshot")
    ap.add_argument("-n", "--limit", type=int, default=5)
    ap.add_argument("-c", "--concurrency", type=int, default=2)
    ap.add_argument("--summarization-split", default="validation")
    ap.add_argument("--poll", type=int, default=15)
    ap.add_argument("--poll-only", action="store_true")
    ap.add_argument("--results-dir", type=str, default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.poll_only:
        dirs = [RESULTS_ROOT / d.strip() for d in (args.results_dir or "summarization,ollama").split(",")]
        print("Polling every", args.poll, "s:", [str(d) for d in dirs])
        while True:
            state = poll_results(dirs, args.poll)
            print("\n[%s]" % time.strftime("%H:%M:%S"))
            print_poll(state)
            time.sleep(args.poll)
        return 0

    results_dirs = [RESULTS_ROOT / "summarization", RESULTS_ROOT / "combined"] if args.runner == "azure" else [RESULTS_ROOT / "ollama"]
    proc = run_azure(args) if args.runner == "azure" else run_ollama(args)
    print("Started %s benchmark (PID %s). Polling every %ss." % (args.runner, proc.pid, args.poll))

    last_stdout = []
    poll_count = 0
    try:
        while proc.poll() is None:
            time.sleep(args.poll)
            poll_count += 1
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                line = line.rstrip()
                last_stdout.append(line)
                if args.verbose:
                    print(line)
                # Real failures only (skip progress "Errors: 0 (0.0%)")
                if "Traceback" in line or "Exception:" in line or "Unknown model" in line:
                    print("[monitor] Possible issue:", line[:120])
                elif "Error" in line and "Errors:" not in line and "0.0%" not in line:
                    print("[monitor] Possible issue:", line[:120])

            state = poll_results(results_dirs, args.poll)
            print("\n[%s] poll #%s" % (time.strftime("%H:%M:%S"), poll_count))
            print_poll(state)

        for line in proc.stdout:
            last_stdout.append(line.rstrip())
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("\n[monitor] Interrupted.")
        return 130

    code = proc.returncode
    state = poll_results(results_dirs, args.poll)
    print("\n--- Final state ---")
    print_poll(state)
    if code != 0:
        print("\nProcess exited with code %s. Last 20 lines:" % code)
        for line in last_stdout[-20:]:
            print(line)
    return code


if __name__ == "__main__":
    sys.exit(main())
