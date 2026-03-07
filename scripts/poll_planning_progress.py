#!/usr/bin/env python3
"""
Poll planning benchmark progress (Azure + Ollama running concurrently).
Keeps polling until both runs are done, then exits.

Usage:
  poetry run python scripts/poll_planning_progress.py
  poetry run python scripts/poll_planning_progress.py --interval 60
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# For ETA: assume ~30s per task for Azure, ~40s per task for Ollama (varies by model)
AVG_SEC_PER_TASK_AZURE = 30
AVG_SEC_PER_TASK_OLLAMA = 40
POLL_INTERVAL_SEC = 30


def find_latest_planning_dirs():
    """Return (azure_run_dirs, ollama_planning_dir)."""
    planning = ROOT / "results" / "planning"
    ollama_base = ROOT / "results" / "ollama"
    azure_dirs = []
    if planning.exists():
        # Azure: dirs named {model}_{timestamp}
        for d in planning.iterdir():
            if d.is_dir() and not d.name.startswith(".") and d.name != "PLANNING_RUN_SUMMARY.md":
                if (d / "OneShotAgent").exists() or any(d.iterdir()):
                    azure_dirs.append(d)
        azure_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    ollama_planning = None
    if ollama_base.exists():
        timestamps = sorted([d for d in ollama_base.iterdir() if d.is_dir() and d.name.isdigit() or "_" in d.name],
                            key=lambda x: x.stat().st_mtime, reverse=True)
        for ts in timestamps:
            pp = ts / "planning"
            if pp.exists():
                ollama_planning = pp
                break
    return azure_dirs, ollama_planning


def count_azure_models():
    """Expected Azure serverless models (match run_benchmark --all-models)."""
    return 10  # phi-4, phi-4-mini, phi-4-mini-reasoning, mistral-nemo, ministral-3b, mistral-small, mistral-large-3, llama-3.2-11b-vision, gpt-4o, llama-3.3-70b


def count_ollama_models():
    """Expected Ollama models from config."""
    try:
        sys.path.insert(0, str(ROOT))
        from src.config.azure_llm_config import OLLAMA_MODELS
        return len(OLLAMA_MODELS)
    except Exception:
        return 9


def load_summary(agent_dir: Path) -> dict | None:
    path = agent_dir / "summary.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def sample_traces(agent_dir: Path, n: int = 2) -> list[dict]:
    """Load up to n trace.json from agent_dir (e.g. .../OneShotAgent/)."""
    traces = []
    if not agent_dir.exists():
        return traces
    for task_dir in agent_dir.iterdir():
        if not task_dir.is_dir():
            continue
        trace_file = task_dir / "trace.json"
        if trace_file.exists():
            try:
                traces.append(json.loads(trace_file.read_text()))
            except Exception:
                pass
        if len(traces) >= n:
            break
    return traces


def run_one_poll(planning_dir: Path, ollama_planning: Path | None, n_azure_expected: int, n_ollama_expected: int) -> tuple[bool, bool]:
    """Print progress once. Returns (azure_done, ollama_done)."""
    azure_done = False
    ollama_done = False

    print("=" * 70)
    print(f"PLANNING BENCHMARK PROGRESS — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Azure
    print("\n--- Azure (run_benchmark --all-models --benchmark planning) ---")
    if not planning_dir.exists():
        print("  No results/planning yet.")
    else:
        # Only count latest run (most recent timestamp across model_YYYYMMDD_HHMMSS dirs)
        model_dirs = [d for d in planning_dir.iterdir() if d.is_dir() and not d.name.startswith(".") and d.name != "PLANNING_RUN_SUMMARY.md"]
        if not model_dirs:
            completed_azure = 0
            azure_summaries = []
        else:
            # Get latest timestamp (last 2 parts of name: YYYYMMDD_HHMMSS)
            def get_ts(p: Path) -> str:
                parts = p.name.split("_")
                if len(parts) >= 2 and parts[-1].isdigit() and parts[-2].isdigit():
                    return "_".join(parts[-2:])
                return ""
            latest_ts = max((get_ts(d) for d in model_dirs), key=lambda x: (x, 0)) if model_dirs else ""
            latest_dirs = [d for d in model_dirs if get_ts(d) == latest_ts] if latest_ts else model_dirs
            completed_azure = 0
            azure_summaries = []
            for d in latest_dirs:
                agent_dir = d / "OneShotAgent"
                if agent_dir.exists():
                    s = load_summary(agent_dir)
                    if s and s.get("num_tasks"):
                        completed_azure += 1
                        model_name = d.name.rsplit("_", 2)[0] if "_" in d.name else d.name
                        azure_summaries.append((model_name, s))
        print(f"  Completed: {completed_azure} / {n_azure_expected} models (latest run)")
        if completed_azure < n_azure_expected and completed_azure > 0:
            tasks_per_model = azure_summaries[0][1].get("num_tasks", 5) if azure_summaries else 5
            remaining = (n_azure_expected - completed_azure) * tasks_per_model
            eta_sec = remaining * AVG_SEC_PER_TASK_AZURE
            print(f"  ETA (rough): ~{eta_sec // 60:.0f} min remaining for Azure")
        if azure_summaries:
            for model, s in sorted(azure_summaries, key=lambda x: -x[1].get("mean_composite_score", 0))[:5]:
                sr = s.get("success_rate") or 0
                sc = s.get("mean_composite_score", 0)
                n = s.get("num_tasks", 0)
                print(f"    {model}: success_rate={sr:.2f} mean_score={sc:.3f} n={n}")
        if completed_azure >= n_azure_expected:
            azure_done = True
            summary_file = planning_dir / "all_models_summary.json"
            if summary_file.exists():
                print("  [Azure run finished - all_models_summary.json written]")
            else:
                print("  [Azure run finished]")

    # Ollama
    print("\n--- Ollama (run_ollama_benchmarks --benchmarks planning) ---")
    if not ollama_planning:
        print("  No results/ollama/*/planning yet.")
    else:
        run_name = ollama_planning.parent.name
        model_dirs = [d for d in ollama_planning.iterdir() if d.is_dir()]
        completed_ollama = 0
        ollama_summaries = []
        for d in model_dirs:
            agent_dir = d / "OllamaAgent"
            if agent_dir.exists():
                s = load_summary(agent_dir)
                if s and s.get("num_tasks"):
                    completed_ollama += 1
                    ollama_summaries.append((d.name, s))
        print(f"  Run: {run_name}")
        print(f"  Completed: {completed_ollama} / {n_ollama_expected} models")
        if completed_ollama < n_ollama_expected and completed_ollama > 0:
            tasks_per_model = ollama_summaries[0][1].get("num_tasks", 5) if ollama_summaries else 5
            remaining = (n_ollama_expected - completed_ollama) * tasks_per_model
            eta_sec = remaining * AVG_SEC_PER_TASK_OLLAMA
            print(f"  ETA (rough): ~{eta_sec // 60:.0f} min remaining for Ollama")
        if completed_ollama >= n_ollama_expected:
            ollama_done = True
            print("  [Ollama run finished]")
        if ollama_summaries:
            for model, s in sorted(ollama_summaries, key=lambda x: -x[1].get("mean_composite_score", 0))[:5]:
                sr = s.get("success_rate") or 0
                sc = s.get("mean_composite_score", 0)
                n = s.get("num_tasks", 0)
                print(f"    {model}: success_rate={sr:.2f} mean_score={sc:.3f} n={n}")

    # Spot-check: sample traces
    print("\n--- Spot-check (sample traces) ---")
    if planning_dir.exists():
        for d in sorted(planning_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if not d.is_dir() or d.name.startswith("."):
                continue
            agent_dir = d / "OneShotAgent"
            if agent_dir.exists():
                traces = sample_traces(agent_dir, n=1)
                if traces:
                    t = traces[0]
                    out = (t.get("final_output") or t.get("predicted") or "")[:200]
                    match = t.get("match", False)
                    print(f"  Azure sample ({d.name}): match={match} output_len={len(out)}")
                    if out:
                        print(f"    \"{out[:120].replace(chr(10), ' ')}...\"")
                    break
    if ollama_planning:
        for d in ollama_planning.iterdir():
            if not d.is_dir():
                continue
            agent_dir = d / "OllamaAgent"
            if agent_dir.exists():
                traces = sample_traces(agent_dir, n=1)
                if traces:
                    t = traces[0]
                    out = (t.get("final_output") or t.get("predicted") or "")[:200]
                    match = t.get("match", False)
                    print(f"  Ollama sample ({d.name}): match={match} output_len={len(out)}")
                    if out:
                        print(f"    \"{out[:120].replace(chr(10), ' ')}...\"")
                    break

    print("\n" + "=" * 70)
    return azure_done, ollama_done


def main():
    parser = argparse.ArgumentParser(description="Poll planning benchmark until both Azure and Ollama runs are done")
    parser.add_argument("--interval", type=int, default=POLL_INTERVAL_SEC, help=f"Seconds between polls (default: {POLL_INTERVAL_SEC})")
    args = parser.parse_args()

    n_azure_expected = count_azure_models()
    n_ollama_expected = count_ollama_models()

    while True:
        _, ollama_planning = find_latest_planning_dirs()
        planning_dir = ROOT / "results" / "planning"
        azure_done, ollama_done = run_one_poll(planning_dir, ollama_planning, n_azure_expected, n_ollama_expected)

        if azure_done and ollama_done:
            print("\n*** Both Azure and Ollama planning runs are done. ***\n")
            return 0

        print(f"\nNext poll in {args.interval}s...\n")
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
