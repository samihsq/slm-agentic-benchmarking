#!/usr/bin/env python3
"""
Quick status check for planning benchmark runs. Run every minute to decide if changes are needed.
Outputs one-line summary + optional details. Exit code 0 = both done, 1 = in progress, 2 = error/stuck.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLANNING = ROOT / "results" / "planning"
OLLAMA_BASE = ROOT / "results" / "ollama"
STATUS_FILE = ROOT / "results" / "planning" / ".last_status"

N_AZURE = 10
N_OLLAMA = 9
STUCK_THRESHOLD_SEC = 300  # no progress for 5 min => suggest stuck


def get_ts(p: Path) -> str:
    parts = p.name.split("_")
    if len(parts) >= 2 and parts[-1].isdigit() and parts[-2].isdigit():
        return "_".join(parts[-2:])
    return ""


def main():
    # Azure: latest run completion count
    azure_done = 0
    if PLANNING.exists():
        model_dirs = [d for d in PLANNING.iterdir() if d.is_dir() and not d.name.startswith(".") and d.name != "PLANNING_RUN_SUMMARY.md"]
        if model_dirs:
            latest_ts = max((get_ts(d) for d in model_dirs if get_ts(d)), default="")
            for d in model_dirs:
                if get_ts(d) != latest_ts:
                    continue
                summary = d / "OneShotAgent" / "summary.json"
                if summary.exists():
                    try:
                        j = json.loads(summary.read_text())
                        if j.get("num_tasks"):
                            azure_done += 1
                    except Exception:
                        pass

    # Ollama: latest planning run completion count
    ollama_done = 0
    ollama_run = None
    if OLLAMA_BASE.exists():
        for ts_dir in sorted(OLLAMA_BASE.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if not ts_dir.is_dir():
                continue
            pp = ts_dir / "planning"
            if pp.exists():
                ollama_run = ts_dir.name
                for model_dir in pp.iterdir():
                    if not model_dir.is_dir():
                        continue
                    summary = model_dir / "OllamaAgent" / "summary.json"
                    if summary.exists():
                        try:
                            j = json.loads(summary.read_text())
                            if j.get("num_tasks"):
                                ollama_done += 1
                        except Exception:
                            pass
                break

    # Persist for stuck detection (so agent/cron can run every minute and see if stuck)
    now = time.time()
    exit_code_extra = 1
    try:
        PLANNING.mkdir(parents=True, exist_ok=True)
        prev = None
        if STATUS_FILE.exists():
            try:
                prev = json.loads(STATUS_FILE.read_text())
            except Exception:
                pass
        if prev and prev.get("azure_done") == azure_done and prev.get("ollama_done") == ollama_done:
            if now - prev.get("ts", 0) > STUCK_THRESHOLD_SEC:
                print("STUCK=possible (no progress for >5 min)")
                exit_code_extra = 2
        STATUS_FILE.write_text(json.dumps({
            "ts": now,
            "azure_done": azure_done,
            "ollama_done": ollama_done,
            "ollama_run": ollama_run,
        }))
    except Exception:
        pass

    # One-line summary for agent
    print(f"AZURE {azure_done}/{N_AZURE} | OLLAMA {ollama_done}/{N_OLLAMA} (run {ollama_run or 'none'})")

    if azure_done >= N_AZURE and ollama_done >= N_OLLAMA:
        print("STATUS=done")
        if STATUS_FILE.exists():
            try:
                STATUS_FILE.unlink()
            except Exception:
                pass
        return 0
    if azure_done > 0 or ollama_done > 0:
        print("STATUS=in_progress")
        return exit_code_extra
    print("STATUS=starting")
    return 1


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--wait", action="store_true", help="Loop every 90s until both runs done")
    p.add_argument("--interval", type=int, default=90, help="Seconds between checks when --wait")
    args = p.parse_args()
    if args.wait:
        while True:
            code = main()
            if code == 0:
                sys.exit(0)
            time.sleep(args.interval)
    sys.exit(main())
