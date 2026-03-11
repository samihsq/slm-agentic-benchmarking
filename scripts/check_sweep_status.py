#!/usr/bin/env python3
"""
BIG-bench sweep status checker with ETA estimation.

Queries the remote Modal volume for progress on every (model, architecture)
combo, computes per-combo rates from a persisted state file, and prints
ETAs for all in-progress and not-started work.

State is stored in /tmp/bb_sweep_state.json between runs.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

MODAL = str(Path(__file__).parent.parent / ".venv/bin/modal")
STATE_FILE = Path("/tmp/bb_sweep_state.json")
TOTAL_PER_COMBO = 308

MODELS = [
    "qwen3-0.6b",
    "gemma3-1b",
    "gemma3n-e2b",
    "phi4-mini-reasoning-ollama",
    "dasd-4b",
    "gemma3-4b",
    "gemma3n-e4b",
    "gpt-oss-20b",
]
ARCHES = ["one_shot", "sequential", "concurrent", "group_chat"]


def modal(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([MODAL, *args], capture_output=True, text=True)


def combo_lines(model: str, arch: str) -> tuple[str, int]:
    """Returns (status, line_count). status: 'done' | 'active' | 'not_started'."""
    base = f"bigbench_lite/ollama/{model}/{arch}"
    if modal("volume", "ls", "slm-bigbench-results", f"{base}/summary.json").returncode == 0:
        return "done", TOTAL_PER_COMBO
    if modal("volume", "ls", "slm-bigbench-results", f"{base}/results.jsonl").returncode == 0:
        r = subprocess.run(
            [MODAL, "volume", "get", "slm-bigbench-results", f"{base}/results.jsonl",
             "/tmp/_bb_check.jsonl", "--force"],
            capture_output=True,
        )
        try:
            with open("/tmp/_bb_check.jsonl") as f:
                lines = sum(1 for ln in f if ln.strip())
            return "active", lines
        except Exception:
            return "active", 0
    return "not_started", 0


def fmt_eta(seconds: float) -> str:
    if seconds <= 0:
        return "done"
    td = timedelta(seconds=int(seconds))
    h, rem = divmod(td.seconds + td.days * 86400, 3600)
    m = rem // 60
    if h >= 48:
        return f"~{td.days}d {h % 24}h"
    if h >= 1:
        return f"~{h}h {m:02d}m"
    return f"~{m}m"


def main() -> None:
    now = time.time()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Load previous state
    prev: dict[str, dict] = {}
    if STATE_FILE.exists():
        try:
            prev = json.loads(STATE_FILE.read_text())
        except Exception:
            prev = {}

    # App health check
    app_r = modal("app", "list")
    active_apps = [ln for ln in app_r.stdout.splitlines() if "ephemeral" in ln]
    print(f"=== {now_str} ===\n")
    if active_apps:
        for ln in active_apps:
            parts = [p.strip() for p in ln.split("│") if p.strip()]
            app_id = parts[0] if parts else "?"
            print(f"Active app: {app_id}  ({len(active_apps)} ephemeral app(s) running)")
    else:
        # Workers spawned with .spawn() continue running even after the local
        # entrypoint exits and the app is marked stopped. Check if data is
        # still flowing before raising the alarm.
        print("No active Modal app (entrypoint exited — workers may still be running independently)")

    # Per-combo progress
    new_state: dict[str, dict] = {}
    done_count = 0
    rows = []

    for model in MODELS:
        for arch in ARCHES:
            key = f"{model}/{arch}"
            status, lines = combo_lines(model, arch)

            if status == "done":
                done_count += 1
                new_state[key] = {"lines": TOTAL_PER_COMBO, "ts": now, "status": "done"}
                rows.append((model, arch, "DONE", TOTAL_PER_COMBO, None, None))
                continue

            new_state[key] = {"lines": lines, "ts": now, "status": status}

            # Compute rate from previous state
            rate_per_sec: float | None = None
            eta_str = "—"
            if status == "active" and key in prev and prev[key]["status"] == "active":
                prev_lines = prev[key]["lines"]
                prev_ts = prev[key]["ts"]
                elapsed = now - prev_ts
                delta = lines - prev_lines
                if elapsed > 0 and delta > 0:
                    rate_per_sec = delta / elapsed
                    remaining = TOTAL_PER_COMBO - lines
                    eta_str = fmt_eta(remaining / rate_per_sec)
            elif status == "active" and lines > 0:
                eta_str = "measuring…"
            elif status == "not_started":
                eta_str = "queued"

            rows.append((model, arch, status.upper().replace("_", " "), lines, rate_per_sec, eta_str))

    # Print table
    print(f"{'Model':<30} {'Arch':<15} {'Status':<12} {'Done':>9}  {'Rate':>10}  {'ETA':<14}")
    print("─" * 95)
    for model, arch, status, lines, rate, eta in rows:
        if status == "DONE":
            print(f"{model:<30} {arch:<15} {'✓ DONE':<12} {f'{lines}/{TOTAL_PER_COMBO}':>9}  {'':>10}  {'':14}")
        elif status == "ACTIVE":
            rate_str = f"{rate * 3600:.1f}/hr" if rate else "—"
            print(f"{model:<30} {arch:<15} {'⟳ ACTIVE':<12} {f'{lines}/{TOTAL_PER_COMBO}':>9}  {rate_str:>10}  {eta:<14}")
        else:
            print(f"{model:<30} {arch:<15} {'  queued':<12} {'':>9}  {'':>10}  {'queued':<14}")

    print("─" * 95)
    in_progress = sum(1 for *_, s, _, _, _ in rows if s == "ACTIVE")
    print(f"\nProgress: {done_count}/32 combos done  |  {in_progress} active  |  {32 - done_count - in_progress} queued\n")

    # Save new state
    STATE_FILE.write_text(json.dumps(new_state))


if __name__ == "__main__":
    main()
