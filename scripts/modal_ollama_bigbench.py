#!/usr/bin/env python3
"""
Modal launcher for sharded Ollama BIG-bench Lite sweeps.

This keeps the existing benchmark semantics intact by invoking the current
`scripts/run_bigbench_lite_sweep.py` entrypoint on remote Modal workers.

Examples
--------
# Launch the default 4xL4 full sweep and sync results locally afterwards:
modal run scripts/modal_ollama_bigbench.py

# Warm model caches only:
modal run scripts/modal_ollama_bigbench.py --prepare-only

# Download the latest persisted results volume into the local repo:
modal run scripts/modal_ollama_bigbench.py --sync-only
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Sequence

import modal
from src.config.azure_llm_config import OLLAMA_MODELS


REPO_ROOT = Path(__file__).resolve().parent.parent

APP_NAME = "slm-agentic-benchmarking-ollama-sweep"
PYTHON_VERSION = "3.11"
OLLAMA_VERSION = "0.6.5"
OLLAMA_PORT = 11434
OLLAMA_URL = f"http://127.0.0.1:{OLLAMA_PORT}"

REMOTE_WORKSPACE = "/workspace"
REMOTE_OLLAMA_MODELS = "/vol/ollama-models"
REMOTE_RESULTS = "/vol/results"
REMOTE_HF_CACHE = "/vol/hf-cache"
REMOTE_RESULTS_ROOT = f"{REMOTE_RESULTS}/bigbench_lite"

DEFAULT_ARCHITECTURES = ["one_shot", "sequential", "concurrent", "group_chat"]
DEFAULT_MODEL_SHARDS: Dict[str, List[str]] = {
    "worker-1": ["qwen3-0.6b"],
    "worker-2": ["gemma3-1b"],
    "worker-3": ["gemma3n-e2b"],
    "worker-4": ["phi4-mini-reasoning-ollama"],
    "worker-5": ["dasd-4b"],
    "worker-6": ["gemma3-4b"],
    "worker-7": ["gemma3n-e4b"],
    "worker-8": ["gpt-oss-20b"],
}
DEFAULT_CONCURRENCY_BY_MODEL: Dict[str, int] = {
    "qwen3-0.6b": 2,
    "gemma3-1b": 2,
}
DEFAULT_SYNC_DIR = REPO_ROOT / "results" / "bigbench_lite"
WARM_ETA_HOURS = "4-10"
COLD_ETA_HOURS = "6-14"

OLLAMA_VOLUME = modal.Volume.from_name("slm-ollama-model-cache", create_if_missing=True)
RESULTS_VOLUME = modal.Volume.from_name("slm-bigbench-results", create_if_missing=True)
HF_CACHE_VOLUME = modal.Volume.from_name("slm-hf-cache", create_if_missing=True)

_IGNORE_PATTERNS = [
    ".git",
    ".venv",
    ".cursor",
    ".pytest_cache",
    "__pycache__",
    "results",
    "plots",
]

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("curl", "ca-certificates", "git", "zstd")
    .pip_install(
        "crewai>=0.80,<0.81",
        "litellm>=1.40,<2",
        "datasets>=2.14,<3",
        "python-dotenv>=1,<2",
        "pyyaml>=6,<7",
        "tqdm>=4.66,<5",
        "rich>=13,<14",
        "aiohttp>=3.9,<4",
        "nltk>=3.9,<4",
        "scipy>=1.11,<2",
    )
    .run_commands(
        f"OLLAMA_VERSION={OLLAMA_VERSION} curl -fsSL https://ollama.com/install.sh | sh",
        f"mkdir -p {REMOTE_WORKSPACE} {REMOTE_OLLAMA_MODELS} {REMOTE_RESULTS} {REMOTE_HF_CACHE}",
    )
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": REMOTE_WORKSPACE,
            "OLLAMA_HOST": f"0.0.0.0:{OLLAMA_PORT}",
            "OLLAMA_MODELS": REMOTE_OLLAMA_MODELS,
            "HF_HOME": REMOTE_HF_CACHE,
            "HUGGINGFACE_HUB_CACHE": REMOTE_HF_CACHE,
            "TRANSFORMERS_CACHE": REMOTE_HF_CACHE,
        }
    )
    .add_local_dir(
        str(REPO_ROOT),
        remote_path=REMOTE_WORKSPACE,
        ignore=_IGNORE_PATTERNS,
    )
)

app = modal.App(APP_NAME, image=image)


def _parse_csv(value: str | None, default: Sequence[str]) -> List[str]:
    if not value:
        return list(default)
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_shards(selected: str | None) -> Dict[str, List[str]]:
    if not selected:
        return {name: list(models) for name, models in DEFAULT_MODEL_SHARDS.items()}

    requested = [item.strip() for item in selected.split(",") if item.strip()]
    unknown = [item for item in requested if item not in DEFAULT_MODEL_SHARDS]
    if unknown:
        raise ValueError(f"Unknown shard(s): {', '.join(unknown)}")
    return {name: list(DEFAULT_MODEL_SHARDS[name]) for name in requested}


def _wait_for_ollama(url: str, timeout_seconds: int = 120) -> None:
    deadline = time.time() + timeout_seconds
    last_error = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/api/tags", timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_error = str(exc)
            time.sleep(2)
    raise RuntimeError(f"Ollama did not become ready within {timeout_seconds}s: {last_error}")


def _start_ollama_server() -> subprocess.Popen[Any]:
    process = subprocess.Popen(["ollama", "serve"])
    _wait_for_ollama(OLLAMA_URL)
    return process


def _stop_process(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _existing_models() -> set[str]:
    proc = subprocess.run(
        ["ollama", "list"],
        check=True,
        capture_output=True,
        text=True,
    )
    return set(proc.stdout.split())


def _resolve_ollama_model_name(model: str) -> str:
    return str(OLLAMA_MODELS.get(model, {}).get("model", model))


def _pull_missing_models(models: Sequence[str]) -> List[str]:
    existing = _existing_models()
    pulled: List[str] = []
    for model in models:
        model_name = _resolve_ollama_model_name(model)
        tag = model_name if ":" in model_name else f"{model_name}:latest"
        if tag in existing:
            continue
        subprocess.run(["ollama", "pull", model_name], check=True)
        pulled.append(model)
    return pulled


def _expected_summary_paths(model: str, architectures: Sequence[str]) -> List[PurePosixPath]:
    return [
        PurePosixPath("bigbench_lite") / "ollama" / model / architecture / "summary.json"
        for architecture in architectures
    ]


def _run_model_sweep(
    model: str,
    architectures: Sequence[str],
    examples_per_task: int,
    concurrency: int,
    resume: bool,
    start_delay_seconds: int,
) -> Dict[str, Any]:
    if start_delay_seconds > 0:
        time.sleep(start_delay_seconds)

    cmd = [
        "python",
        "scripts/run_bigbench_lite_sweep.py",
        "--backends",
        "ollama",
        "--ollama-models",
        model,
        "--architectures",
        *architectures,
        "--results-dir",
        REMOTE_RESULTS_ROOT,
        "--ollama-url",
        OLLAMA_URL,
        "--concurrency",
        str(concurrency),
        "--examples-per-task",
        str(examples_per_task),
    ]
    if resume:
        cmd.append("--resume")

    started_at = time.time()
    subprocess.run(cmd, cwd=REMOTE_WORKSPACE, check=True)
    elapsed = time.time() - started_at

    completed = []
    for path in _expected_summary_paths(model, architectures):
        full_path = PurePosixPath(REMOTE_RESULTS) / path
        if Path(full_path.as_posix()).exists():
            completed.append(path.parts[-2])

    return {
        "model": model,
        "architectures_requested": list(architectures),
        "architectures_completed": completed,
        "elapsed_seconds": elapsed,
        "results_root": REMOTE_RESULTS_ROOT,
    }


def _find_incomplete_combos(architectures: Sequence[str]) -> List[tuple[str, List[str]]]:
    """Check the results volume for (model, [missing_architectures]) pairs."""
    all_models = sorted(
        {m for shard in DEFAULT_MODEL_SHARDS.values() for m in shard}
    )
    incomplete: List[tuple[str, List[str]]] = []
    for model in all_models:
        missing = []
        for arch in architectures:
            summary = Path(f"{REMOTE_RESULTS_ROOT}/ollama/{model}/{arch}/summary.json")
            if not summary.exists():
                missing.append(arch)
        if missing:
            incomplete.append((model, missing))
    return incomplete


@app.function(
    gpu="L4",
    timeout=10 * 60 * 60,
    volumes={
        REMOTE_OLLAMA_MODELS: OLLAMA_VOLUME,
        REMOTE_RESULTS: RESULTS_VOLUME,
        REMOTE_HF_CACHE: HF_CACHE_VOLUME,
    },
)
def run_model_shard(
    shard_name: str,
    models: str,
    architectures: str = ",".join(DEFAULT_ARCHITECTURES),
    examples_per_task: int = 20,
    resume: bool = True,
    prepare_only: bool = False,
) -> Dict[str, Any]:
    """
    Run a set of Ollama models sequentially on a single L4-backed worker.

    The worker keeps one Ollama server alive, pre-pulls assigned models into a
    persistent volume, then invokes the existing sweep script one model at a time.
    """
    model_list = _parse_csv(models, [])
    architecture_list = _parse_csv(architectures, DEFAULT_ARCHITECTURES)

    RESULTS_VOLUME.reload()
    HF_CACHE_VOLUME.reload()
    OLLAMA_VOLUME.reload()

    ollama_process: subprocess.Popen[Any] | None = None
    try:
        ollama_process = _start_ollama_server()
        pulled = _pull_missing_models(model_list)
        if pulled:
            OLLAMA_VOLUME.commit()

        if prepare_only:
            return {
                "shard": shard_name,
                "prepared_models": model_list,
                "newly_pulled_models": pulled,
                "ran_benchmarks": False,
            }

        model_runs = []
        for index, model in enumerate(model_list):
            arch_results = []
            for arch in architecture_list:
                arch_results.append(
                    _run_model_sweep(
                        model=model,
                        architectures=[arch],
                        examples_per_task=examples_per_task,
                        concurrency=DEFAULT_CONCURRENCY_BY_MODEL.get(model, 1),
                        resume=resume,
                        start_delay_seconds=index if arch == architecture_list[0] else 0,
                    )
                )
                RESULTS_VOLUME.commit()  # commit after each architecture for live visibility
            model_runs.append(arch_results)

        # Work-stealing disabled to control GPU budget.
        stolen_runs = []

        return {
            "shard": shard_name,
            "models": model_list,
            "newly_pulled_models": pulled,
            "model_runs": model_runs,
            "stolen_runs": stolen_runs,
            "ran_benchmarks": True,
        }
    finally:
        RESULTS_VOLUME.commit()
        HF_CACHE_VOLUME.commit()
        _stop_process(ollama_process)


def _download_volume_tree(remote_prefix: str, local_root: Path) -> int:
    RESULTS_VOLUME.reload()
    local_root.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    remote_prefix_path = PurePosixPath(remote_prefix)
    for entry in RESULTS_VOLUME.iterdir(remote_prefix, recursive=True):
        if entry.is_dir:
            continue

        remote_path = PurePosixPath(entry.path)
        relative_path = remote_path.relative_to(remote_prefix_path)
        target_path = local_root / Path(*relative_path.parts)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        with open(target_path, "wb") as handle:
            for chunk in RESULTS_VOLUME.read_file(entry.path):
                handle.write(chunk)
        downloaded += 1

    return downloaded


def _validate_standard_layout(local_root: Path, architectures: Sequence[str]) -> Dict[str, List[str]]:
    missing: Dict[str, List[str]] = {}
    for shard_models in DEFAULT_MODEL_SHARDS.values():
        for model in shard_models:
            absent_arches = []
            for architecture in architectures:
                summary_path = local_root / "ollama" / model / architecture / "summary.json"
                if not summary_path.exists():
                    absent_arches.append(architecture)
            if absent_arches:
                missing[model] = absent_arches
    return missing


def _print_eta_banner(shards: Dict[str, List[str]]) -> None:
    print("Modal Ollama BIG-bench sweep")
    print(f"  Shards:            {len(shards)} x L4 workers")
    print(f"  Warm-cache ETA:    ~{WARM_ETA_HOURS} hours")
    print(f"  Cold-cache ETA:    ~{COLD_ETA_HOURS} hours")
    print()


@app.local_entrypoint()
def main(
    architectures: str = ",".join(DEFAULT_ARCHITECTURES),
    examples_per_task: int = 20,
    shards: str = "",
    prepare_only: bool = False,
    sync_only: bool = False,
    no_sync: bool = False,
    resume: bool = True,
    sync_dir: str = str(DEFAULT_SYNC_DIR),
) -> None:
    """
    Launch the default 4xL4 sweep, or sync/prefetch helper workflows.

    Parameters are passed as CLI flags when invoking `modal run`.
    """
    selected_shards = _parse_shards(shards or None)
    selected_architectures = _parse_csv(architectures, DEFAULT_ARCHITECTURES)
    local_sync_dir = Path(sync_dir)

    if sync_only:
        downloaded = _download_volume_tree("bigbench_lite", local_sync_dir)
        missing = _validate_standard_layout(local_sync_dir, selected_architectures)
        print(f"Downloaded {downloaded} files into {local_sync_dir}")
        if missing:
            print("Missing summaries after sync:")
            print(json.dumps(missing, indent=2))
        else:
            print("All expected summary paths are present in the synced tree.")
        return

    _print_eta_banner(selected_shards)

    calls = []
    for shard_name, models in selected_shards.items():
        print(f"Launching {shard_name}: {models}")
        calls.append(
            run_model_shard.spawn(
                shard_name=shard_name,
                models=",".join(models),
                architectures=",".join(selected_architectures),
                examples_per_task=examples_per_task,
                resume=resume,
                prepare_only=prepare_only,
            )
        )

    results = [call.get() for call in calls]
    print(json.dumps(results, indent=2))

    if not no_sync:
        downloaded = _download_volume_tree("bigbench_lite", local_sync_dir)
        print(f"Downloaded {downloaded} files into {local_sync_dir}")
        missing = _validate_standard_layout(local_sync_dir, selected_architectures)
        if missing:
            print("Missing summaries after sync:")
            print(json.dumps(missing, indent=2))
        else:
            print("All expected summary paths are present in the synced tree.")
