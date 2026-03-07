#!/usr/bin/env python3
"""
BIG-bench Lite sweep across all Azure and Ollama models, 4 architectures.

Runs the official 24-task BIG-bench Lite suite (BBL24) at 20 examples per task
for every combination of (backend, model, architecture).

Parallelism model:
  - Azure: all models run concurrently (one thread per model), architectures
    run sequentially within each model's thread.
  - Ollama: models run sequentially smallest → largest (GPU loads one model at
    a time), architectures run sequentially within each model.
  - Azure and Ollama backends run concurrently with each other.
  - --concurrency controls how many examples per model are in-flight at once.

Usage
-----
# Full sweep (Azure + Ollama in parallel):
python scripts/run_bigbench_lite_sweep.py

# Pilot run first (4 examples/task, small model subset, all 4 architectures):
python scripts/run_bigbench_lite_sweep.py --pilot

# Azure only, 3 examples in-flight per model:
python scripts/run_bigbench_lite_sweep.py --backends azure --concurrency 3

# Ollama only (single example at a time per model to avoid GPU contention):
python scripts/run_bigbench_lite_sweep.py --backends ollama --concurrency 1

# Resume — skip (backend, model, architecture) combos already complete:
python scripts/run_bigbench_lite_sweep.py --resume

# Validate task manifest without running:
python scripts/run_bigbench_lite_sweep.py --validate-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv()

from src.agents import (
    OneShotAgent,
    SequentialAgent,
    ConcurrentAgent,
    GroupChatAgent,
    OllamaAgent,
    OllamaSequentialAgent,
    OllamaConcurrentAgent,
    OllamaGroupChatAgent,
)
from src.benchmarks.skills.bigbench.runner import BigBenchRunner
from src.benchmarks.skills.bigbench.task_sets import BBL24_TASK_NAMES
from src.config.azure_llm_config import AVAILABLE_MODELS, OLLAMA_MODELS
from src.evaluation import CostTracker

# ---------------------------------------------------------------------------
# Model whitelists
# ---------------------------------------------------------------------------

AZURE_MODELS: List[str] = [
    "phi-4",
    "ministral-3b",
    "mistral-small",
]

# Ollama models ordered smallest → largest so smaller models complete first.
# GPU can only run one model at a time; this order ensures we get results from
# small models sooner and only load larger models after smaller ones finish.
OLLAMA_MODELS_BY_SIZE: List[str] = [
    "qwen3-0.6b",               # 0.6B
    "gemma3-1b",                # 1B
    "gemma3n-e2b",              # 2B eff
    "phi4-mini-reasoning-ollama", # 3.8B
    "dasd-4b",                  # 4B
    "gemma3-4b",                # 4B
    "gemma3n-e4b",              # 4B eff
    "gpt-oss-20b",              # 20B
]
OLLAMA_MODEL_KEYS: List[str] = OLLAMA_MODELS_BY_SIZE

# Pilot: small fast representatives from each backend
PILOT_AZURE_MODELS: List[str] = ["phi-4", "ministral-3b"]
PILOT_OLLAMA_MODELS: List[str] = ["gemma3-1b", "gpt-oss-20b"]
PILOT_EXAMPLES_PER_TASK: int = 4

# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------

ARCHITECTURES: List[str] = ["one_shot", "sequential", "concurrent", "group_chat"]


def _make_azure_agent(model: str, architecture: str, verbose: bool):
    if architecture == "one_shot":
        return OneShotAgent(model=model, verbose=verbose)
    if architecture == "sequential":
        return SequentialAgent(model=model, verbose=verbose)
    if architecture == "concurrent":
        return ConcurrentAgent(model=model, verbose=verbose)
    if architecture == "group_chat":
        return GroupChatAgent(model=model, verbose=verbose)
    raise ValueError(f"Unknown architecture: {architecture}")


def _make_ollama_agent(model: str, architecture: str, verbose: bool, ollama_base_url: str):
    if architecture == "one_shot":
        return OllamaAgent(model=model, verbose=verbose, ollama_base_url=ollama_base_url)
    if architecture == "sequential":
        return OllamaSequentialAgent(model=model, verbose=verbose, ollama_base_url=ollama_base_url)
    if architecture == "concurrent":
        return OllamaConcurrentAgent(model=model, verbose=verbose, ollama_base_url=ollama_base_url)
    if architecture == "group_chat":
        return OllamaGroupChatAgent(model=model, verbose=verbose, ollama_base_url=ollama_base_url)
    raise ValueError(f"Unknown architecture: {architecture}")


# ---------------------------------------------------------------------------
# Resumability helpers
# ---------------------------------------------------------------------------

def _summary_path(results_root: Path, backend: str, model: str, architecture: str) -> Path:
    return results_root / backend / model / architecture / "summary.json"


def _is_done(results_root: Path, backend: str, model: str, architecture: str) -> bool:
    p = _summary_path(results_root, backend, model, architecture)
    if not p.exists():
        return False
    try:
        with open(p) as f:
            data = json.load(f)
        return bool(data.get("weighted_accuracy") is not None and data.get("num_examples_total", 0) > 0)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_tasks(verbose: bool = True) -> bool:
    """Verify all 24 BBL24 configs are present in tasksource/bigbench."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("ERROR: 'datasets' package not installed. Run: pip install datasets")
        return False

    print(f"Validating {len(BBL24_TASK_NAMES)} BBL24 task configs...")
    missing = []
    for name in BBL24_TASK_NAMES:
        try:
            ds = load_dataset("tasksource/bigbench", name, split="train", trust_remote_code=True)
            if verbose:
                print(f"  OK  {name} ({len(ds)} train examples)")
        except Exception as e:
            missing.append(name)
            print(f"  MISSING  {name}: {e}")
    if missing:
        print(f"\nMissing configs: {missing}")
        return False
    print(f"\nAll {len(BBL24_TASK_NAMES)} configs validated.")
    return True


# ---------------------------------------------------------------------------
# Single (model, architecture) run
# ---------------------------------------------------------------------------

def run_one(
    backend: str,
    model: str,
    architecture: str,
    examples_per_task: int,
    results_root: Path,
    cost_tracker: CostTracker,
    verbose: bool,
    concurrency: int,
    ollama_base_url: str,
    resume: bool,
) -> Dict[str, Any]:
    label = f"{backend}/{model}/{architecture}"

    if resume and _is_done(results_root, backend, model, architecture):
        p = _summary_path(results_root, backend, model, architecture)
        with open(p) as f:
            data = json.load(f)
        w_acc = data.get("weighted_accuracy", 0.0)
        print(f"  [SKIP] {label} already complete ({w_acc * 100:.1f}%)")
        return {"backend": backend, "model": model, "architecture": architecture,
                "weighted_accuracy": w_acc, "skipped": True}

    run_dir = results_root / backend / model / architecture
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        if backend == "azure":
            agent = _make_azure_agent(model, architecture, verbose)
        else:
            agent = _make_ollama_agent(model, architecture, verbose, ollama_base_url)

        runner = BigBenchRunner(
            agent=agent,
            cost_tracker=cost_tracker,
            verbose=verbose,
            concurrency=concurrency,
            run_dir=run_dir,
            suite="bbl24",
            examples_per_task=examples_per_task,
            backend=backend,
            architecture=architecture,
        )
        runner.run(save_results=True)

        p = _summary_path(results_root, backend, model, architecture)
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            return {
                "backend": backend,
                "model": model,
                "architecture": architecture,
                "weighted_accuracy": data.get("weighted_accuracy", 0.0),
                "num_examples": data.get("num_examples_total", 0),
                "total_cost": data.get("total_cost_usd", 0.0),
                "skipped": False,
            }
    except Exception as e:
        print(f"  ERROR {label}: {e}")
        return {
            "backend": backend, "model": model, "architecture": architecture,
            "error": str(e), "weighted_accuracy": None, "skipped": False,
        }

    return {"backend": backend, "model": model, "architecture": architecture,
            "weighted_accuracy": None, "skipped": False}


# ---------------------------------------------------------------------------
# Backend sweep worker
# ---------------------------------------------------------------------------

def _run_model(
    backend: str,
    model: str,
    architectures: List[str],
    examples_per_task: int,
    results_root: Path,
    cost_tracker: CostTracker,
    verbose: bool,
    concurrency: int,
    ollama_base_url: str,
    resume: bool,
) -> List[Dict[str, Any]]:
    """Run all architectures for a single model sequentially."""
    results = []
    for arch in architectures:
        r = run_one(
            backend, model, arch,
            examples_per_task, results_root, cost_tracker,
            verbose, concurrency, ollama_base_url, resume,
        )
        results.append(r)
    return results


def sweep_backend(
    backend: str,
    models: List[str],
    architectures: List[str],
    examples_per_task: int,
    results_root: Path,
    cost_tracker: CostTracker,
    verbose: bool,
    concurrency: int,
    ollama_base_url: str,
    resume: bool,
    parallel_models: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run all architectures for each model in the backend.

    Azure (parallel_models=True): all models run concurrently, one thread per
    model, each progressing through its architectures sequentially.

    Ollama (parallel_models=False): models run one at a time in the order given
    (smallest → largest) so the GPU only loads one model at a time.  Within each
    model, architectures are still sequential.

    In both cases, `concurrency` controls how many examples per model are in-flight.
    """
    results = []
    if parallel_models:
        with ThreadPoolExecutor(max_workers=len(models)) as pool:
            futures = {
                pool.submit(
                    _run_model,
                    backend, model, architectures,
                    examples_per_task, results_root, cost_tracker,
                    verbose, concurrency, ollama_base_url, resume,
                ): model
                for model in models
            }
            for fut in as_completed(futures):
                results.extend(fut.result())
    else:
        for model in models:
            rows = _run_model(
                backend, model, architectures,
                examples_per_task, results_root, cost_tracker,
                verbose, concurrency, ollama_base_url, resume,
            )
            results.extend(rows)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="BIG-bench Lite sweep: Azure + Ollama models, 4 architectures"
    )
    parser.add_argument(
        "--backends", nargs="+", choices=["azure", "ollama"], default=["azure", "ollama"],
        help="Which backends to run (default: both)",
    )
    parser.add_argument("--azure-models", nargs="+", default=None, help="Override Azure model list")
    parser.add_argument("--ollama-models", nargs="+", default=None, help="Override Ollama model list")
    parser.add_argument(
        "--architectures", nargs="+",
        choices=ARCHITECTURES, default=ARCHITECTURES,
        help="Architectures to sweep (default: all 4)",
    )
    parser.add_argument(
        "--examples-per-task", type=int, default=20,
        help="Examples per standard task (undersized tasks use all available, default: 20)",
    )
    parser.add_argument(
        "--pilot", action="store_true",
        help=(
            f"Pilot mode: {PILOT_EXAMPLES_PER_TASK} examples/task, "
            f"{PILOT_AZURE_MODELS} + {PILOT_OLLAMA_MODELS}, all architectures"
        ),
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results/bigbench_lite"),
        help="Root output directory (default: results/bigbench_lite)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="Concurrent examples per model within a single run (default: 1)",
    )
    parser.add_argument("--resume", action="store_true", help="Skip combos with existing complete summaries")
    parser.add_argument("--validate-only", action="store_true", help="Validate task configs and exit")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.validate_only:
        ok = validate_tasks(verbose=True)
        return 0 if ok else 1

    # Pilot overrides
    if args.pilot:
        azure_models = PILOT_AZURE_MODELS
        ollama_models = PILOT_OLLAMA_MODELS
        examples_per_task = PILOT_EXAMPLES_PER_TASK
        results_root = args.results_dir / "pilot"
        print(f"PILOT MODE: {examples_per_task} examples/task | "
              f"Azure: {azure_models} | Ollama: {ollama_models}")
    else:
        azure_models = args.azure_models or AZURE_MODELS
        ollama_models = args.ollama_models or OLLAMA_MODEL_KEYS
        examples_per_task = args.examples_per_task
        results_root = args.results_dir

    results_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    cost_tracker = CostTracker(
        budget_limit=500.0, alert_thresholds=[0.3, 0.6, 0.9], log_file="cost_tracking.json"
    )

    architectures = args.architectures
    backends = args.backends

    # Estimate total combos
    n_azure = len(azure_models) if "azure" in backends else 0
    n_ollama = len(ollama_models) if "ollama" in backends else 0
    total_combos = (n_azure + n_ollama) * len(architectures)
    total_examples = total_combos * examples_per_task * len(BBL24_TASK_NAMES)
    print(f"\nBIG-bench Lite Sweep")
    print(f"  Suite:           BBL24 ({len(BBL24_TASK_NAMES)} tasks)")
    print(f"  Examples/task:   {examples_per_task} (standard); undersized tasks use all available")
    print(f"  Architectures:   {architectures}")
    print(f"  Azure models:    {azure_models if 'azure' in backends else 'skipped'} (concurrent)")
    print(f"  Ollama models:   {ollama_models if 'ollama' in backends else 'skipped'} (sequential, small→large)")
    print(f"  Total combos:    {total_combos}")
    print(f"  Est. examples:   ~{total_examples:,}")
    print(f"  Resume:          {args.resume}")
    print(f"  Results root:    {results_root}\n")

    all_results: List[Dict[str, Any]] = []

    # Launch Azure and Ollama sweeps in parallel (backend-parallel).
    backend_futures: Dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=len(backends)) as backend_pool:
        if "azure" in backends:
            backend_futures["azure"] = backend_pool.submit(
                sweep_backend,
                "azure", azure_models, architectures, examples_per_task,
                results_root, cost_tracker, args.verbose, args.concurrency,
                args.ollama_url, args.resume,
                True,   # parallel_models: all Azure models run concurrently
            )
        if "ollama" in backends:
            backend_futures["ollama"] = backend_pool.submit(
                sweep_backend,
                "ollama", ollama_models, architectures, examples_per_task,
                results_root, cost_tracker, args.verbose, args.concurrency,
                args.ollama_url, args.resume,
                False,  # parallel_models: Ollama models run sequentially (GPU can only load one at a time)
            )

        for backend_name, fut in backend_futures.items():
            try:
                rows = fut.result()
                all_results.extend(rows)
                print(f"\n{backend_name.upper()} backend complete ({len(rows)} combos).")
            except Exception as e:
                print(f"\nERROR in {backend_name} backend: {e}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'backend':<8} {'model':<35} {'architecture':<15} {'wt_acc':>8} {'examples':>10}")
    print("-" * 80)
    for r in sorted(all_results, key=lambda x: (x.get("backend",""), x.get("model",""), x.get("architecture",""))):
        wacc = r.get("weighted_accuracy")
        nexamples = r.get("num_examples", "")
        wacc_str = f"{wacc * 100:.1f}%" if wacc is not None else "ERROR"
        skip_str = " [skip]" if r.get("skipped") else ""
        print(f"{r.get('backend',''):<8} {r.get('model',''):<35} {r.get('architecture',''):<15} "
              f"{wacc_str:>8} {str(nexamples):>10}{skip_str}")
    print("=" * 80)

    summary_path = results_root / f"sweep_summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "suite": "bbl24",
            "examples_per_task": examples_per_task,
            "architectures": architectures,
            "azure_models": azure_models if "azure" in backends else [],
            "ollama_models": ollama_models if "ollama" in backends else [],
            "results": all_results,
        }, f, indent=2)
    print(f"\nSweep summary: {summary_path}")

    if args.pilot:
        _print_pilot_eta(all_results, examples_per_task, azure_models, ollama_models, architectures)

    cost_tracker.print_summary()
    return 0


def _print_pilot_eta(
    pilot_results: List[Dict[str, Any]],
    pilot_n: int,
    pilot_azure: List[str],
    pilot_ollama: List[str],
    architectures: List[str],
) -> None:
    """Extrapolate full-sweep ETA from pilot run timings."""
    print("\n--- PILOT ETA EXTRAPOLATION ---")

    full_azure = AZURE_MODELS
    full_ollama = OLLAMA_MODEL_KEYS
    full_n = 20

    azure_times = []
    ollama_times = []

    for r in pilot_results:
        # Try to read timing from the saved summary
        backend = r.get("backend", "")
        model = r.get("model", "")
        arch = r.get("architecture", "")
        summary_p = Path("results/bigbench_lite/pilot") / backend / model / arch / "summary.json"
        if summary_p.exists():
            try:
                with open(summary_p) as f:
                    d = json.load(f)
                secs = d.get("total_latency_seconds", 0)
                if backend == "azure":
                    azure_times.append(secs)
                else:
                    ollama_times.append(secs)
            except Exception:
                pass

    if azure_times:
        avg_azure_pilot = sum(azure_times) / len(azure_times)
        scale = (len(full_azure) * len(architectures) * full_n) / (len(pilot_azure) * len(architectures) * pilot_n)
        est_azure_hours = avg_azure_pilot * scale / 3600
        print(f"  Azure pilot avg latency: {avg_azure_pilot:.0f}s/combo")
        print(f"  Estimated full Azure sweep: ~{est_azure_hours:.1f} hours")

    if ollama_times:
        avg_ollama_pilot = sum(ollama_times) / len(ollama_times)
        scale = (len(full_ollama) * len(architectures) * full_n) / (len(pilot_ollama) * len(architectures) * pilot_n)
        est_ollama_hours = avg_ollama_pilot * scale / 3600
        print(f"  Ollama pilot avg latency: {avg_ollama_pilot:.0f}s/combo")
        print(f"  Estimated full Ollama sweep: ~{est_ollama_hours:.1f} hours")

    if azure_times and ollama_times:
        print(f"  Backend-parallel ETA: ~{max(est_azure_hours, est_ollama_hours):.1f} hours wall-clock")

    print("-------------------------------\n")


if __name__ == "__main__":
    sys.exit(main())
