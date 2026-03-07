"""
Rescore existing summarization runs with BERTScore (distilbert-base-uncased).

Reads trace.json files from results/summarization/ and results/ollama/*/summarization/,
extracts (predicted, correct) pairs, scores them with BERTScore F1, and writes:
  - Per-run summary: <run_dir>/bertscore_summary.json
  - Aggregate table:  results/bertscore_comparison.json

Usage:
    poetry run python scripts/rescore_bertscore.py [--model-type distilbert-base-uncased]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional
import math

RESULTS_ROOT = Path(__file__).parent.parent / "results"
SUCCESS_THRESHOLD = 0.60   # BERTScore F1 ≥ 0.60 considered a good summary
BATCH_SIZE = 64
# Max chars per prediction/reference to avoid tokenizer overflow with large models
MAX_PAIR_CHARS = 2048
BOOTSTRAP_ITERS = 2000
RNG_SEED = 42


def wilson(n: int, p: float):
    z = 1.96
    if n == 0:
        return (0.0, 0.0)
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(max(0, c - m) * 100, 1), round(min(1, c + m) * 100, 1))


def bootstrap_mean_ci(values: list[float], n_iter: int = BOOTSTRAP_ITERS, seed: int = RNG_SEED):
    """Percentile bootstrap 95% CI on the mean of a list of floats."""
    import random
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    means = []
    for _ in range(n_iter):
        sample = [rng.choice(values) for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_iter)]
    hi = means[int(0.975 * n_iter)]
    return (round(lo, 4), round(hi, 4))


def collect_traces(run_agent_dir: Path) -> list[dict]:
    """Walk a <run>/<Agent>/ dir and collect all trace.json files."""
    traces = []
    for item in sorted(run_agent_dir.iterdir()):
        if item.is_dir():
            trace_file = item / "trace.json"
            if trace_file.exists():
                try:
                    t = json.loads(trace_file.read_text())
                    pred = (t.get("predicted") or t.get("final_output") or "").strip()[:MAX_PAIR_CHARS]
                    ref = (t.get("correct") or "").strip()[:MAX_PAIR_CHARS]
                    if pred and ref:
                        traces.append({"task_id": item.name, "predicted": pred, "reference": ref})
                except Exception:
                    pass
    return traces


def find_runs() -> list[tuple[str, str, Path]]:
    """Return list of (model_name, provider, agent_dir) for all summarization runs."""
    runs = []

    # Azure summarization
    az_root = RESULTS_ROOT / "summarization"
    if az_root.exists():
        for run_dir in sorted(az_root.iterdir()):
            agent_dir = run_dir / "OneShotAgent"
            summary = agent_dir / "summary.json"
            if summary.exists():
                s = json.loads(summary.read_text())
                runs.append((s.get("model", run_dir.name), "Azure", agent_dir))

    # Ollama summarization (any timestamped ollama run)
    ollama_root = RESULTS_ROOT / "ollama"
    if ollama_root.exists():
        for ts_dir in sorted(ollama_root.iterdir()):
            sum_dir = ts_dir / "summarization"
            if not sum_dir.exists():
                continue
            for model_dir in sorted(sum_dir.iterdir()):
                agent_dir = model_dir / "OllamaAgent"
                summary = agent_dir / "summary.json"
                if summary.exists():
                    s = json.loads(summary.read_text())
                    runs.append((s.get("model", model_dir.name), "Ollama", agent_dir))

    return runs


def score_batch(metric, preds: list[str], refs: list[str], model_type: str) -> list[float]:
    out = metric.compute(
        predictions=preds,
        references=refs,
        lang="en",
        model_type=model_type,
        verbose=False,
    )
    f1s = out.get("f1", [])
    return [float(x) for x in f1s]


def _patch_tokenizer_max_length():
    """Cap tokenizer model_max_length to avoid OverflowError with DeBERTa and similar models."""
    try:
        from transformers import tokenization_utils_base
        base = tokenization_utils_base.PreTrainedTokenizerBase
        if getattr(base, "_bertscore_max_length_patched", False):
            return
        _orig_getattr = object.__getattribute__

        def _patched_getattr(self, name):
            val = _orig_getattr(self, name)
            if name == "model_max_length" and isinstance(val, int) and val > 65536:
                return 512
            return val

        base.__getattribute__ = _patched_getattr  # type: ignore[assignment]
        base._bertscore_max_length_patched = True  # type: ignore[attr-defined]
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        default="distilbert-base-uncased",
        help="HuggingFace model type for BERTScore (default: distilbert-base-uncased)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=SUCCESS_THRESHOLD,
        help=f"BERTScore F1 success threshold (default: {SUCCESS_THRESHOLD})",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="If set, write to bertscore_{suffix}_summary.json and bertscore_{suffix}_comparison.json",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for BERTScore (default: {BATCH_SIZE}; use smaller e.g. 16 for large models)",
    )
    args = parser.parse_args()
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""

    _patch_tokenizer_max_length()
    import evaluate  # type: ignore
    print(f"Loading BERTScore metric (model: {args.model_type})...")
    metric = evaluate.load("bertscore")

    runs = find_runs()
    # Deduplicate: for a model appearing in multiple timestamps, keep highest n;
    # break ties by preferring the most recent run (dir name sorts later = newer timestamp).
    seen: dict[tuple[str, str], tuple[int, str, Path]] = {}
    for model, provider, agent_dir in runs:
        key = (model, provider)
        summary = agent_dir / "summary.json"
        n = json.loads(summary.read_text()).get("num_tasks", 0)
        run_ts = agent_dir.parent.name  # e.g. gpt-4o_20260225_042134
        prev = seen.get(key)
        if prev is None or n > prev[0] or (n == prev[0] and run_ts > prev[1]):
            seen[key] = (n, run_ts, agent_dir)

    deduped = [(model, provider, agent_dir) for (model, provider), (_, _ts, agent_dir) in seen.items()]
    deduped.sort(key=lambda x: x[0])

    print(f"Found {len(deduped)} model runs to rescore.\n")

    all_results = []

    for model, provider, agent_dir in deduped:
        traces = collect_traces(agent_dir)
        n = len(traces)
        if n == 0:
            print(f"  {model}: no traces found, skipping.")
            continue

        print(f"  {model} ({provider}): {n} pairs...", end=" ", flush=True)

        # Score in batches
        all_f1s: list[float] = []
        batch_size = args.batch_size
        for i in range(0, n, batch_size):
            batch = traces[i : i + batch_size]
            preds = [t["predicted"] for t in batch]
            refs = [t["reference"] for t in batch]
            f1s = score_batch(metric, preds, refs, args.model_type)
            all_f1s.extend(f1s)

        avg_f1 = sum(all_f1s) / len(all_f1s)
        bs_lo, bs_hi = bootstrap_mean_ci(all_f1s)

        print(f"avg F1={avg_f1:.4f}  95% CI=[{bs_lo},{bs_hi}]")

        run_summary = {
            "model": model,
            "provider": provider,
            "num_tasks": n,
            "bertscore_model_type": args.model_type,
            "avg_bertscore_f1": round(avg_f1, 6),
            "bs_ci_lo": bs_lo,
            "bs_ci_hi": bs_hi,
        }
        all_results.append(run_summary)

        # Write per-run summary alongside existing summary.json
        out_path = agent_dir / f"bertscore{suffix}_summary.json"
        out_path.write_text(json.dumps(run_summary, indent=2))

    # Write aggregate
    agg_path = RESULTS_ROOT / f"bertscore{suffix}_comparison.json"
    agg_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nAggregate written to {agg_path}")


if __name__ == "__main__":
    main()
