#!/usr/bin/env python3
"""
Join BIG-bench Lite sweep results with existing skill-benchmark scores.

Produces a joined CSV keyed by (backend, model, architecture) with:
  - weighted_accuracy: equal-task-weighted BBL24 score
  - planning, criticality, recall, summarization, instruction_following:
    mean of benchmark scores within each skill bucket

The output is consumed by scripts/plot_skill_vs_bigbench.py.

Usage
-----
python scripts/join_bigbench_skill_scores.py
python scripts/join_bigbench_skill_scores.py --bigbench-dir results/bigbench_lite --output joined_scores.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Skill bucket definition
# ---------------------------------------------------------------------------

SKILL_BUCKETS: Dict[str, List[str]] = {
    "planning":              ["planning", "plan_bench"],
    "criticality":           ["criticality", "criticality_v2"],
    "recall":                ["recall", "matrix_recall"],
    "summarization":         ["summarization"],
    "instruction_following": ["instruction_following", "word_instruction_following"],
}

SKILLS = list(SKILL_BUCKETS.keys())

BENCHMARK_TO_SKILL: Dict[str, str] = {
    bench: skill
    for skill, benchmarks in SKILL_BUCKETS.items()
    for bench in benchmarks
}


# ---------------------------------------------------------------------------
# Model key normalization (reuse logic from get_best_models_per_skill)
# ---------------------------------------------------------------------------

def _build_model_key_lookup() -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    try:
        from src.config.azure_llm_config import AVAILABLE_MODELS, OLLAMA_MODELS
        for key, cfg in {**AVAILABLE_MODELS, **OLLAMA_MODELS}.items():
            model_str = cfg.get("model", "")
            if model_str:
                lookup[model_str.lower()] = key
                bare = model_str.lower().split("/")[-1]
                if bare not in lookup:
                    lookup[bare] = key
        for key in {**AVAILABLE_MODELS, **OLLAMA_MODELS}:
            lookup[key.lower()] = key
    except ImportError:
        pass
    return lookup


_MODEL_KEY_LOOKUP: Optional[Dict[str, str]] = None


def normalize_model_key(raw: str) -> str:
    global _MODEL_KEY_LOOKUP
    if _MODEL_KEY_LOOKUP is None:
        _MODEL_KEY_LOOKUP = _build_model_key_lookup()
    key = _MODEL_KEY_LOOKUP.get(raw.lower())
    if key:
        return key
    stripped = raw.lower().split(":")[0].replace("/", "-").replace("_", "-")
    return _MODEL_KEY_LOOKUP.get(stripped, raw.lower())


# ---------------------------------------------------------------------------
# Load BIG-bench Lite results
# ---------------------------------------------------------------------------

def _benchmark_from_path(path: Path) -> Optional[str]:
    parts = path.parts
    for bench in BENCHMARK_TO_SKILL:
        if bench in parts:
            return bench
    if "plan_bench" in parts:
        return "plan_bench"
    return None


def load_bigbench_results(bigbench_dir: Path) -> List[Dict[str, Any]]:
    """
    Walk bigbench_dir for summary.json files written by the BBL24 runner.
    Returns list of dicts with keys: backend, model, architecture, weighted_accuracy.
    """
    rows = []
    for p in bigbench_dir.rglob("summary.json"):
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception:
            continue
        # Only accept BBL24 summaries written by the new runner
        if d.get("suite") != "bbl24" or d.get("weighted_accuracy") is None:
            continue
        model_raw = d.get("model", "")
        rows.append({
            "backend":           d.get("backend", "azure"),
            "model":             normalize_model_key(model_raw) if model_raw else "",
            "architecture":      d.get("architecture", "one_shot"),
            "weighted_accuracy": float(d["weighted_accuracy"]),
            "_source":           str(p),
        })
    return rows


# ---------------------------------------------------------------------------
# Load skill benchmark results
# ---------------------------------------------------------------------------

def _extract_score(obj: Dict[str, Any]) -> Optional[float]:
    for key in ("accuracy", "success_rate"):
        if key in obj and obj[key] is not None:
            return float(obj[key])
    metrics = obj.get("metrics") or {}
    for key in ("top1_accuracy", "mean_composite_score"):
        if key in metrics:
            return float(metrics[key])
    if "mean_composite_score" in obj:
        return float(obj["mean_composite_score"])
    return None


def _extract_architecture(path: Path) -> str:
    """Infer architecture label from agent class name in path."""
    for part in reversed(path.parts):
        lp = part.lower()
        if "oneshot" in lp or "one_shot" in lp:
            return "one_shot"
        if "sequential" in lp:
            return "sequential"
        if "concurrent" in lp:
            return "concurrent"
        if "groupchat" in lp or "group_chat" in lp:
            return "group_chat"
    return "one_shot"


def _extract_backend(path: Path) -> str:
    parts_lower = [p.lower() for p in path.parts]
    if "ollama" in parts_lower:
        return "ollama"
    return "azure"


def load_skill_scores(results_root: Path) -> Dict[Tuple[str, str, str], Dict[str, List[float]]]:
    """
    Walk results_root for skill benchmark summary.json files.
    Returns { (backend, model, architecture) -> { benchmark_name: [scores] } }.
    """
    scores: Dict[Tuple[str, str, str], Dict[str, List[float]]] = {}

    for p in results_root.rglob("summary.json"):
        benchmark = _benchmark_from_path(p)
        if benchmark is None or benchmark not in BENCHMARK_TO_SKILL:
            continue
        try:
            with open(p) as f:
                data = json.load(f)
        except Exception:
            continue

        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                continue
            model_raw = item.get("model") or ""
            if not model_raw:
                # Try to infer from path
                parts = p.parts
                agent_classes = {
                    "OneShotAgent", "OllamaAgent", "SequentialAgent",
                    "ConcurrentAgent", "GroupChatAgent",
                    "OllamaSequentialAgent", "OllamaConcurrentAgent", "OllamaGroupChatAgent",
                }
                for part in reversed(parts):
                    if part in agent_classes:
                        model_raw = p.parent.parent.name  # model is grandparent of Agent dir
                        break
            if not model_raw:
                continue
            model = normalize_model_key(model_raw)
            arch = _extract_architecture(p)
            backend = _extract_backend(p)
            score = _extract_score(item)
            if score is None:
                continue
            key = (backend, model, arch)
            if key not in scores:
                scores[key] = {}
            if benchmark not in scores[key]:
                scores[key][benchmark] = []
            scores[key][benchmark].append(score)

    return scores


# ---------------------------------------------------------------------------
# Join
# ---------------------------------------------------------------------------

def compute_skill_score(
    benchmark_scores: Dict[str, List[float]],
    skill: str,
) -> Optional[float]:
    """Mean of best scores across benchmarks in the skill bucket."""
    benchmarks = SKILL_BUCKETS.get(skill, [])
    vals = []
    for b in benchmarks:
        if b in benchmark_scores and benchmark_scores[b]:
            vals.append(max(benchmark_scores[b]))
    return sum(vals) / len(vals) if vals else None


def join(
    bigbench_rows: List[Dict[str, Any]],
    skill_scores: Dict[Tuple[str, str, str], Dict[str, List[float]]],
) -> List[Dict[str, Any]]:
    """
    Merge BBL rows with skill scores.
    Returns one row per (backend, model, architecture) with all skill scores included.
    """
    joined = []
    for row in bigbench_rows:
        key = (row["backend"], row["model"], row["architecture"])
        bench_scores = skill_scores.get(key, {})

        # Also try without backend (Azure results may lack explicit backend label)
        if not bench_scores:
            for candidate_backend in ("azure", "ollama"):
                alt_key = (candidate_backend, row["model"], row["architecture"])
                if alt_key in skill_scores:
                    bench_scores = skill_scores[alt_key]
                    break

        out = {
            "backend":           row["backend"],
            "model":             row["model"],
            "architecture":      row["architecture"],
            "weighted_accuracy": row["weighted_accuracy"],
            "_bbl_source":       row.get("_source", ""),
        }
        for skill in SKILLS:
            s = compute_skill_score(bench_scores, skill)
            out[skill] = s
        joined.append(out)
    return joined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Join BBL24 results with skill benchmark scores")
    parser.add_argument(
        "--bigbench-dir", type=Path, default=Path("results/bigbench_lite"),
        help="Root of BIG-bench Lite results (default: results/bigbench_lite)",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results"),
        help="Root of skill benchmark results (default: results)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/bigbench_skill_join.json"),
        help="Output JSON path (default: results/bigbench_skill_join.json)",
    )
    parser.add_argument("--csv", action="store_true", help="Also write a CSV alongside the JSON")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"Loading BBL24 results from: {args.bigbench_dir}")
    bbl_rows = load_bigbench_results(args.bigbench_dir)
    print(f"  Found {len(bbl_rows)} (backend, model, architecture) BBL24 entries.")

    if not bbl_rows:
        print("No BBL24 results found. Writing empty joined file; run scripts/run_bigbench_lite_sweep.py to populate.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump([], f, indent=2)
        print(f"  Wrote empty join: {args.output}")
        return 0

    print(f"\nLoading skill scores from: {args.results_dir}")
    skill_scores = load_skill_scores(args.results_dir)
    print(f"  Found {len(skill_scores)} (backend, model, architecture) skill entries.")

    joined = join(bbl_rows, skill_scores)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(joined, f, indent=2)
    print(f"\nJoined {len(joined)} rows → {args.output}")

    if args.csv:
        csv_path = args.output.with_suffix(".csv")
        import csv
        fieldnames = ["backend", "model", "architecture", "weighted_accuracy"] + SKILLS
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(joined)
        print(f"CSV → {csv_path}")

    if args.verbose:
        print("\nPreview (first 10 rows):")
        print(f"{'backend':<8} {'model':<30} {'arch':<15} {'bbl':>6}  " +
              "  ".join(f"{s[:6]:>6}" for s in SKILLS))
        print("-" * 90)
        for r in joined[:10]:
            skill_vals = "  ".join(
                f"{r[s] * 100:6.1f}" if r.get(s) is not None else "   N/A"
                for s in SKILLS
            )
            print(
                f"{r['backend']:<8} {r['model']:<30} {r['architecture']:<15} "
                f"{r['weighted_accuracy'] * 100:6.1f}  {skill_vals}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
