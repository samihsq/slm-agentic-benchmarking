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
    "instruction_following": ["instruction_following"],  # matrix IF only; word IF is too easy
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


def _wilson_se(p: float, n: int, z: float = 1.96) -> float:
    """
    Wilson CI half-width divided by z, as a drop-in SE replacement.
    Always > 0, handles p=0 and p=1 correctly.
    Wilson half-width: z * sqrt(p_tilde*(1-p_tilde) / (n+z^2))
    where p_tilde = (n*p + z^2/2) / (n + z^2)
    """
    import math
    n = max(n, 1)
    z2 = z * z
    p_tilde = (n * p + z2 / 2.0) / (n + z2)
    hw = z * math.sqrt(max(p_tilde * (1.0 - p_tilde), 1e-12) / (n + z2))
    return hw / z  # SE-equivalent so caller's 1.96× gives the Wilson CI


def load_bigbench_results(bigbench_dir: Path) -> List[Dict[str, Any]]:
    """
    Walk bigbench_dir for summary.json files written by the BBL24 runner.
    Returns list of dicts with keys: backend, model, architecture, weighted_accuracy,
    weighted_accuracy_se (std of per-task accuracies / sqrt(n_tasks)).
    """
    import math
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

        # Compute SE from per-task accuracy variance; fall back to Wilson SE when std=0
        breakdown = d.get("breakdown_by_task", {})
        task_accs = [
            v["accuracy"] for v in breakdown.values()
            if isinstance(v, dict) and v.get("accuracy") is not None
        ]
        task_ns = [
            v["total"] for v in breakdown.values()
            if isinstance(v, dict) and v.get("total") is not None
        ]
        if len(task_accs) >= 2:
            mean_acc = sum(task_accs) / len(task_accs)
            variance = sum((a - mean_acc) ** 2 for a in task_accs) / (len(task_accs) - 1)
            if variance > 0:
                se = math.sqrt(variance / len(task_accs))
            else:
                # All tasks same accuracy — use Wilson SE on overall accuracy
                n_total = sum(task_ns) if task_ns else len(task_accs) * 20
                wacc = float(d["weighted_accuracy"])
                se = _wilson_se(wacc, n_total)
        else:
            se = None

        rows.append({
            "backend":              d.get("backend", "azure"),
            "model":                normalize_model_key(model_raw) if model_raw else "",
            "architecture":         d.get("architecture", "one_shot"),
            "weighted_accuracy":    float(d["weighted_accuracy"]),
            "weighted_accuracy_se": se,
            "_source":              str(p),
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
    # Also detect Ollama from agent class name in path
    ollama_agents = {"ollamaagent", "ollamasequentialagent", "ollamaconcurrentagent", "ollamagroupchatagent"}
    if any(p in ollama_agents for p in parts_lower):
        return "ollama"
    return "azure"


def _extract_n(obj: Dict[str, Any]) -> Optional[int]:
    """Extract number of examples from a skill benchmark summary."""
    for key in ("num_tasks", "num_rollouts", "num_examples", "total_examples", "n"):
        if key in obj and obj[key] is not None:
            try:
                return int(obj[key])
            except (TypeError, ValueError):
                pass
    return None


def load_skill_scores(
    results_root: Path,
) -> Tuple[
    Dict[Tuple[str, str, str], Dict[str, List[float]]],
    Dict[Tuple[str, str, str], Dict[str, List[int]]],
]:
    """
    Walk results_root for skill benchmark summary.json files.
    Returns:
      scores:    { (backend, model, architecture) -> { benchmark_name: [scores] } }
      n_samples: { (backend, model, architecture) -> { benchmark_name: [n_examples] } }
    """
    scores: Dict[Tuple[str, str, str], Dict[str, List[float]]] = {}
    n_samples: Dict[Tuple[str, str, str], Dict[str, List[int]]] = {}

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
            n = _extract_n(item)
            key = (backend, model, arch)
            if key not in scores:
                scores[key] = {}
                n_samples[key] = {}
            if benchmark not in scores[key]:
                scores[key][benchmark] = []
                n_samples[key][benchmark] = []
            scores[key][benchmark].append(score)
            if n is not None:
                n_samples[key][benchmark].append(n)

    return scores, n_samples


# ---------------------------------------------------------------------------
# Join
# ---------------------------------------------------------------------------

def _best_score_for_benchmark(
    scores: List[float],
    ns: List[int],
) -> float:
    """Pick the most representative score for a benchmark with multiple runs.

    Strategy: among runs with the maximum sample size, exclude any that
    scored exactly 0 when other runs with the same n scored > 0 (these are
    likely infrastructure failures, e.g. all-Error traces).  Then average
    the remaining runs.  Falls back to mean of all runs when sample sizes
    are unavailable.
    """
    if ns and len(ns) == len(scores):
        max_n = max(ns)
        candidates = [(s, n) for s, n in zip(scores, ns) if n == max_n]
        # If some max-n runs scored > 0 and others scored exactly 0,
        # drop the zeros (likely broken runs).
        nonzero = [s for s, _ in candidates if s > 0]
        if nonzero and len(nonzero) < len(candidates):
            return sum(nonzero) / len(nonzero)
        best = [s for s, _ in candidates]
        return sum(best) / len(best)
    # No sample-size info: mean across runs
    return sum(scores) / len(scores)


def compute_skill_score(
    benchmark_scores: Dict[str, List[float]],
    skill: str,
    benchmark_n: Optional[Dict[str, List[int]]] = None,
) -> Optional[float]:
    """Mean of representative scores across benchmarks in the skill bucket.

    For each benchmark, we average runs with the largest sample size
    (most representative) rather than ``max`` which cherry-picks the best
    run regardless of sample size.
    """
    benchmarks = SKILL_BUCKETS.get(skill, [])
    vals = []
    for b in benchmarks:
        if b in benchmark_scores and benchmark_scores[b]:
            ns = (benchmark_n or {}).get(b, [])
            vals.append(_best_score_for_benchmark(benchmark_scores[b], ns))
    return sum(vals) / len(vals) if vals else None


def compute_skill_se(
    benchmark_scores: Dict[str, List[float]],
    benchmark_n: Dict[str, List[int]],
    skill: str,
) -> Optional[float]:
    """
    Standard error of the skill score.
    - Multiple benchmarks in bucket with spread: std(scores) / sqrt(n_benchmarks)
    - Multiple benchmarks all same score, or single benchmark: Wilson SE
    """
    import math
    benchmarks = SKILL_BUCKETS.get(skill, [])
    vals, ns = [], []
    for b in benchmarks:
        if b in benchmark_scores and benchmark_scores[b]:
            scores = benchmark_scores[b]
            b_ns = benchmark_n.get(b, [])
            val = _best_score_for_benchmark(scores, b_ns)
            vals.append(val)
            if b_ns:
                max_n = max(b_ns)
                ns.append(max_n)
            else:
                ns.append(50)  # fallback
    if not vals:
        return None
    if len(vals) >= 2:
        mean_v = sum(vals) / len(vals)
        variance = sum((v - mean_v) ** 2 for v in vals) / (len(vals) - 1)
        if variance > 0:
            return math.sqrt(variance / len(vals))
        # All benchmarks identical — fall through to Wilson pooled over benchmarks
    # Single benchmark or all-identical: mean Wilson SE over available benchmarks
    wilson_ses = [
        _wilson_se(p, n)
        for p, n in zip(vals, ns if len(ns) == len(vals) else [50] * len(vals))
    ]
    return sum(wilson_ses) / len(wilson_ses)


def join(
    bigbench_rows: List[Dict[str, Any]],
    skill_scores: Dict[Tuple[str, str, str], Dict[str, List[float]]],
    skill_n: Dict[Tuple[str, str, str], Dict[str, List[int]]],
) -> List[Dict[str, Any]]:
    """
    Merge BBL rows with skill scores.
    Returns one row per (backend, model, architecture) with skill scores and SEs included.
    """
    joined = []
    for row in bigbench_rows:
        model = row["model"]
        arch = row["architecture"]

        # Merge skill scores across all backends and architectures for this model.
        # Priority: exact (backend, model, arch) → any backend same arch → any backend one_shot.
        # Merge per-benchmark so that e.g. criticality from azure backend fills in
        # even when the primary match has only instruction_following from ollama.
        merged_scores: Dict[str, List[float]] = {}
        merged_n: Dict[str, List[int]] = {}

        candidate_keys = []
        for be in (row["backend"], "azure", "ollama"):
            candidate_keys.append((be, model, arch))
        for be in (row["backend"], "azure", "ollama"):
            candidate_keys.append((be, model, "one_shot"))
        # Deduplicate while preserving order
        seen_keys: set = set()
        unique_keys = []
        for k in candidate_keys:
            if k not in seen_keys:
                seen_keys.add(k)
                unique_keys.append(k)

        for k in unique_keys:
            if k in skill_scores:
                for bench, scores_list in skill_scores[k].items():
                    if bench not in merged_scores:
                        merged_scores[bench] = scores_list
                        if k in skill_n and bench in skill_n[k]:
                            merged_n[bench] = skill_n[k][bench]

        out = {
            "backend":              row["backend"],
            "model":                model,
            "architecture":         arch,
            "weighted_accuracy":    row["weighted_accuracy"],
            "weighted_accuracy_se": row.get("weighted_accuracy_se"),
            "_bbl_source":          row.get("_source", ""),
        }
        for skill in SKILLS:
            out[skill] = compute_skill_score(merged_scores, skill, benchmark_n=merged_n)
            out[f"{skill}_se"] = compute_skill_se(merged_scores, merged_n, skill)
        joined.append(out)
    return joined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Join BBL24 results with skill benchmark scores")
    parser.add_argument(
        "--bigbench-dir", type=Path, nargs="+",
        default=[Path("results/bigbench_lite_chained"), Path("results/bigbench_lite")],
        help="Root(s) of BIG-bench Lite results (default: results/bigbench_lite_chained results/bigbench_lite). First listed takes priority on duplicates.",
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

    bbl_rows = []
    for bbl_dir in args.bigbench_dir:
        if bbl_dir.exists():
            print(f"Loading BBL24 results from: {bbl_dir}")
            rows = load_bigbench_results(bbl_dir)
            print(f"  Found {len(rows)} (backend, model, architecture) BBL24 entries.")
            bbl_rows.extend(rows)
        else:
            print(f"Skipping (not found): {bbl_dir}")

    # Deduplicate: if same (backend, model, architecture) appears in both dirs,
    # prefer the entry from the first directory listed (i.e. chained over old CrewAI runs).
    seen = set()
    deduped = []
    for r in bbl_rows:
        key = (r["backend"], r["model"], r["architecture"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    bbl_rows = deduped
    print(f"  Total unique: {len(bbl_rows)} (backend, model, architecture) BBL24 entries.")

    if not bbl_rows:
        print("No BBL24 results found. Writing empty joined file; run scripts/run_bigbench_lite_sweep.py to populate.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump([], f, indent=2)
        print(f"  Wrote empty join: {args.output}")
        return 0

    print(f"\nLoading skill scores from: {args.results_dir}")
    skill_scores, skill_n = load_skill_scores(args.results_dir)
    print(f"  Found {len(skill_scores)} (backend, model, architecture) skill entries.")

    joined = join(bbl_rows, skill_scores, skill_n)

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
