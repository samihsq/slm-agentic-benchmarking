#!/usr/bin/env python3
"""
Create small CSV "review packs" from already-run benchmark artifacts.

This script DOES NOT run any models. It only reads existing outputs like:
  - results/**/results.jsonl
  - results/**/<task_id>/trace.json

Then it writes CSVs with sampled examples to speed up manual review.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _truncate(s: str, n: int = 240) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else (s[: n - 1] + "…")


def _percentiles(values: List[float], ps: Iterable[int] = (10, 25, 50, 75, 90)) -> Dict[str, float]:
    if not values:
        return {}
    vals = sorted(values)
    out: Dict[str, float] = {}
    n = len(vals)
    for p in ps:
        # nearest-rank
        k = max(1, int(round(p / 100 * n))) - 1
        k = min(max(k, 0), n - 1)
        out[f"p{p}"] = float(vals[k])
    return out


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    common = [
        "benchmark",
        "run_dir",
        "results_jsonl",
        "task_id",
        "score",
        "label",
        "trace_path",
        "pred_preview",
        "ref_preview",
    ]
    keys = set().union(*(r.keys() for r in rows))
    header = [k for k in common if k in keys] + sorted(k for k in keys if k not in common)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _find_files(root: Path, name: str) -> List[Path]:
    hits: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        if name in filenames:
            hits.append(Path(dirpath) / name)
    return sorted(hits)


def _guess_score_key(rows: List[Dict[str, Any]], preferred: List[str]) -> Optional[str]:
    if not rows:
        return None
    keys = set().union(*(r.keys() for r in rows))
    for k in preferred:
        if k in keys:
            return k
    if "score" in keys:
        return "score"
    return None


@dataclass(frozen=True)
class RunIndex:
    benchmark: str
    run_dir: Path
    results_jsonl: Path


def _index_runs(results_root: Path) -> List[RunIndex]:
    indices: List[RunIndex] = []
    for p in _find_files(results_root, "results.jsonl"):
        parts = list(p.parts)
        bench = "unknown"
        for candidate in ("summarization", "criticality_v2", "criticality", "recall", "episodic_memory"):
            if candidate in parts:
                bench = candidate
                break
        indices.append(RunIndex(benchmark=bench, run_dir=p.parent, results_jsonl=p))
    return indices


def _summarization_review_pack(idx: RunIndex, out_dir: Path, n: int) -> Tuple[Path, Dict[str, Any]]:
    rows = _read_jsonl(idx.results_jsonl)
    score_key = _guess_score_key(
        rows,
        [
            # Prefer the runner's normalized scalar
            "score",
            # ROUGE variants
            "rougeL",
            "rouge_l",
            "rouge-l",
            "rougeL_f1",
            "rougeL_f",
            # BERTScore variants
            "bertscore_f1",
            "bert_score_f1",
            "bertscore",
            # BARTScore variants
            "bartscore_geomean_prob",
            "bartscore",
            "bart_score",
        ],
    )
    if not score_key:
        return out_dir / "summarization_review.csv", {"error": "No score key found in results.jsonl"}

    def score_of(r: Dict[str, Any]) -> float:
        try:
            return float(r.get(score_key, 0.0) or 0.0)
        except Exception:
            return 0.0

    rows = [r for r in rows if r.get("task_id") is not None]
    rows_sorted = sorted(rows, key=score_of)

    low = rows_sorted[:n]
    high = list(reversed(rows_sorted[-n:])) if len(rows_sorted) >= n else list(reversed(rows_sorted))

    samples: List[Dict[str, Any]] = []
    for label, subset in (("low", low), ("high", high)):
        for r in subset:
            task_id = str(r["task_id"])
            trace_path = idx.run_dir / task_id / "trace.json"
            t = _safe_read_json(trace_path) or {}
            pred = t.get("predicted") or t.get("final_output") or ""
            ref = t.get("correct") or ""
            samples.append(
                {
                    "benchmark": "summarization",
                    "run_dir": str(idx.run_dir),
                    "results_jsonl": str(idx.results_jsonl),
                    "task_id": task_id,
                    "score": score_of(r),
                    "label": label,
                    "trace_path": str(trace_path),
                    "pred_preview": _truncate(str(pred)),
                    "ref_preview": _truncate(str(ref)),
                    "score_key": score_key,
                }
            )

    scores = [score_of(r) for r in rows_sorted]
    stats = {
        "count": len(scores),
        "score_key": score_key,
        "min": float(min(scores)) if scores else None,
        "max": float(max(scores)) if scores else None,
        "mean": float(statistics.mean(scores)) if scores else None,
        "median": float(statistics.median(scores)) if scores else None,
        "percentiles": _percentiles(scores),
    }

    out_csv = out_dir / "summarization_review.csv"
    _write_csv(out_csv, samples)
    return out_csv, stats


def _criticality_v2_review_pack(idx: RunIndex, out_dir: Path, n: int) -> Tuple[Path, Dict[str, Any]]:
    rows = _read_jsonl(idx.results_jsonl)

    def ffloat(x: Any) -> Optional[float]:
        try:
            return None if x is None else float(x)
        except Exception:
            return None

    def is_wrong(r: Dict[str, Any]) -> bool:
        v = r.get("is_correct")
        if isinstance(v, bool):
            return not v
        v2 = r.get("match")
        return bool(v2) is False

    def is_right(r: Dict[str, Any]) -> bool:
        v = r.get("is_correct")
        if isinstance(v, bool):
            return v
        v2 = r.get("match")
        return bool(v2) is True

    for r in rows:
        r["_margin"] = ffloat(r.get("margin"))
        r["_entropy"] = ffloat(r.get("entropy"))

    wrong = [r for r in rows if r.get("task_id") is not None and is_wrong(r) and r.get("_margin") is not None]
    right = [r for r in rows if r.get("task_id") is not None and is_right(r) and r.get("_margin") is not None]

    wrong_sorted = sorted(wrong, key=lambda r: (r.get("_margin") or 0.0), reverse=True)[:n]
    right_sorted = sorted(right, key=lambda r: (r.get("_margin") or 0.0), reverse=True)[:n]

    samples: List[Dict[str, Any]] = []
    for label, subset in (("confident_wrong", wrong_sorted), ("confident_correct", right_sorted)):
        for r in subset:
            task_id = str(r["task_id"])
            trace_path = idx.run_dir / task_id / "trace.json"
            t = _safe_read_json(trace_path) or {}
            samples.append(
                {
                    "benchmark": "criticality_v2",
                    "run_dir": str(idx.run_dir),
                    "results_jsonl": str(idx.results_jsonl),
                    "task_id": task_id,
                    "score": 1.0 if is_right(r) else 0.0,
                    "label": label,
                    "trace_path": str(trace_path),
                    "pred_preview": _truncate(str(t.get("predicted") or t.get("content") or r.get("predicted") or "")),
                    "ref_preview": _truncate(str(t.get("correct_label") or r.get("correct") or r.get("correct_label") or "")),
                    "margin": r.get("_margin"),
                    "entropy": r.get("_entropy"),
                    "extraction_source": r.get("extraction_source"),
                }
            )

    margins = [ffloat(r.get("margin")) for r in rows if ffloat(r.get("margin")) is not None]
    stats = {"count": len(rows), "with_margin": len(margins), "margin_percentiles": _percentiles(margins)}

    out_csv = out_dir / "criticality_v2_review.csv"
    _write_csv(out_csv, samples)
    return out_csv, stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Create review packs from an existing results/ directory")
    ap.add_argument("--results-dir", type=str, default="results", help="Path to results root (default: ./results)")
    ap.add_argument("--n", type=int, default=25, help="Number of samples per slice (default: 25)")
    args = ap.parse_args()

    results_root = Path(args.results_dir).expanduser().resolve()
    out_dir = Path("review_pack") / time.strftime("%Y%m%d_%H%M%S")

    if not results_root.exists():
        print(f"❌ Results directory not found: {results_root}")
        return 2

    indices = _index_runs(results_root)
    if not indices:
        print(f"❌ No results.jsonl found under: {results_root}")
        return 3

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Found {len(indices)} results.jsonl files under {results_root}")
    print(f"Writing review pack to: {out_dir}")

    sum_runs = [i for i in indices if i.benchmark == "summarization"]
    if sum_runs:
        i = sum_runs[-1]
        out_csv, stats = _summarization_review_pack(i, out_dir, args.n)
        print(f"📝 Summarization review CSV: {out_csv}")
        print("   stats:", json.dumps(stats, indent=2))
    else:
        print("ℹ️  No summarization runs found under results/ (expected: results/summarization/**/results.jsonl)")

    cv2_runs = [i for i in indices if i.benchmark == "criticality_v2"]
    if cv2_runs:
        i = cv2_runs[-1]
        out_csv, stats = _criticality_v2_review_pack(i, out_dir, args.n)
        print(f"📝 Criticality v2 review CSV: {out_csv}")
        print("   stats:", json.dumps(stats, indent=2))
    else:
        print("ℹ️  No criticality_v2 runs found under results/ (expected: results/criticality_v2/**/results.jsonl)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

