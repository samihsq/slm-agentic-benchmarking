#!/usr/bin/env python3
"""
Generate 5 scatter plots: skill score (x) vs. BIG-bench Lite accuracy (y).

Each plot covers one skill bucket.  Every model contributes 4 points (one per
architecture).  Points are colored by model and shaped by architecture.

Input: joined JSON produced by scripts/join_bigbench_skill_scores.py
       (default: results/bigbench_skill_join.json)

Usage
-----
python scripts/plot_skill_vs_bigbench.py
python scripts/plot_skill_vs_bigbench.py --input results/bigbench_skill_join.json --output-dir plots/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SKILLS = ["planning", "criticality", "recall", "summarization", "instruction_following"]

ARCHITECTURE_MARKERS = {
    "one_shot":   "o",
    "sequential": "s",
    "concurrent": "^",
    "group_chat": "D",
}

ARCHITECTURE_LABELS = {
    "one_shot":   "One-shot",
    "sequential": "Sequential",
    "concurrent": "Concurrent",
    "group_chat": "Group Chat",
}

# 15 visually distinct colors for up to 15 models
_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
]


def _assign_colors(models: List[str]) -> Dict[str, str]:
    return {m: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(sorted(set(models)))}


def _short_model(model: str) -> str:
    """Shorten long model keys for legend readability."""
    return model.replace("phi4-mini-reasoning-ollama", "phi4-mini-rea(ol)").replace(
        "llama-3.2-11b-vision", "llama-3.2-11b"
    ).replace("llama-3.3-70b", "llama-3.3-70b").replace(
        "falcon-h1-90m", "falcon-90m"
    )


def load_joined(path: Path) -> List[Dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def plot_all(
    joined: List[Dict[str, Any]],
    output_dir: Path,
    show: bool,
    dpi: int,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.lines as mlines
    except ImportError:
        print("ERROR: matplotlib is required. Install with: pip install matplotlib")
        sys.exit(1)

    all_models = sorted({r["model"] for r in joined})
    all_archs = sorted({r["architecture"] for r in joined})
    color_map = _assign_colors(all_models)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = []

    for skill in SKILLS:
        # Filter to rows that have both skill score and BBL accuracy
        rows = [r for r in joined if r.get(skill) is not None and r.get("weighted_accuracy") is not None]
        if not rows:
            print(f"  Skipping {skill}: no data")
            continue

        fig, ax = plt.subplots(figsize=(9, 6))

        for r in rows:
            x = r[skill] * 100
            y = r["weighted_accuracy"] * 100
            model = r["model"]
            arch = r["architecture"]
            backend = r.get("backend", "azure")
            marker = ARCHITECTURE_MARKERS.get(arch, "o")
            color = color_map.get(model, "#888888")
            # Outline Azure points with black, Ollama with grey, for quick backend discrimination
            edge_color = "#000000" if backend == "azure" else "#888888"
            ax.scatter(
                x, y,
                marker=marker,
                c=color,
                edgecolors=edge_color,
                linewidths=0.8,
                s=80,
                zorder=3,
            )

        ax.set_xlabel(f"{skill.replace('_', ' ').title()} Score (%)", fontsize=11)
        ax.set_ylabel("BIG-bench Lite Accuracy (%) — equal task weight", fontsize=11)
        ax.set_title(
            f"BIG-bench Lite vs. {skill.replace('_', ' ').title()} Skill",
            fontsize=13, fontweight="bold",
        )
        ax.grid(True, linestyle="--", alpha=0.5)

        # Legend: color = model
        model_handles = [
            mpatches.Patch(color=color_map[m], label=_short_model(m))
            for m in sorted(set(r["model"] for r in rows))
        ]
        # Legend: shape = architecture
        arch_handles = [
            mlines.Line2D(
                [], [],
                color="grey",
                marker=ARCHITECTURE_MARKERS.get(a, "o"),
                linestyle="None",
                markersize=8,
                label=ARCHITECTURE_LABELS.get(a, a),
            )
            for a in sorted(set(r["architecture"] for r in rows))
        ]

        # Two-column legend: models on left, architectures on right
        leg1 = ax.legend(
            handles=model_handles, title="Model",
            bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, title_fontsize=8,
        )
        ax.add_artist(leg1)
        ax.legend(
            handles=arch_handles, title="Architecture",
            bbox_to_anchor=(1.01, 0.35), loc="upper left", fontsize=8, title_fontsize=8,
        )

        fig.tight_layout()
        out_path = output_dir / f"skill_vs_bigbench_{skill}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plot_paths.append(out_path)
        print(f"  Saved: {out_path}")
        plt.close(fig)

    # Write machine-readable summary alongside the plots
    summary = {
        "skills": SKILLS,
        "models": all_models,
        "architectures": all_archs,
        "rows": len(joined),
        "plot_paths": [str(p) for p in plot_paths],
        "color_map": {_short_model(m): color_map[m] for m in all_models},
        "architecture_markers": ARCHITECTURE_MARKERS,
    }
    summary_path = output_dir / "plot_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {summary_path}")

    if show:
        import subprocess
        for p in plot_paths:
            subprocess.Popen(["open", str(p)])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot skill score vs. BIG-bench Lite accuracy (5 plots)"
    )
    parser.add_argument(
        "--input", type=Path, default=Path("results/bigbench_skill_join.json"),
        help="Joined JSON from join_bigbench_skill_scores.py",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("plots/bigbench_skill"),
        help="Directory for output PNG files (default: plots/bigbench_skill)",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Output DPI (default: 150)")
    parser.add_argument("--show", action="store_true", help="Open plots after saving (macOS open)")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: joined data not found at {args.input}")
        print("Run scripts/join_bigbench_skill_scores.py first.")
        return 1

    print(f"Loading joined data from: {args.input}")
    joined = load_joined(args.input)
    print(f"  {len(joined)} rows")

    if not joined:
        print("No joined rows; creating output dir and summary only. Run the sweep and join to get plots.")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with open(args.output_dir / "plot_summary.json", "w") as f:
            json.dump({"skills": SKILLS, "rows": 0, "plot_paths": [], "message": "No data"}, f, indent=2)
        return 0

    print(f"\nGenerating 5 scatter plots → {args.output_dir}/")
    plot_all(joined, args.output_dir, show=args.show, dpi=args.dpi)
    return 0


if __name__ == "__main__":
    sys.exit(main())
