#!/usr/bin/env python3
"""Generate a network-graph PNG showing perceived correlations between skills."""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "skill_correlations.json"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "skill_correlation_map.png"

NICE_NAMES = {
    "criticality": "Criticality",
    "recall": "Recall",
    "episodic_memory": "Episodic\nMemory",
}

DESCRIPTIONS = {
    "criticality": "Argument quality\nassessment",
    "recall": "Keyword-based\nsentence retrieval",
    "episodic_memory": "Long-context\nstate tracking",
}


def correlation_color(r: float) -> tuple:
    """Map correlation value in [-1, 1] to an RGBA color.

    Positive  → green   (0, 0.6, 0.2)
    Negative  → red     (0.8, 0.1, 0.1)
    Intensity scales with |r|.
    """
    alpha = min(abs(r) * 1.3, 1.0)  # boost so moderate values are visible
    if r >= 0:
        return (0.1, 0.65, 0.25, alpha)
    else:
        return (0.85, 0.15, 0.15, alpha)


def edge_width(r: float) -> float:
    return 2.0 + abs(r) * 8.0  # range ~2–10


def main() -> None:
    with open(DATA_PATH) as f:
        data = json.load(f)

    labels = data["matrix"]["labels"]
    matrix = np.array(data["matrix"]["values"])
    n = len(labels)

    # --- Layout: equilateral triangle, centered ---
    angles = [math.pi / 2 + i * 2 * math.pi / n for i in range(n)]
    radius = 2.2
    positions = {labels[i]: (radius * math.cos(angles[i]), radius * math.sin(angles[i])) for i in range(n)}

    fig, ax = plt.subplots(figsize=(10, 9))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    # --- Draw edges ---
    for i in range(n):
        for j in range(i + 1, n):
            r = matrix[i][j]
            x0, y0 = positions[labels[i]]
            x1, y1 = positions[labels[j]]
            color = correlation_color(r)
            lw = edge_width(r)

            ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw, solid_capstyle="round", zorder=1)

            # Label on edge
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            # Offset label perpendicular to edge to avoid overlap with line
            dx, dy = x1 - x0, y1 - y0
            length = math.hypot(dx, dy)
            nx, ny = -dy / length, dx / length  # unit normal
            offset = 0.22
            lx, ly = mx + nx * offset, my + ny * offset

            sign = "+" if r >= 0 else ""
            ax.text(
                lx, ly, f"r = {sign}{r:.2f}",
                ha="center", va="center",
                fontsize=13, fontweight="bold",
                color=color[:3],  # strip alpha for text
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#fafafa", edgecolor="none", alpha=0.85),
                zorder=3,
            )

    # --- Draw nodes ---
    node_radius = 0.55
    for label in labels:
        x, y = positions[label]
        circle = plt.Circle((x, y), node_radius, color="white", ec="#333333", linewidth=2.2, zorder=4)
        ax.add_patch(circle)
        ax.text(x, y + 0.08, NICE_NAMES[label], ha="center", va="center",
                fontsize=14, fontweight="bold", color="#222222", zorder=5)
        ax.text(x, y - 0.65 - node_radius * 0.3, DESCRIPTIONS[label], ha="center", va="top",
                fontsize=9.5, color="#666666", fontstyle="italic", zorder=5)

    # --- Legend ---
    legend_elements = [
        mpatches.Patch(facecolor=(0.1, 0.65, 0.25, 0.9), edgecolor="none", label="Positive correlation"),
        mpatches.Patch(facecolor=(0.85, 0.15, 0.15, 0.9), edgecolor="none", label="Negative correlation"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=11, framealpha=0.9,
              edgecolor="#cccccc", fancybox=True)

    # --- Title ---
    ax.set_title(
        "Perceived Skill Score Correlations\n(Hypothesized — pending empirical validation)",
        fontsize=16, fontweight="bold", color="#222222", pad=18,
    )

    # --- Formatting ---
    margin = 1.6
    ax.set_xlim(-radius - margin, radius + margin)
    ax.set_ylim(-radius - margin - 0.4, radius + margin)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
