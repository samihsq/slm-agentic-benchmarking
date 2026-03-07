"""
Generate a network-graph PNG from data/skill_correlations.json.

Nodes  = skills (circular layout, clustered by category)
         Fill colour: green ↔ gray based on how many strong connections a node has
Edges  = perceived pairwise correlations
  • 3 discrete thickness tiers: strong / moderate / weak
  • colour  → green (positive) / red (negative)
  • hidden  → |r| < 0.20
  • no numeric labels on edges

Output → data/skill_correlation_map.png
"""

import json
import math
import pathlib

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── paths ───────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
JSON_PATH = DATA / "skill_correlations.json"
OUT_PATH  = DATA / "skill_correlation_map.png"

# ── load data ───────────────────────────────────────────────────────────
with open(JSON_PATH) as f:
    data = json.load(f)

labels = data["matrix"]["labels"]
matrix = np.array(data["matrix"]["values"])
n = len(labels)

# ── pretty display names ────────────────────────────────────────────────
DISPLAY = {
    "summarization":              "Summarization",
    "recall":                     "Recall",
    "criticality":                "Criticality",
    "instruction_following":      "Instruction\nFollowing",
    "logic_structured_reasoning": "Logic /\nStructured\nReasoning",
    "planning":                   "Planning",
    "moderation":                 "Moderation",
    "structured_output":          "Structured\nOutput",
    "tool_calling":               "Tool\nCalling",
    "task_tracking":              "Task\nTracking",
    "information_seeking":        "Information\nSeeking",
}

# ── collect edges ───────────────────────────────────────────────────────
edges = []
for i in range(n):
    for j in range(i + 1, n):
        r = matrix[i][j]
        edges.append((labels[i], labels[j], r))

# ── layout: circle with cluster ordering ────────────────────────────────
ordered = [
    "summarization",              # Text Processing
    "recall",
    "criticality",                # Evaluative
    "moderation",
    "logic_structured_reasoning", # Reasoning
    "planning",
    "task_tracking",              # Meta-cognitive
    "information_seeking",
    "tool_calling",               # Precise Execution
    "structured_output",
    "instruction_following",
]

radius = 6.0
pos = {}
for idx, node in enumerate(ordered):
    angle = 2 * math.pi * idx / len(ordered) - math.pi / 2
    pos[node] = (radius * math.cos(angle), radius * math.sin(angle))

# ── tier helpers (fully discrete) ───────────────────────────────────────
STRONG_THRESH = 0.6
MOD_THRESH    = 0.3
HIDE_THRESH   = 0.20

def tier(r: float):
    """Return (width, alpha, tier_name) for a given correlation value."""
    a = abs(r)
    if a >= STRONG_THRESH:
        return 7.0, 0.90, "strong"
    elif a >= MOD_THRESH:
        return 3.0, 0.55, "moderate"
    else:
        return 1.0, 0.25, "weak"

def edge_rgb(r: float):
    """Green for positive, red for negative."""
    if r >= 0:
        return (0.10, 0.65, 0.25)
    else:
        return (0.85, 0.18, 0.18)

# ── compute node "connectivity strength" ────────────────────────────────
# Count strong connections per node → drives fill colour (green ↔ gray)
strong_counts = {lab: 0 for lab in labels}
for u, v, r in edges:
    if abs(r) >= STRONG_THRESH:
        strong_counts[u] += 1
        strong_counts[v] += 1

max_strong = max(strong_counts.values()) or 1

def node_fill(node: str):
    """Interpolate between gray (0 strong connections) and green (max)."""
    t = strong_counts[node] / max_strong  # 0 → 1
    # gray  = (0.88, 0.88, 0.88)
    # green = (0.75, 0.94, 0.78)
    r = 0.88 + t * (0.75 - 0.88)
    g = 0.88 + t * (0.94 - 0.88)
    b = 0.88 + t * (0.78 - 0.88)
    return (r, g, b)

def node_edge_color(node: str):
    """Border: darker green for high-connectivity nodes, gray for low."""
    t = strong_counts[node] / max_strong
    # gray border = (0.55, 0.55, 0.55)
    # green border = (0.10, 0.50, 0.20)
    r = 0.55 + t * (0.10 - 0.55)
    g = 0.55 + t * (0.50 - 0.55)
    b = 0.55 + t * (0.20 - 0.55)
    return (r, g, b)

# ── figure ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 18))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_aspect("equal")
ax.axis("off")

ax.set_title(
    "Perceived Skill Score Correlations\n(Hypothesized — pending empirical validation)",
    fontsize=21, fontweight="bold", pad=28, fontfamily="sans-serif",
)

# ── draw edges (skip |r| < HIDE_THRESH) ────────────────────────────────
sorted_edges = sorted(edges, key=lambda e: abs(e[2]))

for u, v, r in sorted_edges:
    if abs(r) < HIDE_THRESH:
        continue

    x0, y0 = pos[u]
    x1, y1 = pos[v]
    width, alpha, _ = tier(r)
    rgb = edge_rgb(r)
    color = (*rgb, alpha)

    ax.plot([x0, x1], [y0, y1], color=color, linewidth=width,
            solid_capstyle="round", zorder=1)

# ── draw nodes ──────────────────────────────────────────────────────────
node_r = 0.78
for node in ordered:
    x, y = pos[node]
    fc = node_fill(node)
    ec = node_edge_color(node)
    circle = plt.Circle((x, y), node_r, facecolor=fc, edgecolor=ec,
                         linewidth=2.5, zorder=10)
    ax.add_patch(circle)
    display = DISPLAY.get(node, node)
    ax.text(x, y, display, fontsize=8.5, fontweight="bold", fontfamily="sans-serif",
            ha="center", va="center", zorder=11, color="#222222",
            linespacing=1.1)

# ── cluster labels ──────────────────────────────────────────────────────
cluster_labels = [
    ("Text\nProcessing",       0, 1),
    ("Evaluative\nJudgment",   2, 3),
    ("Reasoning\n& Strategy",  4, 5),
    ("Meta-\ncognitive",       6, 7),
    ("Precise\nExecution",     8, 10),
]

for clabel, start_idx, end_idx in cluster_labels:
    mid_idx = (start_idx + end_idx) / 2
    angle = 2 * math.pi * mid_idx / len(ordered) - math.pi / 2
    lx = (radius + 2.2) * math.cos(angle)
    ly = (radius + 2.2) * math.sin(angle)
    ax.text(lx, ly, clabel, fontsize=9, fontstyle="italic", fontfamily="sans-serif",
            ha="center", va="center", color="#888888", zorder=0)

# ── fit view (extra room at bottom for legend) ──────────────────────────
margin = 4.5
ax.set_xlim(-radius - margin, radius + margin)
ax.set_ylim(-radius - margin - 3.5, radius + margin)

# ── legend (single block at bottom, left-aligned) ──────────────────────
leg_left = -radius - margin + 1.0
leg_top  = -radius - margin - 0.5

# Row 1: edge thickness scale
sy = leg_top
samples = [
    (0.80, "Strong  (r ≥ 0.6)"),
    (0.45, "Moderate  (0.3 ≤ r < 0.6)"),
    (0.25, "Weak  (0.2 ≤ r < 0.3)"),
]
for val, label_text in samples:
    w, a, _ = tier(val)
    rgb = edge_rgb(val)
    ax.plot([leg_left, leg_left + 1.6], [sy, sy], color=(*rgb, a),
            linewidth=w, solid_capstyle="round")
    ax.text(leg_left + 2.0, sy, label_text, fontsize=10, va="center",
            fontfamily="sans-serif", color="#444444")
    sy -= 0.85

ax.text(leg_left, sy - 0.15, "Connections with r < 0.2 hidden", fontsize=9,
        fontfamily="sans-serif", fontstyle="italic", color="#999999")

# Row 2: node colour legend (to the right of thickness scale)
nc_left = leg_left + 9.0
nc_sy = leg_top
for t_val, lbl in [(1.0, "Many strong connections"), (0.0, "Few strong connections")]:
    fc = (0.88 + t_val * (0.75 - 0.88), 0.88 + t_val * (0.94 - 0.88), 0.88 + t_val * (0.78 - 0.88))
    ec_ = (0.55 + t_val * (0.10 - 0.55), 0.55 + t_val * (0.50 - 0.55), 0.55 + t_val * (0.20 - 0.55))
    c = plt.Circle((nc_left + 0.3, nc_sy), 0.32, facecolor=fc, edgecolor=ec_,
                    linewidth=2.0, zorder=10)
    ax.add_patch(c)
    ax.text(nc_left + 0.9, nc_sy, lbl, fontsize=10, va="center",
            fontfamily="sans-serif", color="#444444")
    nc_sy -= 0.85

# Row 3: direction (below node colour)
green_patch = mpatches.Patch(color=(0.10, 0.65, 0.25), label="Positive correlation")
red_patch   = mpatches.Patch(color=(0.85, 0.18, 0.18), label="Negative correlation")
dir_sy = nc_sy - 0.3
ax.add_patch(mpatches.FancyBboxPatch((nc_left, dir_sy - 0.2), 0.5, 0.35,
             boxstyle="round,pad=0.05", facecolor=(0.10, 0.65, 0.25), edgecolor="none"))
ax.text(nc_left + 0.75, dir_sy - 0.02, "Positive correlation", fontsize=10,
        va="center", fontfamily="sans-serif", color="#444444")
dir_sy -= 0.75
ax.add_patch(mpatches.FancyBboxPatch((nc_left, dir_sy - 0.2), 0.5, 0.35,
             boxstyle="round,pad=0.05", facecolor=(0.85, 0.18, 0.18), edgecolor="none"))
ax.text(nc_left + 0.75, dir_sy - 0.02, "Negative correlation", fontsize=10,
        va="center", fontfamily="sans-serif", color="#444444")

fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight", facecolor="white", pad_inches=0.4)
print(f"Saved → {OUT_PATH}")
plt.close(fig)
