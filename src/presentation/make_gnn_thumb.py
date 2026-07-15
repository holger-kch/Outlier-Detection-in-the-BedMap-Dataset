#!/usr/bin/env python3
"""Tiny, text-free thumbnail of the supervised GNN on pseudo-labels.

to drop into the 'Supervised GNN on Pseudo labels' column of the strategy-overview
slide. Just a drawing: a small spatial graph where a few nodes carry pseudo-labels
(red = outlier, green = inlier) but most are unlabeled (grey); message passing
scores the central node from its neighbours. House style, transparent, square,
SVG (text as paths) + PNG into PP/."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(ROOT / ".mplcache")
os.environ["TMPDIR"] = str(ROOT / ".tmp")
os.makedirs(ROOT / ".mplcache", exist_ok=True)
os.makedirs(ROOT / ".tmp", exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

PP = ROOT / "PP"; PP.mkdir(exist_ok=True)
WHITE = "#FFFFFF"; GREY = "#C8CCD2"; DIM = "#9AA0A8"; ACC = "#6FD3FF"
RED = "#FF6B6B"; GRN = "#5AE0A0"
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path"})

# ---- even, deterministic point spread (phyllotaxis) ---------------------------
N = 19
i = np.arange(N)
r = np.sqrt(i / (N - 1)) * 0.92
theta = i * 2.399963229728653            # golden angle
P = np.column_stack([0.5 + 0.5 * r * np.cos(theta),
                     0.5 + 0.5 * r * np.sin(theta)])
# gentle deterministic jitter so it does not look like a perfect spiral
P[1:, 0] += 0.035 * np.sin(2.7 * i[1:])
P[1:, 1] += 0.035 * np.cos(1.9 * i[1:])

# ---- k-nearest-neighbour edges ------------------------------------------------
D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
np.fill_diagonal(D, np.inf)
edges = set()
for a in range(N):
    for b in np.argsort(D[a])[:3]:
        edges.add((min(a, b), max(a, b)))

# ---- pseudo-labels: few red/green, most grey; node 0 = focus (unlabeled) ------
focus = 0
red = [4, 9, 14]
grn = [3, 7, 11, 16]
col = {k: DIM for k in range(N)}
for k in red:
    col[k] = RED
for k in grn:
    col[k] = GRN
col[focus] = RED   # the node being scored lights up as an outlier
nbrs = [(b if a == focus else a) for (a, b) in edges if focus in (a, b)][:5]

# ================================ DRAW =========================================
fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=200)
fig.patch.set_alpha(0.0)
ax.set_facecolor("none"); ax.axis("off")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect("equal")

# edges
for a, b in edges:
    ax.plot([P[a, 0], P[b, 0]], [P[a, 1], P[b, 1]],
            color=ACC, lw=1.6, alpha=0.30, zorder=2, solid_capstyle="round")

# glow behind the labelled (outlier/inlier) nodes
for k in red:
    ax.scatter(*P[k], s=900, c=RED, alpha=0.13, edgecolors="none", zorder=3)
for k in grn:
    ax.scatter(*P[k], s=700, c=GRN, alpha=0.12, edgecolors="none", zorder=3)

# nodes
for k in range(N):
    if k == focus:
        continue
    big = k in red or k in grn
    ax.scatter(*P[k], s=170 if big else 95, c=col[k],
               alpha=0.97 if big else 0.8,
               edgecolors=WHITE, linewidths=0.8 if big else 0.0, zorder=5)

# message passing: neighbours -> focus node
fx, fy = P[focus]
for nb in nbrs:
    ax.add_patch(FancyArrowPatch((P[nb, 0], P[nb, 1]), (fx, fy),
                 arrowstyle="-|>", mutation_scale=15, lw=1.8,
                 color=WHITE, alpha=0.6, shrinkA=8, shrinkB=13, zorder=6))
ax.scatter([fx], [fy], s=900, c=RED, alpha=0.20, edgecolors="none", zorder=4)
ax.add_patch(Circle((fx, fy), 0.045, fill=False, edgecolor=WHITE,
             lw=2.2, alpha=0.95, zorder=7))
ax.scatter([fx], [fy], s=190, c=RED, alpha=0.97, edgecolors=WHITE,
           linewidths=0.9, zorder=8)

fig.savefig(PP / "gnn_thumb.png", transparent=True, dpi=400, bbox_inches="tight", pad_inches=0.02)
fig.savefig(PP / "gnn_thumb.svg", transparent=True, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

from PIL import Image
f = Image.open(PP / "gnn_thumb.png").convert("RGBA"); pad = 30; W_, H_ = f.size
c = Image.new("RGBA", (W_ + 2 * pad, H_ + 2 * pad), (12, 26, 48, 255))
c.alpha_composite(f, (pad, pad)); c.convert("RGB").save(PP / "_prev_gnn_thumb.png")
print("nodes", N, "edges", len(edges), "focus nbrs", nbrs)
print("wrote gnn_thumb.svg/png + _prev_gnn_thumb.png to", PP, " size", f.size)
