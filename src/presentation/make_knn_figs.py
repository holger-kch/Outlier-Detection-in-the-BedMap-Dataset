#!/usr/bin/env python3
"""Dark, transparent figures for the k-NN-16 map slides (2.1 / 2.2).
Same style as the pseudo-label figures. Saves SVG+PNG into presentation/PP/.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path

OUT = Path(__file__).resolve().parent / "figs"; OUT.mkdir(exist_ok=True)
PP = Path(__file__).resolve().parents[1] / "PP"; PP.mkdir(exist_ok=True)
rng = np.random.default_rng(3)

WHITE = "#FFFFFF"; GREY = "#C8CCD2"; ACC = "#6FD3FF"; RED = "#FF6B6B"; GRN = "#5AE0A0"; ORANGE = "#FFB74D"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": WHITE,
                     "axes.edgecolor": GREY, "xtick.color": GREY, "ytick.color": GREY, "axes.labelcolor": GREY})


def dark_preview(png, name):
    from PIL import Image
    f = Image.open(png).convert("RGBA"); pad = 40; W, H = f.size
    c = Image.new("RGBA", (W + 2 * pad, H + 2 * pad), (12, 17, 24, 255))
    c.alpha_composite(f, (pad, pad)); c.convert("RGB").save(PP / name)


# ============================================================ 2.1 the k-NN-16 map
fig, ax = plt.subplots(figsize=(7.6, 6.2), dpi=150)
ax.set_facecolor("none"); ax.set_aspect("equal"); ax.axis("off")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# build a few survey "tracks" (lines of measurement points) crossing a region
tracks = [((0.08, 0.22), (0.92, 0.52)), ((0.14, 0.88), (0.80, 0.12)),
          ((0.05, 0.55), (0.95, 0.72)), ((0.42, 0.06), (0.58, 0.94)), ((0.20, 0.10), (0.70, 0.95))]
pts = []
for (x0, y0), (x1, y1) in tracks:
    n = 15; t = np.linspace(0, 1, n)
    pts.append(np.column_stack([x0 + (x1 - x0) * t + rng.normal(0, 0.007, n),
                                y0 + (y1 - y0) * t + rng.normal(0, 0.007, n)]))
P = np.vstack(pts)
tree = cKDTree(P)

# faint background mesh: each point to its 2 nearest (shows "it's a graph")
_, idx2 = tree.query(P, k=3)
for i in range(len(P)):
    for j in idx2[i, 1:]:
        ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], color=GREY, lw=0.4, alpha=0.20, zorder=1)
ax.scatter(P[:, 0], P[:, 1], s=13, color=GREY, alpha=0.7, zorder=2)

# highlight ONE central point and its 16 nearest neighbours
c = int(np.argmin(np.hypot(P[:, 0] - 0.5, P[:, 1] - 0.5)))
_, ii = tree.query(P[c], k=17); nn = ii[1:]
for j in nn:
    ax.plot([P[c, 0], P[j, 0]], [P[c, 1], P[j, 1]], color=ACC, lw=1.3, alpha=0.95, zorder=3)
ax.scatter(P[nn, 0], P[nn, 1], s=42, color=ACC, edgecolor="k", lw=0.3, zorder=4)
ax.scatter([P[c, 0]], [P[c, 1]], s=130, color=WHITE, edgecolor="k", zorder=5)
ax.annotate("16 nearest\nneighbours", (P[c, 0], P[c, 1]), (0.72, 0.04), color=ACC, fontsize=12,
            ha="center", weight="bold", arrowprops=dict(arrowstyle="->", color=ACC))
ax.set_title("every point  →  its 16 nearest neighbours", color=WHITE, fontsize=15, pad=8)
fig.tight_layout()
fig.savefig(OUT / "fig_knn_map.png", transparent=True, bbox_inches="tight")
fig.savefig(PP / "knn_map.png", transparent=True, dpi=400, bbox_inches="tight")
fig.savefig(PP / "knn_map.svg", transparent=True, bbox_inches="tight")
plt.close(fig)
dark_preview(PP / "knn_map.png", "_prev_knn_map.png")

# ============================================================ 2.2 what a link measures (ramp vs spike)
fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 4.7), dpi=150)
for a in (axL, axR):
    a.set_facecolor("none"); a.axis("off"); a.set_xlim(0, 1); a.set_ylim(0, 1)


def caxes(ax):
    ax.annotate("", (0.97, 0.13), (0.10, 0.13), arrowprops=dict(arrowstyle="->", color=GREY, lw=1.3))
    ax.annotate("", (0.10, 0.95), (0.10, 0.13), arrowprops=dict(arrowstyle="->", color=GREY, lw=1.3))
    ax.text(0.97, 0.06, "distance", color=GREY, fontsize=10, ha="right")
    ax.text(0.05, 0.55, "ice depth", color=GREY, fontsize=10, rotation=90, va="center")


# left: gentle ramp -> real
caxes(axL)
xs = np.linspace(0.18, 0.9, 7); ys = 0.25 + 0.5 * (xs - 0.18) / 0.72
axL.plot(xs, ys, "-o", color=GRN, ms=6, lw=1.8, zorder=3)
axL.plot([xs[3], xs[4]], [ys[3], ys[3]], ls="--", color=WHITE, lw=1.0)
axL.plot([xs[4], xs[4]], [ys[3], ys[4]], ls="--", color=WHITE, lw=1.0)
axL.text(xs[4] + 0.02, (ys[3] + ys[4]) / 2, "small Δ", color=GRN, fontsize=11, va="center")
axL.set_title("gentle ramp  →  real", color=GRN, fontsize=13)

# right: sudden spike -> error
caxes(axR)
xs = np.linspace(0.18, 0.9, 7); ys = np.full(7, 0.32); ys[3] = 0.85
axR.plot(xs, ys, "-o", color=RED, ms=6, lw=1.8, zorder=3)
axR.plot([xs[2], xs[3]], [ys[2], ys[2]], ls="--", color=WHITE, lw=1.0)
axR.plot([xs[3], xs[3]], [ys[2], ys[3]], ls="--", color=WHITE, lw=1.0)
axR.text(xs[3] + 0.02, (ys[2] + ys[3]) / 2, "huge Δ", color=RED, fontsize=11, va="center")
axR.set_title("sudden spike  →  likely error", color=RED, fontsize=13)

fig.suptitle("each link stores:  distance  +  thickness change (Δ)", color=WHITE, fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "fig_knn_edge.png", transparent=True, bbox_inches="tight")
fig.savefig(PP / "knn_edge.png", transparent=True, dpi=400, bbox_inches="tight")
fig.savefig(PP / "knn_edge.svg", transparent=True, bbox_inches="tight")
plt.close(fig)
dark_preview(PP / "knn_edge.png", "_prev_knn_edge.png")

print("wrote knn_map + knn_edge (svg/png) to", PP)
