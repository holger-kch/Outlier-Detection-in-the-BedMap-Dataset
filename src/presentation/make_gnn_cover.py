#!/usr/bin/env python3
"""Cover / title figure for the GNN section. NO MATH — just a drawing that
gives a feel for our GNN: Antarctic ice-thickness survey points form a spatial
k-nearest-neighbour graph; the network learns each point from its neighbours
and the outliers (errors) light up red among the trusted points (cyan).
House style, transparent, SVG (text as paths) + PNG into PP/."""
import os
ROOT = "/lustre/hpc/icecube/holgerkc/ML-ICE-PROJECT"
os.environ["MPLCONFIGDIR"] = ROOT + "/.mplcache"
os.environ["TMPDIR"] = ROOT + "/.tmp"
os.makedirs(ROOT + "/.mplcache", exist_ok=True)
os.makedirs(ROOT + "/.tmp", exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from pathlib import Path

PP = Path(ROOT) / "PP"; PP.mkdir(exist_ok=True)
WHITE = "#FFFFFF"; GREY = "#C8CCD2"; DIM = "#9AA0A8"; ACC = "#6FD3FF"
RED = "#FF6B6B"; GRN = "#5AE0A0"; ORANGE = "#FFB74D"; PANEL = (0.05, 0.08, 0.12, 0.55)
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path"})

rng = np.random.default_rng(7)

# ---- build survey "tracks": gently curved polylines of points -----------------
GX0, GX1, GY0, GY1 = 0.045, 0.955, 0.115, 0.735     # graph region
tracks = []
for t in range(16):
    n = int(rng.integers(13, 26))
    x0 = rng.uniform(GX0 + 0.06, GX1 - 0.06)
    y0 = rng.uniform(GY0 + 0.06, GY1 - 0.06)
    a = rng.uniform(0, 2 * np.pi)
    step = rng.uniform(0.020, 0.030)
    xs, ys = [x0], [y0]
    for _ in range(n - 1):
        a += rng.normal() * 0.16                      # gentle curvature
        nx = xs[-1] + step * np.cos(a)
        ny = ys[-1] + step * np.sin(a)
        # soft reflect at the borders so tracks stay inside the card
        if nx < GX0 or nx > GX1:
            a = np.pi - a; nx = xs[-1] + step * np.cos(a)
        if ny < GY0 or ny > GY1:
            a = -a; ny = ys[-1] + step * np.sin(a)
        xs.append(min(max(nx, GX0), GX1)); ys.append(min(max(ny, GY0), GY1))
    tracks.append((np.array(xs), np.array(ys)))

# flatten to one point cloud, remember which track each point belongs to
P, tid = [], []
for i, (xs, ys) in enumerate(tracks):
    for x, y in zip(xs, ys):
        P.append((x, y)); tid.append(i)
P = np.array(P); tid = np.array(tid); N = len(P)

# ---- edges: along-track + a few cross-track nearest neighbours ----------------
along = []
off = 0
for xs, ys in tracks:
    for j in range(len(xs) - 1):
        along.append((off + j, off + j + 1))
    off += len(xs)

cross = set()
D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
np.fill_diagonal(D, np.inf)
for i in range(N):
    order = np.argsort(D[i])
    picked = 0
    for j in order:
        if D[i, j] > 0.055:
            break
        if tid[j] != tid[i]:
            cross.add((min(i, j), max(i, j))); picked += 1
        if picked >= 2:
            break

# ---- pick outliers: a contiguous bad run on one track + a few scattered -------
lengths = [len(xs) for xs, _ in tracks]
bad_track = int(np.argmax(lengths))
bad_idx = np.where(tid == bad_track)[0]
lo = len(bad_idx) // 5
run = bad_idx[lo: lo + max(6, len(bad_idx) // 2)]
is_out = np.zeros(N, bool)
is_out[run] = True
scatter_out = rng.choice(np.where(~is_out)[0], 4, replace=False)
is_out[scatter_out] = True

# ---- a focus node to hint at message passing (neighbours -> node) -------------
adj = {i: set() for i in range(N)}
for a_, b_ in along + list(cross):
    adj[a_].add(b_); adj[b_].add(a_)
cand = [i for i in range(N) if not is_out[i] and len(adj[i]) >= 4
        and 0.30 < P[i, 0] < 0.62 and 0.30 < P[i, 1] < 0.62]
focus = cand[0] if cand else int(np.argmax([len(adj[i]) for i in range(N)]))
nbrs = list(adj[focus])[:5]

# ================================ DRAW =========================================
fig, ax = plt.subplots(figsize=(12.6, 7.3), dpi=150)
fig.patch.set_alpha(0.0)
ax.set_facecolor("none"); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# subtle card behind the graph
ax.add_patch(FancyBboxPatch((0.018, 0.075), 0.964, 0.700,
             boxstyle="round,pad=0.004,rounding_size=0.02",
             linewidth=1.6, edgecolor=ACC, facecolor=PANEL, zorder=1))

# edges
for a_, b_ in along:
    ax.plot([P[a_, 0], P[b_, 0]], [P[a_, 1], P[b_, 1]],
            color=ACC, lw=1.1, alpha=0.30, zorder=2, solid_capstyle="round")
for a_, b_ in cross:
    ax.plot([P[a_, 0], P[b_, 0]], [P[a_, 1], P[b_, 1]],
            color=WHITE, lw=0.7, alpha=0.13, zorder=2, solid_capstyle="round")

# good nodes
good = ~is_out
ax.scatter(P[good, 0], P[good, 1], s=20, c=ACC, alpha=0.85,
           edgecolors="none", zorder=4)

# outliers: glow + node
ax.scatter(P[is_out, 0], P[is_out, 1], s=320, c=RED, alpha=0.12,
           edgecolors="none", zorder=3)
ax.scatter(P[is_out, 0], P[is_out, 1], s=58, c=RED, alpha=0.97,
           edgecolors=WHITE, linewidths=0.7, zorder=5)

# message-passing hint: neighbours -> focus node
fx, fy = P[focus]
for nb in nbrs:
    ax.add_patch(FancyArrowPatch((P[nb, 0], P[nb, 1]), (fx, fy),
                 arrowstyle="-|>", mutation_scale=11, lw=1.3,
                 color=WHITE, alpha=0.55, shrinkA=6, shrinkB=9, zorder=6))
ax.scatter([fx], [fy], s=260, c=ACC, alpha=0.16, edgecolors="none", zorder=4)
ax.add_patch(Circle((fx, fy), 0.018, fill=False, edgecolor=WHITE,
             lw=1.6, alpha=0.9, zorder=7))
ax.scatter([fx], [fy], s=30, c=WHITE, alpha=0.95, edgecolors="none", zorder=8)

# ---- text: kicker + title + subtitle (no math) --------------------------------
ax.text(0.5, 0.952, "B E D M A P   ·   O U T L I E R   D E T E C T I O N",
        color=ORANGE, fontsize=14, weight="bold", ha="center", va="center")
ax.text(0.5, 0.880, "Graph Neural Network", color=ACC, fontsize=33,
        weight="bold", ha="center", va="center")
ax.text(0.5, 0.812,
        "every ice-thickness point is scored from its neighbours — the errors light up",
        color=GREY, fontsize=15, ha="center", va="center")

# ---- mini legend (below the card) --------------------------------------------
ly = 0.038
ax.scatter([0.250], [ly], s=70, c=ACC, edgecolors="none", zorder=5)
ax.text(0.272, ly, "trusted point", color=GREY, fontsize=13, va="center", ha="left")
ax.scatter([0.470], [ly], s=90, c=RED, edgecolors=WHITE, linewidths=0.7, zorder=5)
ax.text(0.492, ly, "outlier", color=GREY, fontsize=13, va="center", ha="left")
ax.plot([0.610, 0.660], [ly, ly], color=ACC, lw=1.6, alpha=0.6, zorder=5)
ax.text(0.672, ly, "nearest-neighbour graph", color=GREY, fontsize=13, va="center", ha="left")

fig.savefig(PP / "gnn_cover.png", transparent=True, dpi=400, bbox_inches="tight", pad_inches=0.03)
fig.savefig(PP / "gnn_cover.svg", transparent=True, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)

from PIL import Image
f = Image.open(PP / "gnn_cover.png").convert("RGBA"); pad = 40; W_, H_ = f.size
c = Image.new("RGBA", (W_ + 2 * pad, H_ + 2 * pad), (12, 17, 24, 255))
c.alpha_composite(f, (pad, pad)); c.convert("RGB").save(PP / "_prev_gnn_cover.png")
print("nodes", N, "cross-edges", len(cross), "focus deg", len(adj[focus]),
      "outliers", int(is_out.sum()))
print("wrote gnn_cover.svg/png + _prev_gnn_cover.png to", PP, " size", f.size)
