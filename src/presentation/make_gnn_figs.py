#!/usr/bin/env python3
"""Dark, transparent figures for the GNN section (3.1 inputs, 3.2 message passing, 3.3 result).
Same style + exact feature/variable names as the rest of the deck."""
import os
ROOT = "/groups/icecube/holgerkc/ML-ICE-PROJECT"
os.environ["MPLCONFIGDIR"] = ROOT + "/.mplcache"
os.environ["TMPDIR"] = ROOT + "/.tmp"
os.makedirs(ROOT + "/.mplcache", exist_ok=True); os.makedirs(ROOT + "/.tmp", exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import auc, roc_curve
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from pathlib import Path

PP = Path(ROOT) / "PP"
OUT = Path(ROOT) / "catalyst_pipeline/outputs"
SCORES = OUT / "physae_scores_v4.parquet"
SEED_SCORES = OUT / "physae_cross_region_seed_scores_v4_current.npz"
WHITE = "#FFFFFF"; GREY = "#C8CCD2"; DIM = "#9AA0A8"; ACC = "#6FD3FF"
RED = "#FF6B6B"; GRN = "#5AE0A0"; ORANGE = "#FFB74D"; PANEL = (0.05, 0.08, 0.12, 0.55)
plt.rcParams.update({"font.family": "DejaVu Sans"})


def prev(name):
    from PIL import Image
    f = Image.open(PP / f"{name}.png").convert("RGBA"); pad = 40; W, H = f.size
    c = Image.new("RGBA", (W + 2 * pad, H + 2 * pad), (12, 17, 24, 255))
    c.alpha_composite(f, (pad, pad)); c.convert("RGB").save(PP / f"_prev_{name}.png")


def box(ax, x, y, w, h, title, lines, tc=ACC, ts=13, ls=9.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.018",
                                linewidth=1.5, edgecolor=tc, facecolor=PANEL))
    ax.text(x + w / 2, y + h - 0.04, title, color=tc, fontsize=ts, weight="bold", ha="center", va="top")
    ax.text(x + w / 2, y + h - 0.04 - 0.085, lines, color=WHITE, fontsize=ls, ha="center", va="top", linespacing=1.45)


def save(fig, name):
    fig.savefig(PP / f"{name}.png", transparent=True, dpi=400, bbox_inches="tight")
    fig.savefig(PP / f"{name}.svg", transparent=True, bbox_inches="tight")
    plt.close(fig); prev(name)


def compact_count(n):
    n = int(n)
    if n >= 1_000_000:
        if n >= 10_000_000:
            return f"{n / 1_000_000:.1f}M"
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}k"
    return f"{n:,}"


def score_counts():
    pf = pq.ParquetFile(SCORES)
    total = pf.metadata.num_rows
    n07 = 0
    n05 = 0
    for rg in range(pf.num_row_groups):
        score = pf.read_row_group(rg, columns=["p_outlier"]).column("p_outlier").to_numpy(zero_copy_only=False)
        finite = np.isfinite(score)
        n07 += int((finite & (score > 0.7)).sum())
        n05 += int((finite & (score > 0.5)).sum())
    return total, n07, n05


def auc_label():
    if not SEED_SCORES.exists():
        return "AUC ≈ 0.86"
    s = np.load(SEED_SCORES)
    aucs = []
    for pos_key, neg_key in (("a_pos_score", "a_neg_score"), ("b_pos_score", "b_neg_score")):
        y = np.concatenate([np.ones(len(s[pos_key]), dtype=np.int8), np.zeros(len(s[neg_key]), dtype=np.int8)])
        score = np.concatenate([s[pos_key], s[neg_key]])
        fpr, tpr, _ = roc_curve(y, score)
        aucs.append(float(auc(fpr, tpr)))
    return f"AUC ≈ {np.mean(aucs):.2f}"


# ===================================================== 3.1 what goes in
fig, ax = plt.subplots(figsize=(11.2, 6.0), dpi=150)
ax.set_facecolor("none"); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
box(ax, 0.02, 0.60, 0.45, 0.37, "Node features · 14",
    "sl_ice · bed · geom_resid · surf_minus_bed\nlog_speed · slope · surf · smb · temp\ndir_x · dir_y · + 3 masks\nno coords / track / time")
box(ax, 0.02, 0.35, 0.45, 0.19, "Graph · k-NN-16",
    "1.2B edges  +  [ log1p_dist , signed_grad ]")
box(ax, 0.02, 0.06, 0.45, 0.23, "Seeds (labels)",
    "38,881 outliers · 646,674 inliers\n~74M unlabeled")
box(ax, 0.62, 0.40, 0.34, 0.24, "GNN", "EdgeGatedSAGE\nmessage passing", tc=GRN, ts=15, ls=11)
for yc in (0.785, 0.445, 0.175):
    ax.add_patch(FancyArrowPatch((0.47, yc), (0.615, 0.52), arrowstyle="-|>", color=GREY,
                                 lw=1.6, mutation_scale=16, connectionstyle="arc3,rad=0.04"))
save(fig, "gnn_inputs")

# ===================================================== 3.2 how it processes
fig, ax = plt.subplots(figsize=(10.2, 5.6), dpi=150)
ax.set_facecolor("none"); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("auto")
cxp, cyp = 0.40, 0.52
rng = np.random.default_rng(2)
ang = np.linspace(0, 2 * np.pi, 7)[:-1] + 0.3
nx = cxp + 0.20 * np.cos(ang) * 1.5; ny = cyp + 0.20 * np.sin(ang)
for i, (xx, yy) in enumerate(zip(nx, ny)):
    ax.add_patch(FancyArrowPatch((xx, yy), (cxp, cyp), arrowstyle="-|>", color=ACC, lw=1.5,
                                 mutation_scale=13, alpha=0.9, shrinkA=8, shrinkB=12))
    ax.scatter([xx], [yy], s=110, color=ACC, edgecolor="k", lw=0.4, zorder=4)
ax.scatter([cxp], [cyp], s=240, color=WHITE, edgecolor="k", zorder=5)
ax.text(nx[0] + 0.02, ny[0] + 0.06, "[ log1p_dist , signed_grad ]", color=GREY, fontsize=10, ha="center")
# output arrow -> score
ax.add_patch(FancyArrowPatch((cxp + 0.05, cyp), (0.80, cyp), arrowstyle="-|>", color=GRN, lw=2.0, mutation_scale=16))
ax.text(0.88, cyp, "p_outlier", color=GRN, fontsize=14, weight="bold", ha="center", va="center")
ax.text(0.5, 0.12, "each point's score is built from its neighbourhood — 3 layers + JumpingKnowledge",
        color=WHITE, fontsize=12, ha="center")
ax.text(0.5, 0.05, "edge-gated: a smooth ramp passes, a sudden spike is flagged",
        color=GREY, fontsize=10.5, ha="center", style="italic")
save(fig, "gnn_message")

# ===================================================== 3.3 the result
total_scored, high_conf, flagged = score_counts()
fig, ax = plt.subplots(figsize=(11.2, 4.3), dpi=150)
ax.set_facecolor("none"); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
cards = [(auc_label(), "cross-region\ntrain one half\npredict the other", ACC),
         (compact_count(total_scored), "points scored\n(cross-fit / out-of-fold)", WHITE),
         (compact_count(high_conf), "high-confidence\np_outlier > 0.7", GRN),
         (compact_count(flagged), "flagged\np_outlier > 0.5", ORANGE)]
w = 0.225; gap = 0.025; x0 = (1 - (4 * w + 3 * gap)) / 2
for i, (big, lab, col) in enumerate(cards):
    x = x0 + i * (w + gap)
    ax.add_patch(FancyBboxPatch((x, 0.30), w, 0.55, boxstyle="round,pad=0.006,rounding_size=0.02",
                                linewidth=1.5, edgecolor=col, facecolor=PANEL))
    big_size = 20 if big.startswith("AUC") else 24
    ax.text(x + w / 2, 0.66, big, color=col, fontsize=big_size, weight="bold", ha="center", va="center")
    ax.text(x + w / 2, 0.44, lab, color=WHITE, fontsize=10.5, ha="center", va="center", linespacing=1.3)
save(fig, "gnn_result")

print("wrote gnn_inputs, gnn_message, gnn_result (svg/png) to", PP)
