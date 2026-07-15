#!/usr/bin/env python3
"""Dark conceptual figures for the Pseudo-labels slides (1.1 / 1.2 / 1.3).
Pure illustration (no data read) — matches the dark deck. Saves PNGs into presentation/figs/.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Wedge
from pathlib import Path

OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(exist_ok=True)
PP = Path(__file__).resolve().parents[1] / "PP"      # ML-ICE-PROJECT/PP
PP.mkdir(exist_ok=True)
rng = np.random.default_rng(7)

PANEL = "#0E1620"; WHITE = "#FFFFFF"; GREY = "#C8CCD2"
ACC = "#6FD3FF"; RED = "#FF6B6B"; GRN = "#5AE0A0"; ORANGE = "#FFB74D"; PURPLE = "#B79CFF"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": WHITE,
                     "axes.edgecolor": GREY, "xtick.color": GREY, "ytick.color": GREY,
                     "axes.labelcolor": GREY})


def _fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(PANEL); ax.set_facecolor(PANEL)
    return fig, ax


# ---------------------------------------------------------------- 1.1 double hit (TRANSPARENT bg)
fig, ax = plt.subplots(figsize=(6.3, 4.8), dpi=150)
ax.set_facecolor("none")
# two crossing survey tracks (lines of measurement points)
t = np.linspace(0.08, 0.92, 14)
ax.plot(t, 0.30 + 0.55 * t, "-o", color=ACC, ms=5, lw=1.8, label="track A")    # blue: bottom-left -> top-right
ax.plot(t, 0.92 - 0.62 * t, "-o", color=ORANGE, ms=5, lw=1.8, label="track B")  # orange: top-left -> bottom-right
cx, cy = 0.52, 0.585
ax.add_patch(Circle((cx, cy), 0.055, fill=False, color=WHITE, lw=1.8, ls="--"))
ax.annotate("two tracks within 25 m\n= double hit", (cx, cy), (0.55, 0.16),
            color=WHITE, fontsize=11, ha="center",
            arrowprops=dict(arrowstyle="->", color=WHITE))
# thickness call-outs in the CLEAR space above each track's arm (no line overlap)
ax.text(0.70, 0.88, "h$_A$ = 1850 m", color=ACC, fontsize=12, weight="bold", ha="center")    # track A (blue) is the right arm
ax.text(0.30, 0.88, "h$_B$ = 520 m", color=ORANGE, fontsize=12, weight="bold", ha="center")  # track B (orange) is the left arm
ax.text(0.5, 0.97, "disagree  →  one is wrong  →  OUTLIER", color=RED, fontsize=12,
        ha="center", weight="bold")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.04); ax.axis("off")
ax.legend(loc="lower right", facecolor="none", edgecolor=GREY, labelcolor=WHITE, fontsize=10)
ax.set_title("A crossover measured twice, independently", color=WHITE, fontsize=13, pad=8)
fig.tight_layout()
fig.savefig(OUT / "fig_doublehit.png", transparent=True, bbox_inches="tight")
fig.savefig(PP / "double_hit.png", transparent=True, dpi=400, bbox_inches="tight")  # hi-res transparent PNG fallback
fig.savefig(PP / "double_hit.svg", transparent=True, bbox_inches="tight")           # vector -> insert in PowerPoint
plt.close(fig)

# ---------------------------------------------------------------- 1.2 support
fig, ax = _fig(6.3, 4.8)
ax.add_patch(Circle((0.5, 0.5), 0.42, fill=False, color=WHITE, lw=1.6, ls="--"))
ax.text(0.5, 0.95, "2 km support", color=WHITE, fontsize=12, ha="center", weight="bold")
# support points from several tracks (colour = track)
clrs = [ACC, ORANGE, GRN, PURPLE, "#FF9FB0"]
for k, c in enumerate(clrs):
    ang = rng.uniform(0, 2 * np.pi, 18) + k
    rad = rng.uniform(0.05, 0.40, 18)
    ax.scatter(0.5 + rad * np.cos(ang), 0.5 + rad * np.sin(ang), s=22, color=c, alpha=0.9)
# the candidate pair (centre)
ax.scatter([0.485, 0.515], [0.5, 0.5], s=90, color=WHITE, edgecolor="k", zorder=5)
ax.text(0.5, 0.42, "candidate pair", color=WHITE, fontsize=10, ha="center")
# spikes removed by cleaning
sx, sy = [0.30, 0.74, 0.62], [0.78, 0.30, 0.72]
ax.scatter(sx, sy, s=80, marker="x", color=RED, lw=2.2, zorder=6)
ax.text(0.5, 0.03, "≥5 tracks · ≥2 surveys · ≥100 pts   ·   drop >200 m spikes (×)",
        color=GREY, fontsize=10.5, ha="center")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.0); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("Step 1 · gather & clean the local support", color=WHITE, fontsize=13, pad=8)
fig.tight_layout()
fig.savefig(OUT / "fig_support.png", facecolor=PANEL, bbox_inches="tight")
fig.savefig(PP / "support.svg", facecolor=PANEL, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 1.3 cone verdict (2 panels, TRANSPARENT)
fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 5.4), dpi=150)
for a in (axL, axR):
    a.set_facecolor("none")

# left: octant surround (UNCHANGED, just transparent)
axL.set_aspect("equal"); axL.axis("off")
for k in range(8):
    axL.add_patch(Wedge((0, 0), 1.0, k * 45, (k + 1) * 45, fill=False, edgecolor=GREY, lw=0.9))
for k in range(8):
    a0 = np.deg2rad(k * 45 + 22.5); r = rng.uniform(0.45, 0.9, 2)
    aa = a0 + rng.uniform(-0.25, 0.25, 2)
    axL.scatter(r * np.cos(aa), r * np.sin(aa), s=26, color=ACC, alpha=0.9)
axL.scatter([0], [0], s=130, color=WHITE, edgecolor="k", zorder=5)
axL.set_xlim(-1.05, 1.05); axL.set_ylim(-1.05, 1.05)
axL.set_title("surrounded?\n(≥1 support in all 8 octants)", color=WHITE, fontsize=14)

# right: the band test — ONLY inlier + outlier + the candidate dot at (0,0)
d = np.linspace(0, 1, 100); band = 0.45 * d + 0.12
axR.fill_between(d, -band, band, color=ACC, alpha=0.18, label="support band")
axR.plot(d, band, color=ACC, lw=1.7); axR.plot(d, -band, color=ACC, lw=1.7)
axR.scatter([0], [0], s=130, color=ACC, edgecolor="k", lw=0.6, zorder=6)            # candidate at ΔH=0 (blue)
axR.scatter([0.46], [0.16], s=150, color=GRN, edgecolor="k", lw=0.6, zorder=5, label="inside → inlier")
axR.scatter([0.55], [0.80], s=150, color=RED, edgecolor="k", lw=0.6, zorder=5, label="outside → outlier")
axR.set_xlabel("distance d", fontsize=14); axR.set_ylabel("ΔH", fontsize=14)
axR.set_xlim(0, 1); axR.set_ylim(-1, 1)
axR.tick_params(colors=GREY, labelsize=12)
axR.set_title("within the band?\n|ΔH| ≤ s_max·d + 2η", color=WHITE, fontsize=14)
axR.legend(loc="lower right", facecolor="none", edgecolor=GREY, labelcolor=WHITE, fontsize=11)
for sp in axR.spines.values():
    sp.set_color(GREY)
fig.tight_layout()
fig.savefig(OUT / "fig_cone.png", transparent=True, bbox_inches="tight")
fig.savefig(PP / "cone.png", transparent=True, dpi=400, bbox_inches="tight")
fig.savefig(PP / "cone.svg", transparent=True, bbox_inches="tight")
plt.close(fig)

# ----------------------------------------- Required support qualifications (TRANSPARENT, double-hit style)
fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.8, 4.8), dpi=150,
                               gridspec_kw={"width_ratios": [1.0, 1.15]})
for a in (axL, axR):
    a.set_facecolor("none")

# left: the 2 km neighbourhood (true circle), densely sampled by several tracks
axL.set_aspect("equal"); axL.axis("off"); axL.set_xlim(-1.3, 1.3); axL.set_ylim(-1.45, 1.35)
axL.add_patch(Circle((0, 0), 1.0, fill=False, color=WHITE, lw=1.6, ls="--"))
axL.text(0, 1.18, "2 km", color=WHITE, fontsize=12, ha="center", weight="bold")
tcolors = [ACC, ORANGE, GRN, PURPLE, "#FF9FB0", "#FFD166"]
for k, c in enumerate(tcolors):
    n = 22
    ang = rng.uniform(0, 2 * np.pi, n) + k * 1.3
    rad = np.sqrt(rng.uniform(0.10, 0.95, n))
    axL.scatter(rad * np.cos(ang), rad * np.sin(ang), s=15, color=c, alpha=0.9, zorder=2)
axL.scatter([-0.05, 0.05], [0, 0], s=70, color=WHITE, edgecolor="k", zorder=5)
axL.annotate("candidate pair", (0, 0), (0, -1.24), color=WHITE, fontsize=9,
             ha="center", arrowprops=dict(arrowstyle="->", color=WHITE))

# right: the qualification checklist
axR.axis("off"); axR.set_xlim(0, 1); axR.set_ylim(0, 1)
axR.text(0.0, 0.93, "Support qualifies only if ALL hold:", color=ACC, fontsize=13, weight="bold")
for n, it in enumerate(["within 2 km of the pair", "≥ 100 support points",
                        "≥ 5 independent tracks", "≥ 2 surveys (files)"]):
    yy = 0.72 - n * 0.145
    axR.text(0.0, yy, "✓", color=GRN, fontsize=16, weight="bold", va="center")
    axR.text(0.06, yy, it, color=WHITE, fontsize=13, va="center")
axR.text(0.0, 0.04, "re-checked after cleaning  ·  else → pair discarded", color=GREY, fontsize=9.5)

fig.tight_layout()
fig.savefig(OUT / "fig_support_qual.png", transparent=True, bbox_inches="tight")
fig.savefig(PP / "support_qualifications.png", transparent=True, dpi=400, bbox_inches="tight")
fig.savefig(PP / "support_qualifications.svg", transparent=True, bbox_inches="tight")
plt.close(fig)

# ----------------------------------------- The slope relation  |dH| <= s_max d + 2 eta  (TRANSPARENT)
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 5.4), dpi=150,
                               gridspec_kw={"width_ratios": [1.0, 1.12]})
for a in (axA, axB):
    a.set_facecolor("none"); a.axis("off"); a.set_xlim(0, 1); a.set_ylim(0, 1)
fig.subplots_adjust(top=0.80, bottom=0.12, left=0.03, right=0.99, wspace=0.10)
fig.text(0.5, 0.95, r"$|\Delta H|\ \leq\ s_{max}\,d\ +\ 2\eta$",
         color=WHITE, fontsize=23, ha="center", va="top", weight="bold")


def _caxes(ax, xlabel, ylabel, ox=0.13, oy=0.14):
    ax.annotate("", (0.97, oy), (ox, oy), arrowprops=dict(arrowstyle="->", color=GREY, lw=1.4))
    ax.annotate("", (ox, 0.95), (ox, oy), arrowprops=dict(arrowstyle="->", color=GREY, lw=1.4))
    ax.text(0.97, oy - 0.07, xlabel, color=GREY, fontsize=9.5, ha="right")
    ax.text(ox - 0.05, 0.55, ylabel, color=GREY, fontsize=10, ha="center", va="center", rotation=90)
    return ox, oy

# --- Panel A: deriving s_max from the support (cross-section) ---
ox, oy = _caxes(axA, "horizontal distance", "ice depth")
xs = np.linspace(0.24, 0.86, 11)
ys = oy + 0.10 + 0.52 * (xs - 0.24) / 0.62 + rng.normal(0, 0.028, xs.size)
axA.scatter(xs, ys, s=26, color=GREY, alpha=0.85, zorder=2)
imin, imax = int(np.argmin(ys)), int(np.argmax(ys))
axA.scatter([xs[imin], xs[imax]], [ys[imin], ys[imax]], s=85, color=ORANGE, edgecolor="k", zorder=5)
axA.text(xs[imin] - 0.02, ys[imin] - 0.08, r"$P_{min}$", color=ORANGE, fontsize=11, ha="center")
axA.text(xs[imax], ys[imax] + 0.05, r"$P_{max}$", color=ORANGE, fontsize=11, ha="center")
axA.plot([xs[imin], xs[imax]], [ys[imin], ys[imin]], ls="--", color=WHITE, lw=1.2)   # d_max leg
axA.plot([xs[imax], xs[imax]], [ys[imin], ys[imax]], ls="--", color=WHITE, lw=1.2)   # h_max leg
axA.plot([xs[imin], xs[imax]], [ys[imin], ys[imax]], color=ORANGE, lw=1.7)           # the slope
axA.text((xs[imin] + xs[imax]) / 2, ys[imin] + 0.025, r"$d_{max}$", color=WHITE, fontsize=11, ha="center", va="bottom")
axA.text(xs[imax] + 0.035, (ys[imin] + ys[imax]) / 2, r"$h_{max}$", color=WHITE, fontsize=11, va="center")
axA.set_title("1 · the support's steepest slope\n" r"$s_{max}=2\,h_{max}/d_{max}$",
              color=ACC, fontsize=12.5, pad=6)

# --- Panel B: the band test for ONE point vs its 4 nearest neighbours ---
ox, oy = _caxes(axB, r"$d$   (distance to neighbour)", r"$|\Delta H|$")
y0 = oy + 0.17; x1, yend = 0.95, 0.84
axB.fill([ox, x1, x1, ox], [y0, yend, oy, oy], color=ACC, alpha=0.16, zorder=1)        # allowed band
axB.plot([ox, x1], [y0, yend], color=ACC, lw=1.9, zorder=3)                            # |dH| = s_max d + 2eta
axB.annotate("", (ox, y0), (ox, oy), arrowprops=dict(arrowstyle="<->", color=WHITE, lw=1.0))
axB.text(ox - 0.02, (oy + y0) / 2, r"$2\eta$", color=WHITE, fontsize=11, ha="right", va="center")
axB.text(0.74, 0.70, r"slope $=s_{max}$", color=ACC, fontsize=10.5, rotation=20, ha="center")


def _band(d):
    return y0 + (yend - y0) * (d - ox) / (x1 - ox)


# the point under test sits at d = 0 (the origin); show ITS 4 nearest neighbours
axB.scatter([ox], [oy], s=85, color=WHITE, edgecolor="k", zorder=6)
axB.text(ox + 0.015, oy - 0.06, "the point", color=WHITE, fontsize=9, ha="left")
nd = np.array([0.30, 0.45, 0.68, 0.85])
inside = np.array([True, False, True, True])                                            # 3 inside, 1 outside
ny = np.where(inside, oy + 0.45 * (_band(nd) - oy), _band(nd) + 0.10)
for xx, yy in zip(nd, ny):
    axB.plot([ox, xx], [oy, yy], ls=":", color=GREY, lw=0.9, zorder=2)                  # point -> each neighbour
axB.scatter(nd[inside], ny[inside], s=60, color=GRN, edgecolor="k", lw=0.5, zorder=5)
axB.scatter(nd[~inside], ny[~inside], s=85, color=RED, edgecolor="k", lw=0.5, zorder=5)
axB.text(nd[~inside][0], ny[~inside][0] + 0.05, "1 of the 4 is outside\n→ point dropped",
         color=RED, fontsize=10, ha="center", va="bottom")
axB.text(0.66, oy + 0.04, "the other 3 are inside", color=GRN, fontsize=10, ha="center")
axB.set_title("2 · a point vs its 4 nearest neighbours", color=WHITE, fontsize=12.5, pad=6)

fig.text(0.5, 0.025,
         r"$|\Delta H|$ = ice-depth difference to a neighbour    ·    $d$ = distance to it    ·    "
         r"$\eta=\max(15\,\mathrm{m},\ 0.01\times$ support ice depth$)$",
         color=GREY, fontsize=9.5, ha="center")

fig.savefig(OUT / "fig_support_relation.png", transparent=True, bbox_inches="tight")
fig.savefig(PP / "support_relation.png", transparent=True, dpi=400, bbox_inches="tight")
fig.savefig(PP / "support_relation.svg", transparent=True, bbox_inches="tight")
plt.close(fig)

print("wrote:", *[p.name for p in sorted(OUT.glob('*.png'))])
