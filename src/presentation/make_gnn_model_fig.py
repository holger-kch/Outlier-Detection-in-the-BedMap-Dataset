#!/usr/bin/env python3
"""The WHOLE GNN model in one figure (forward + training), given x_i and e_ij are known.

EXTENDS the 'Message passing' figure (make_gnn_rf_fig.py) on both sides:
  ADD #1 (left):  three seed classes (OUTLIER red 38,881 / INLIER green 646,674 /
                  unlabeled grey 2,000,000 sample) merge into the box input x_i , e_ij; the unlabeled
                  take a distinct DASHED route (nnPU, handled at the loss).
  ADD #2 (right): a 'Training loss' box — each seed class pushes its point's LOGIT up/down
                  + the one loss = three GROUP MEANS (outlier / inlier / unlabeled) — and a
                  loop back to the box weights W.
Central 'Message passing' box (3-view zoom-cascade receptive field, cumulative nested
braces -> n^(1)/n^(2)/n^(3), m_ij above, z_i below) -> green MLP head -> p_outlier are KEPT.
House style, transparent.

v3 (08/06/2026): user-driven space + clarity pass on top of v2.
  - TALLER: cascade + boxes spread to use the slide's vertical room (less wide overall).
  - Bigger node-explosion views so x_i/x_j/x_k are clearly visible.
  - Less air around the MLP head: bigger green box, blue + orange boxes pulled closer.
  - Seeds -> [x_i;e_ij] feeders are smooth curved arrows again (clear of the chip text).
  - Loss made unambiguous: it is NOT one sum over all points — it is a per-GROUP MEAN
    (angle brackets <.>), three of them, weighted and added. Weights are FIXED constants.
  - Small labels enlarged. Export is cropped to the actual INK (alpha-based), so the file
    carries no transparent margin (v2 fixed the old ~34% baked-in border)."""
import os
ROOT = "/lustre/hpc/icecube/holgerkc/ML-ICE-PROJECT"
os.environ["MPLCONFIGDIR"] = ROOT + "/.mplcache"
os.environ["TMPDIR"] = ROOT + "/.tmp"
os.makedirs(ROOT + "/.mplcache", exist_ok=True)
os.makedirs(ROOT + "/.tmp", exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
from matplotlib.transforms import Bbox
from pathlib import Path

PP = Path(ROOT) / "PP"; PP.mkdir(exist_ok=True)
WHITE = "#FFFFFF"; ACC = "#6FD3FF"; ORANGE = "#FFB74D"; GREY = "#AEB5BE"; DIM = "#9AA0A8"
UNL_GREY = GREY
GRN = "#5AE0A0"; RED = "#FF6B6B"; PINK = "#FF75D8"; PANEL = (0.05, 0.08, 0.12, 0.55)
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})

FW, FH = 28.5, 15.5
fig, ax = plt.subplots(figsize=(FW, FH), dpi=150)
fig.patch.set_alpha(0.0)            # transparent in the LIVE buffer too (so ink-crop works)
ax.set_facecolor("none"); ax.axis("off"); ax.set_aspect("equal")
XMIN = -0.02
XMAX = 1.29
ax.set_xlim(XMIN, XMAX); ax.set_ylim(-0.08, 0.66)

S = 0.560          # scale of the original forward-model content
SHIFT = 0.165      # push the original content this far to the right
LIFT = 0.075       # ...and this far up (frees the bottom band for the loop)
def X(x):
    return x * S + SHIFT
def Y(y):
    return y * S + LIFT

N = 16
rr = 0.078 * S               # neighbour spoke length (bigger views)
Rc = 0.102 * S               # framing circle radius
sr = 0.0140 * S              # small zoom circle radius on the down-neighbour
cxs = 0.110
cys = [0.680, 0.400, 0.120]  # spread vertically to fill the taller box
centers = [(cxs, y) for y in cys]
labels = [r"$x_i$", r"$x_j$", r"$x_k$"]
a = 2 * np.pi * np.arange(N) / N
nb = np.stack([np.cos(a), np.sin(a)], axis=1)
DOWN = 12   # the straight-down neighbour index


def draw_view(C, lab, framed, center_col, zoom_down):
    C = np.array([X(C[0]), Y(C[1])], dtype=float)
    p = C + rr * nb
    for k in range(N):
        col = WHITE if (zoom_down and k == DOWN) else ACC
        ax.plot([C[0], p[k, 0]], [C[1], p[k, 1]], color=col, lw=1.9, alpha=0.9, zorder=2)
        ax.scatter([p[k, 0]], [p[k, 1]], s=50, color=col, edgecolor="k", lw=0.35, zorder=4)
    if framed:
        ax.add_patch(Circle(C, Rc, fill=False, ec=GREY, lw=1.4, alpha=0.42, zorder=1))
    # big central node so the x_i / x_j / x_k label reads clearly
    ax.scatter([C[0]], [C[1]], s=610, color=center_col, edgecolor="k", lw=0.7, zorder=6)
    tcol = "#111" if center_col == WHITE else "#3a2600"
    ax.text(C[0], C[1], lab, color=tcol, fontsize=20, weight="bold",
            ha="center", va="center", zorder=7)
    return C, p


def zoom_cone(node, Cnext):
    node = np.array(node, dtype=float)
    Cnext = np.array([X(Cnext[0]), Y(Cnext[1])], dtype=float)
    ax.add_patch(Circle(node, sr, fill=False, ec=WHITE, lw=1.5, alpha=0.95, zorder=5))
    L = node + np.array([-sr, 0]); R = node + np.array([sr, 0])
    tl = Cnext + Rc * np.array([np.cos(np.radians(112)), np.sin(np.radians(112))])
    tr = Cnext + Rc * np.array([np.cos(np.radians(68)), np.sin(np.radians(68))])
    for s_pt, b_pt in [(L, tl), (R, tr)]:
        ax.plot([s_pt[0], b_pt[0]], [s_pt[1], b_pt[1]], color=WHITE, lw=1.2,
                alpha=0.55, ls=(0, (4, 3)), zorder=3)


def brace_span(ymin, ymax, xpos, width, color, lw=2.2):
    """Right-pointing curly brace over [ymin, ymax]; returns the tip y (the middle).
    ymin/ymax/xpos/width are already in FIGURE coordinates."""
    half = (ymax - ymin) / 2.0
    res = 260
    beta = 230.0 / (ymax - ymin)
    yfull = np.linspace(ymin, ymax, res)
    yh = yfull[:res // 2 + 1]
    xh = 1 / (1. + np.exp(-beta * (yh - ymin))) + 1 / (1. + np.exp(-beta * (yh - ymin - half)))
    x = np.concatenate((xh, xh[-2::-1]))
    x = (x - x.min()) / (x.max() - x.min())
    y = np.linspace(ymin, ymax, x.shape[0])
    ax.plot(xpos + width * x, y, color=color, lw=lw, clip_on=False,
            zorder=4, solid_capstyle="round")
    return ymin + half


# ============================ THE BOX (wraps A-E) ====================================
BX0, BY0, BX1, BY1 = X(0.015), Y(-0.035), X(0.805), Y(0.840)
ax.add_patch(FancyBboxPatch((BX0, BY0), BX1 - BX0, BY1 - BY0,
                            boxstyle="round,pad=0.004,rounding_size=0.012",
                            linewidth=2.2, edgecolor=ACC, facecolor=PANEL, zorder=0))
ax.text(BX0 + 0.014, BY1 + 0.038, "Message passing", color=ACC, fontsize=29,
        weight="bold", ha="left", va="center")


# ---- A) receptive field zoom cascade ----
C1, p1 = draw_view(centers[0], labels[0], False, WHITE, True)
C2, p2 = draw_view(centers[1], labels[1], True, WHITE, True)
C3, p3 = draw_view(centers[2], labels[2], True, WHITE, False)
zoom_cone(p1[DOWN], centers[1])
zoom_cone(p2[DOWN], centers[2])

# ---- C) the three update-rule formulas (placed at the brace tips, see B) ----
FORM_X = X(0.340)
forms = [r"$n_i^{(1)} = \dfrac{1}{16}\sum_j m_{ij}(x_j)\ +\ W_s\,x_i\ \in\mathbb{R}^{384}$",
         r"$n_i^{(2)} = \dfrac{1}{16}\sum_j m_{ij}(n_j^{(1)})\ +\ W_s\,n_i^{(1)}\ \in\mathbb{R}^{384}$",
         r"$n_i^{(3)} = \dfrac{1}{16}\sum_j m_{ij}(n_j^{(2)})\ +\ W_s\,n_i^{(2)}\ \in\mathbb{R}^{192}$"]

# ---- B) cumulative nested braces (figure coords) ----
top = Y(cys[0]) + Rc + 0.012
bottoms = [Y(cys[0]) - Rc - 0.012,
           Y(cys[1]) - Rc - 0.012,
           Y(cys[2]) - Rc - 0.012]
bx = [X(0.210), X(0.250), X(0.290)]
bw = [0.024 * S, 0.027 * S, 0.030 * S]
form_ys = []
for i, (bot, w, form) in enumerate(zip(bottoms, bw, forms)):
    tip = brace_span(bot, top, bx[i], w, ACC, lw=2.3 + 0.4 * i)
    form_ys.append(tip)
    ax.annotate("", xy=(FORM_X - 0.006, tip), xytext=(bx[i] + w, tip),
                arrowprops=dict(arrowstyle="-", color=ACC, lw=1.1, alpha=0.95,
                                shrinkA=0, shrinkB=0), zorder=3)
    ax.text(FORM_X, tip, form, color=WHITE, fontsize=16.6, ha="left", va="center")

# ---- D) m_ij(z) ABOVE the n_i column ----
mij_y = BY1 - 0.028
ax.text(FORM_X, mij_y, r"$m_{ij}(z) = (W_m\,z)\odot\sigma(W_e\,e_{ij})\ \in\mathbb{R}^{384}$",
        color=GREY, fontsize=16.2, ha="left", va="center")

# ---- E) downward arrow from the formula column to z_i ----
lowest_tip = form_ys[2]
ARR_X = FORM_X + 0.075 * S
zi_y = Y(0.082)             # raised so the MLP head clears the bottom loop band
ax.add_patch(FancyArrowPatch((ARR_X, lowest_tip - 0.024), (ARR_X, zi_y + 0.026),
                             arrowstyle="-|>", color=WHITE, lw=2.3,
                             mutation_scale=18, zorder=5))
ax.text(FORM_X, zi_y, r"$z_i = [\,n_i^{(1)}\,;\ n_i^{(2)}\,;\ n_i^{(3)}\,]\ \in\mathbb{R}^{192}$",
        color=ACC, fontsize=20.5, weight="bold", ha="left", va="center")


# ===================== G) box -> MLP head -> p_outlier  (folded VERTICALLY) ==========
Z_Y = zi_y
MCX = BX1 + 0.106                      # centre x of the MLP / p_out column (pulled IN)
MLP_W, MLP_H = 0.152, 0.084            # bigger green box -> less air around it
MY = Z_Y
ax.add_patch(FancyArrowPatch((BX1 - 0.018, MY), (MCX - MLP_W / 2, MY), arrowstyle="-|>",
                             color=WHITE, lw=2.3, mutation_scale=19, zorder=5))
ax.add_patch(FancyBboxPatch((MCX - MLP_W / 2, MY - MLP_H / 2), MLP_W, MLP_H,
                            boxstyle="round,pad=0.004,rounding_size=0.014",
                            linewidth=2.0, edgecolor=PINK, facecolor=PANEL, zorder=4))
ax.text(MCX, MY, "MLP head", color=PINK, fontsize=22, weight="bold",
        ha="center", va="center", zorder=5)
# p_outlier label above the MLP/output arrow
P_Y = MY + MLP_H / 2 + 0.128
PER_Y = P_Y - 0.056
LOGIT_Y = P_Y - 0.090
ax.text(MCX, P_Y, r"$p_{\mathrm{outlier}} = \sigma(\mathrm{logit})$",
        color=PINK, fontsize=17.0, weight="bold", ha="center", va="center", zorder=5)
ax.text(MCX, P_Y - 0.032, r"$\in (0,1)$",
        color=GREY, fontsize=15.5, ha="center", va="center", zorder=5)
ax.text(MCX, PER_Y, r"per-point outlier probability",
        color=GREY, fontsize=15.5, ha="center", va="center", style="italic", zorder=5)
ax.plot([MCX - 0.058, MCX + 0.058], [PER_Y - 0.022, PER_Y - 0.022],
        color=DIM, lw=1.0, alpha=0.5, zorder=5)
ax.text(MCX, LOGIT_Y, r"$\mathrm{logit}=\mathrm{MLP}(z_i)$",
        color=PINK, fontsize=18.5, weight="bold", ha="center", va="center", zorder=5)
ROUTE_X = MCX
O_Y = P_Y + 0.056
ax.add_patch(FancyArrowPatch((MCX, MY + MLP_H / 2), (MCX, LOGIT_Y - 0.026),
                             arrowstyle="-|>", color=WHITE, lw=2.3,
                             mutation_scale=17, zorder=5))
ax.add_patch(FancyArrowPatch((ROUTE_X, P_Y + 0.032), (ROUTE_X, O_Y),
                             arrowstyle="-", color=WHITE, lw=2.3,
                             shrinkA=0, shrinkB=0, zorder=5))

# ===================================================================================
# ADD #1 — SEEDS on the INPUT (far LEFT) -> merge into the box input (x_i , e_ij)
# ===================================================================================
SEED_X = 0.000
CHIP_W, CHIP_H = 0.112, 0.078
dot_r = 0.012
box_mid = (BY0 + BY1) / 2.0
seed_cys = [box_mid + 0.170, box_mid + 0.000, box_mid - 0.170]
seed_defs = [("OUTLIERS", "label = 1", "38,881", RED),
             ("INLIERS", "label = 0", "646,674", GRN),
             ("unlabeled", "label = -1", "2,000,000", UNL_GREY)]
ax.text(SEED_X + CHIP_W / 2, seed_cys[0] + CHIP_H / 2 + 0.044, "seed classes",
        color=WHITE, fontsize=20, weight="bold", ha="center", va="center")
chip_right = []
for (name, lab_code, cnt, col), cy in zip(seed_defs, seed_cys):
    ax.add_patch(FancyBboxPatch((SEED_X, cy - CHIP_H / 2), CHIP_W, CHIP_H,
                                boxstyle="round,pad=0.004,rounding_size=0.010",
                                linewidth=1.9, edgecolor=col, facecolor=PANEL, zorder=4))
    ax.add_patch(Circle((SEED_X + 0.019, cy + 0.014), dot_r, color=col,
                        ec="k", lw=0.4, zorder=6))
    ax.text(SEED_X + 0.038, cy + 0.020, name, color=col, fontsize=15.5,
            weight="bold", ha="left", va="center", zorder=6)
    ax.text(SEED_X + 0.038, cy - 0.003, lab_code, color=WHITE, fontsize=14,
            ha="left", va="center", zorder=6)
    ax.text(SEED_X + 0.038, cy - 0.026, cnt, color=GREY, fontsize=15,
            ha="left", va="center", zorder=6)
    chip_right.append((SEED_X + CHIP_W, cy))

# merge point just left of the box (the model input)
MERGE_X = BX0 - 0.026
BOX_H = 0.054
VEC_W, VEC_H = 0.034, BOX_H
BR_W, BR_H = 0.005, VEC_H - 0.008
ax.add_patch(FancyBboxPatch((MERGE_X - VEC_W / 2, box_mid - BOX_H / 2), VEC_W, BOX_H,
                            boxstyle="round,pad=0.001,rounding_size=0.004",
                            linewidth=1.2, edgecolor=GREY,
                            facecolor=(0.78, 0.80, 0.82, 0.18), zorder=5))
ax.text(MERGE_X, box_mid + 0.013, r"$x_i$",
        color=WHITE, fontsize=18.5, weight="bold", ha="center", va="center", zorder=6)
ax.text(MERGE_X, box_mid - 0.013, r"$e_{ij}$",
        color=WHITE, fontsize=18.5, weight="bold", ha="center", va="center", zorder=6)
for sx in (-1, 1):
    x0 = MERGE_X + sx * 0.013
    x1 = x0 + (-sx) * BR_W
    y0, y1 = box_mid - BR_H / 2, box_mid + BR_H / 2
    ax.plot([x0, x0], [y0, y1], color=WHITE, lw=1.4, zorder=6)
    ax.plot([x0, x1], [y1, y1], color=WHITE, lw=1.4, zorder=6)
    ax.plot([x0, x1], [y0, y0], color=WHITE, lw=1.4, zorder=6)
# three smooth feeder curves: solid OUTLIER+INLIER, DASHED unlabeled (nnPU). Each starts at
# the chip's right edge and the merge box sits to its RIGHT; both curved ones bow OUTWARD
# (convex toward the merge box) so they never re-enter the chip text/border. Per feeder:
# (start_dx, start_dy, end_dy, rad) — the top bows down-right, the bottom up-right.
END_X = MERGE_X - VEC_W / 2 - 0.005
feed = [(0.006,  0.004,  0.014, -0.20),    # OUTLIERS (top) -> bend like unlabeled
        (0.006,  0.000,  0.000,  0.00),    # INLIERS (mid)  -> straight
        (0.008, -0.008, -0.014,  0.20)]    # unlabeled (bot)-> up-right, bow OUT (clear of 'd')
for idx, (rx, cy) in enumerate(chip_right):
    dashed = (idx == 2)
    col = seed_defs[idx][3]
    sdx, sdy, edy, rd = feed[idx]
    ax.add_patch(FancyArrowPatch((rx + sdx, cy + sdy),
                                 (END_X, box_mid + edy),
                                 arrowstyle="-|>", color=col, lw=2.3,
                                 mutation_scale=16, zorder=3,
                                 linestyle="--" if dashed else "-",
                                 connectionstyle="arc3,rad=%.2f" % rd,
                                 alpha=0.95))
# merge -> box
ax.add_patch(FancyArrowPatch((MERGE_X + VEC_W / 2 + 0.004, box_mid), (BX0, box_mid),
                             arrowstyle="-|>", color=WHITE, lw=2.5,
                             mutation_scale=16, zorder=5))


# ===================================================================================
# ADD #2 — TRAINING LOSS box  (each target pushes the LOGIT; one loss = 3 group means)
# ===================================================================================
fig.canvas.draw()
RND = fig.canvas.get_renderer()


def put_seg(x, y, txt, col, fs, weight="bold", style="normal"):
    """Place a text segment left-aligned at x; return the x just past it (data coords)."""
    t = ax.text(x, y, txt, color=col, fontsize=fs, weight=weight, style=style,
                ha="left", va="center", zorder=6)
    bb = t.get_window_extent(RND).transformed(ax.transData.inverted())
    return bb.x1


LOSS_X0 = MCX + MLP_W / 2 + 0.044       # pulled IN toward the MLP head
LOSS_W = 0.372
LOSS_Y0 = BY0
LOSS_Y1 = BY1
ax.add_patch(FancyBboxPatch((LOSS_X0, LOSS_Y0), LOSS_W, LOSS_Y1 - LOSS_Y0,
                            boxstyle="round,pad=0.004,rounding_size=0.012",
                            linewidth=2.2, edgecolor=ORANGE, facecolor=PANEL, zorder=4))
ax.text(LOSS_X0 + 0.012, LOSS_Y1 + 0.038, "Training loss", color=ORANGE, fontsize=26,
        weight="bold", ha="left", va="center")
ax.add_patch(FancyArrowPatch((ROUTE_X, O_Y), (LOSS_X0, O_Y),
                             arrowstyle="-|>", color=WHITE, lw=2.3,
                             mutation_scale=18, shrinkA=0, shrinkB=0, zorder=5))

LtX = LOSS_X0 + 0.020
RtX = LOSS_X0 + LOSS_W - 0.018
yL = LOSS_Y1 - 0.044
ax.text(LtX, yL, "weighted cross-entropy + extra unlabeled term:", color=GREY, fontsize=16.5,
        ha="left", va="center", style="italic", zorder=6)
rows = [(RED,  "outlier  ·  target = 1   →   logit ↑"),
        (GRN,  "inlier  ·  target = 0   →   logit ↓"),
        (GREY, "unlabeled  ·  target = −1   →   logit ↓")]
yL -= 0.058
for col, txt in rows:
    ax.add_patch(Circle((LtX + 0.012, yL), 0.0125, color=col, ec="k", lw=0.3, zorder=6))
    ax.text(LtX + 0.038, yL, txt, color=col, fontsize=17.5,
            weight="bold", ha="left", va="center", zorder=6)
    yL -= 0.054

# separator
yL += 0.006
ax.plot([LtX, RtX], [yL, yL], color=DIM, lw=1.0, alpha=0.5, zorder=5)
# the two per-logit penalties (define l+ / l-)
yL -= 0.038
ax.text(LtX, yL,
        r"$\ell^{+}=\ln(1+e^{-\mathrm{logit}})$"
        r"$\qquad\ell^{-}=\ln(1+e^{+\mathrm{logit}})$",
        color=GREY, fontsize=14.8, ha="left", va="center", zorder=6)
# the ONE loss = three GROUP MEANS  <.>  (NOT a single sum over all points)
yL -= 0.048
xx = put_seg(LtX, yL, r"$\mathcal{L}\; =\;$", WHITE, 18)
xx = put_seg(xx, yL, r"$w_{\mathrm{out}}\langle\ell^{+}\rangle_{\,\mathrm{out}}$", RED, 18)
xx = put_seg(xx, yL, r"$\;+\;$", WHITE, 18)
xx = put_seg(xx, yL, r"$w_{\mathrm{in}}\langle\ell^{-}\rangle_{\,\mathrm{in}}$", GRN, 18)
xx = put_seg(LtX + 0.040, yL - 0.042, r"$+\;$", WHITE, 18)
xx = put_seg(xx, yL - 0.042,
             r"$w_{\mathrm{unl}}\max(0,\langle\ell^{-}\rangle_{\,\mathrm{unl}}"
             r"-\pi\langle\ell^{-}\rangle_{\,\mathrm{out}})$",
             GREY, 18)
ax.plot([LtX, RtX], [yL - 0.070, yL - 0.070], color=DIM, lw=1.0, alpha=0.5, zorder=5)
# footer: the aggregation, then the Optuna weights
yL -= 0.084
ax.text(LtX, yL,
        r"$\langle\,\cdot\,\rangle$ = mean over the group's points",
        color=DIM, fontsize=14.5, ha="left", va="center", style="italic", zorder=6)
yL -= 0.026
ax.text(LtX, yL,
        r"Optuna weights:  $w_{\mathrm{out}}{=}3.92,\ w_{\mathrm{in}}{=}0.996$",
        color=DIM, fontsize=13.8, ha="left", va="center", style="italic", zorder=6)
yL -= 0.022
ax.text(LtX, yL,
        r"$w_{\mathrm{unl}}{=}1,\ \pi{=}0.0021$",
        color=DIM, fontsize=13.8, ha="left", va="center", style="italic", zorder=6)

# ---- the training loop: loss -> down -> back along the BOTTOM band -> box weights W ----
LOOP_Y = BY0 - 0.070           # dropped so the loop label clears the MLP-head box bottom
W_X = BX0 + 0.105
loss_cx = LOSS_X0 + LOSS_W / 2.0
ax.add_patch(FancyArrowPatch((loss_cx, LOSS_Y0), (loss_cx, LOOP_Y),
                             arrowstyle="-", color=ORANGE, lw=2.7, shrinkA=0, shrinkB=0,
                             zorder=3))
ax.add_patch(FancyArrowPatch((loss_cx, LOOP_Y), (W_X, LOOP_Y),
                             arrowstyle="-", color=ORANGE, lw=2.7, shrinkA=0, shrinkB=0,
                             zorder=3))
ax.add_patch(FancyArrowPatch((W_X, LOOP_Y), (W_X, BY0 - 0.002),
                             arrowstyle="-|>", color=ORANGE, lw=2.7,
                             mutation_scale=21, shrinkA=0, shrinkB=0, zorder=3))
ax.text((loss_cx + W_X) / 2.0, LOOP_Y + 0.026,
        r"gradient descent  ·  update $W$  ·  repeat",
        color=ORANGE, fontsize=20, weight="bold", ha="center", va="center", zorder=6)
ax.text(W_X + 0.014, BY0 - 0.026, r"$W:\ W_m,\ W_e,\ W_s$", color=ORANGE,
        fontsize=18, weight="bold", ha="left", va="center", zorder=6)


# ===================================================================================
# SAVE — crop to the actual INK (alpha channel), so the file has no transparent margin.
# ===================================================================================
def ink_bbox_inches(pad_in=0.07):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())          # (H, W, 4), row 0 = TOP
    alpha = buf[:, :, 3]
    ys, xs = np.where(alpha > 6)
    Hpx, Wpx = alpha.shape
    dpi = fig.dpi
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    bx0 = x0 / dpi - pad_in
    bx1 = x1 / dpi + pad_in
    by0 = (Hpx - y1) / dpi - pad_in
    by1 = (Hpx - y0) / dpi + pad_in
    return Bbox.from_extents(bx0, by0, bx1, by1)


bb_in = ink_bbox_inches()
fig.savefig(PP / "gnn_model.png", transparent=True, dpi=400, bbox_inches=bb_in, pad_inches=0.0)
fig.savefig(PP / "gnn_model.svg", transparent=True, bbox_inches=bb_in, pad_inches=0.0)
plt.close(fig)

from PIL import Image
f = Image.open(PP / "gnn_model.png").convert("RGBA"); pad = 40; W, H = f.size
cc = Image.new("RGBA", (W + 2 * pad, H + 2 * pad), (12, 17, 24, 255))
cc.alpha_composite(f, (pad, pad)); cc.convert("RGB").save(PP / "_prev_gnn_model.png")
print("wrote gnn_model.svg/png + _prev_gnn_model.png to", PP, "  size", f.size,
      "  aspect", round(W / H, 3))
