#!/usr/bin/env python3
"""Slide 3.1 (intro, BEFORE the big GNN diagram): the node-feature table.
The 14 physics-only node features, written as formulas, with the 8 raw source
columns they are products of MARKED in orange. House style, transparent,
SVG (text as paths) + PNG into PP/. Authoritative list from
catalyst_pipeline/scripts/physae_prepare_v4.py (PHYS_COLS / STD/DIR/MASK)."""
import os
ROOT = "/lustre/hpc/icecube/holgerkc/ML-ICE-PROJECT"
os.environ["MPLCONFIGDIR"] = ROOT + "/.mplcache"
os.environ["TMPDIR"] = ROOT + "/.tmp"
os.makedirs(ROOT + "/.mplcache", exist_ok=True)
os.makedirs(ROOT + "/.tmp", exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

PP = Path(ROOT) / "PP"; PP.mkdir(exist_ok=True)
WHITE = "#FFFFFF"; GREY = "#C8CCD2"; DIM = "#9AA0A8"; ACC = "#6FD3FF"
ORANGE = "#FFB74D"; PANEL = (0.05, 0.08, 0.12, 0.55)
MONO = "DejaVu Sans Mono"; SANS = "DejaVu Sans"
plt.rcParams.update({"font.family": SANS, "svg.fonttype": "path"})

# colour roles for the formula segments
def A(t): return (t, ACC, True, MONO)      # feature name
def O(t): return (t, ORANGE, True, MONO)   # raw source column (MARKED)
def W(t): return (t, WHITE, False, MONO)   # operator / function
def G(t): return (t, DIM, False, MONO)     # " = " / index

# 11 node features, left column (1-6) and right column (7-11)
COL_L = [
    [G("1  "), A("sl_ice"), G(" = "), W("ln(1+"), O("ice_thickness"), W(")")],
    [G("2  "), O("bed_evel")],
    [G("3  "), A("geom_resid"), G(" = "), O("z"), W(" − "), O("bed_evel"), W(" − "), O("ice_thickness")],
    [G("4  "), A("surf_minus_bed"), G(" = "), O("z"), W(" − "), O("bed_evel")],
    [G("5  "), A("log_speed"), G(" = "), W("ln(1+√("), O("vx"), W("²+"), O("vy"), W("²))")],
    [G("6  "), O("s")],
]
COL_R = [
    [G("7  "), O("z")],
    [G("8  "), O("smb")],
    [G("9  "), O("temp")],
    [G("10 "), A("dir_x"), G(" = "), O("vx"), W(" / √("), O("vx"), W("²+"), O("vy"), W("²)")],
    [G("11 "), A("dir_y"), G(" = "), O("vy"), W(" / √("), O("vx"), W("²+"), O("vy"), W("²)")],
]

fig, ax = plt.subplots(figsize=(11.8, 5.7), dpi=150)
fig.patch.set_alpha(0.0)
ax.set_facecolor("none"); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
renderer = fig.canvas.get_renderer()
inv = ax.transData.inverted()


def seg_line(x0, y, segs, fs):
    x = x0
    for txt, col, bold, fam in segs:
        t = ax.text(x, y, txt, color=col, fontsize=fs, family=fam,
                    weight=("bold" if bold else "normal"), ha="left", va="center", zorder=5)
        ext = t.get_window_extent(renderer=renderer)
        (xa, _), (xb, _) = inv.transform((ext.x0, 0)), inv.transform((ext.x1, 0))
        x += (xb - xa)
    return x


# ---- TOP: the 8 raw inputs (marked) -------------------------------------------------
ax.add_patch(FancyBboxPatch((0.0, 0.815), 1.0, 0.175, boxstyle="round,pad=0.004,rounding_size=0.02",
             linewidth=1.6, edgecolor=ORANGE, facecolor=PANEL, zorder=2))
ax.text(0.5, 0.945, "8 raw physics inputs  —  everything below is a product of these",
        color=GREY, fontsize=14, ha="center", va="center", zorder=4)
ax.text(0.5, 0.86, "ice_thickness · bed_evel · z · s · vx · vy · smb · temp",
        color=ORANGE, fontsize=17, weight="bold", family=MONO, ha="center", va="center", zorder=4)

# ---- MAIN: the 11 node features -----------------------------------------------------
ax.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 0.78, boxstyle="round,pad=0.004,rounding_size=0.02",
             linewidth=1.6, edgecolor=ACC, facecolor=PANEL, zorder=1))
ax.text(0.5, 0.735, "11 node features  —  physics only, each built from the marked inputs",
        color=ACC, fontsize=14, weight="bold", ha="center", va="center", zorder=4)

ys = [0.64, 0.535, 0.430, 0.325, 0.220, 0.115]
LX, RX = 0.045, 0.64
maxL = 0.0
for y, segs in zip(ys, COL_L):
    maxL = max(maxL, seg_line(LX, y, segs, 13.5))
maxR = 0.0
for y, segs in zip(ys, COL_R):
    maxR = max(maxR, seg_line(RX, y, segs, 13.5))
print(f"left rows reach x={maxL:.3f} (right col starts {RX}) · right rows reach x={maxR:.3f}")

ax.text(0.5, 0.030, "no coords / track / time / BedMachine",
        color=DIM, fontsize=10.5, ha="center", va="center", style="italic", zorder=4)

fig.savefig(PP / "gnn_features.png", transparent=True, dpi=400, bbox_inches="tight", pad_inches=0.03)
fig.savefig(PP / "gnn_features.svg", transparent=True, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)

from PIL import Image
f = Image.open(PP / "gnn_features.png").convert("RGBA"); pad = 40; W_, H_ = f.size
c = Image.new("RGBA", (W_ + 2 * pad, H_ + 2 * pad), (12, 17, 24, 255))
c.alpha_composite(f, (pad, pad)); c.convert("RGB").save(PP / "_prev_gnn_features.png")
print("wrote gnn_features.svg/png + _prev_gnn_features.png to", PP, " size", f.size)
