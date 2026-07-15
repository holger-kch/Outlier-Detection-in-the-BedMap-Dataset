#!/usr/bin/env python3
"""Dark, transparent 'edge table' figure for the k-NN-16 table slide (1.2B x 4),
   with a footer showing how log1p_dist and signed_grad are computed."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PP = Path(__file__).resolve().parents[1] / "PP"; PP.mkdir(exist_ok=True)
WHITE = "#FFFFFF"; GREY = "#C8CCD2"; DIM = "#9AA0A8"; ACC = "#6FD3FF"; ORANGE = "#FFB74D"
plt.rcParams.update({"font.family": "DejaVu Sans"})

fig, ax = plt.subplots(figsize=(11.6, 7.6), dpi=150)
ax.set_facecolor("none"); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

cx = {"node": 0.12, "neighbour": 0.34, "log1p_dist": 0.62, "signed_grad": 0.86}

# group labels + who|what divider
ax.text(0.23, 0.975, "the link (who)", color=DIM, fontsize=12, ha="center", style="italic")
ax.text(0.74, 0.975, "what's on the link", color=DIM, fontsize=12, ha="center", style="italic")
ax.plot([0.475, 0.475], [0.36, 0.93], color=GREY, lw=0.8, alpha=0.35)

# headers
hy = 0.905
ax.text(cx["node"], hy, "node", color=ACC, fontsize=15, ha="center", weight="bold")
ax.text(cx["neighbour"], hy, "neighbour", color=ACC, fontsize=15, ha="center", weight="bold")
ax.text(cx["log1p_dist"], hy, "log1p_dist", color=ACC, fontsize=15, ha="center", weight="bold")
ax.text(cx["signed_grad"], hy, "signed_grad", color=ACC, fontsize=15, ha="center", weight="bold")
ax.plot([0.04, 0.96], [hy - 0.045, hy - 0.045], color=GREY, lw=1.0, alpha=0.5)

rows = [
    ("0", "1", "5.5234", "−1.5049", False),
    ("0", "2", "6.0859", "−0.8936", False),
    "DOTS",                                       # node 0 has 16 neighbours — rows 3..13 omitted
    ("0", "73319443", "7.5664", "+3.9316", True),
    ("0", "73319442", "7.7227", "+2.7773", True),
    ("1", "2", "5.2500", "−0.0538", False),
    ("1", "0", "5.5234", "+1.5049", False),
]
y = hy - 0.115
for row in rows:
    if row == "DOTS":
        for k in cx.values():
            ax.text(k, y, "⋮", color=GREY, fontsize=15, ha="center")
        y -= 0.066
        continue
    node, nbr, ld, g, cross = row
    col = ORANGE if cross else WHITE
    ax.text(cx["node"], y, node, color=GREY, fontsize=14, ha="center")
    ax.text(cx["neighbour"], y, nbr, color=col, fontsize=14, ha="center")
    ax.text(cx["log1p_dist"], y, ld, color=col, fontsize=14, ha="center")
    ax.text(cx["signed_grad"], y, g, color=col, fontsize=14, ha="center")
    y -= 0.066

for k in cx.values():
    ax.text(k, y + 0.010, "⋮", color=GREY, fontsize=16, ha="center")
ax.text(0.5, 0.300, "1,195,952,496 rows   =   74,747,031 points  ×  16 neighbours",
        color=WHITE, fontsize=13, ha="center", weight="bold")
ax.text(0.5, 0.252, "orange = neighbour from another survey (cross-track) → here the surveys disagree",
        color=ORANGE, fontsize=11, ha="center", style="italic")

# footer: how the two features are computed (same symbols as the rest of the deck: x, y, d, ΔH)
ax.plot([0.04, 0.96], [0.205, 0.205], color=GREY, lw=1.0, alpha=0.4)
ax.text(0.06, 0.165, "how the two features are computed:", color=DIM, fontsize=12, ha="left", style="italic")
ax.text(0.06, 0.115, "log1p_dist", color=ACC, fontsize=13.5, weight="bold", ha="left")
ax.text(0.27, 0.115, r"$=\ \ln(1+d),\quad d=\sqrt{\Delta x^2+\Delta y^2}$",
        color=WHITE, fontsize=13.5, ha="left")
ax.text(0.06, 0.055, "signed_grad", color=ACC, fontsize=13.5, weight="bold", ha="left")
ax.text(0.27, 0.055, r"$=\ \mathrm{sign}(\Delta H/d)\,\ln(1+|\Delta H/d|)\,/\,\mathrm{scale}$",
        color=WHITE, fontsize=13.5, ha="left")

fig.savefig(PP / "edge_table.png", transparent=True, dpi=400, bbox_inches="tight")
fig.savefig(PP / "edge_table.svg", transparent=True, bbox_inches="tight")
plt.close(fig)

from PIL import Image
f = Image.open(PP / "edge_table.png").convert("RGBA"); pad = 40; W, H = f.size
c = Image.new("RGBA", (W + 2 * pad, H + 2 * pad), (12, 17, 24, 255))
c.alpha_composite(f, (pad, pad)); c.convert("RGB").save(PP / "_prev_table.png")
print("wrote edge_table.svg/png", f.size)
