#!/usr/bin/env python3
"""Combined cross-region logit distribution for PhysAE/GNN seed scores."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from physae_cross_region_seed_scores import load_seed_scores


ROOT = Path("/groups/icecube/holgerkc/ML-ICE-PROJECT")
OUTDIR = ROOT / "PP"
OUT_PNG = OUTDIR / "physae_cross_region_logit_distribution.png"
OUT_SVG = OUTDIR / "physae_cross_region_logit_distribution.svg"

N_BINS = 220
SMOOTH_SIGMA = 1.2
LOGIT_70 = float(np.log(0.7 / 0.3))
TITLE_SIZE = 23
LABEL_SIZE = 18
TICK_SIZE = 15
LEGEND_SIZE = 15


def scaled_hist(logits: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    density, _ = np.histogram(logits, bins=edges, density=True)
    density = gaussian_filter1d(density, SMOOTH_SIGMA)
    if density.max() > 0:
        density = density / density.max()
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, density


def main() -> None:
    scores = load_seed_scores()

    out_logits = np.concatenate([
        scores["a_pos_logit"],
        scores["b_pos_logit"],
    ])
    in_logits = np.concatenate([
        scores["a_neg_logit"],
        scores["b_neg_logit"],
    ])

    all_logits = np.concatenate([out_logits, in_logits])
    lo = float(np.floor(all_logits.min()) - 0.5)
    hi = float(np.ceil(all_logits.max()) + 0.5)
    edges = np.linspace(lo, hi, N_BINS + 1)

    x_out, y_out = scaled_hist(out_logits, edges)
    x_in, y_in = scaled_hist(in_logits, edges)

    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    out_color = "#ff6b6b"
    in_color = "#5ae0a0"
    ax.fill_between(x_out, y_out, color=out_color, alpha=0.28, step="mid")
    ax.plot(x_out, y_out, color=out_color, lw=2.8, label="outlier")
    ax.fill_between(x_in, y_in, color=in_color, alpha=0.22, step="mid")
    ax.plot(x_in, y_in, color=in_color, lw=2.8, label="inlier")
    ax.axvline(0, color="white", lw=1.4, ls="--", alpha=0.75, label="logit = 0")
    ax.axvline(LOGIT_70, color="#ffb74d", lw=1.4, ls="--", alpha=0.9, label="p_outlier = 0.7")

    ax.set_title("Cross-region logit distribution", color="white", fontsize=TITLE_SIZE, pad=14)
    ax.set_xlabel("logit", color="white", fontsize=LABEL_SIZE)
    ax.set_ylabel("scaled density", color="white", fontsize=LABEL_SIZE)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1.08)
    ax.tick_params(colors="white", labelsize=TICK_SIZE)
    for spine in ax.spines.values():
        spine.set_color("white")
        spine.set_alpha(0.55)
    ax.grid(color="white", alpha=0.16, lw=0.8)

    leg = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        frameon=False,
        fontsize=LEGEND_SIZE,
        ncol=2,
        borderaxespad=0.0,
    )
    for txt in leg.get_texts():
        txt.set_color("white")

    fig.subplots_adjust(left=0.13, right=0.98, top=0.86, bottom=0.38)
    fig.savefig(OUT_PNG, dpi=250, transparent=True)
    fig.savefig(OUT_SVG, transparent=True)
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_SVG}")
    print(
        "outlier logits: "
        f"n={len(out_logits):,}, median={np.median(out_logits):.3f}, "
        f"p10={np.quantile(out_logits, 0.1):.3f}, p90={np.quantile(out_logits, 0.9):.3f}"
    )
    print(
        "inlier logits: "
        f"n={len(in_logits):,}, median={np.median(in_logits):.3f}, "
        f"p10={np.quantile(in_logits, 0.1):.3f}, p90={np.quantile(in_logits, 0.9):.3f}"
    )


if __name__ == "__main__":
    main()
