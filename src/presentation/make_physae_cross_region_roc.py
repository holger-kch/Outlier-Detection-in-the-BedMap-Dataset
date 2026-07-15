#!/usr/bin/env python3
"""Cross-region ROC curves for the two PhysAE/GNN fold models."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

from physae_cross_region_seed_scores import load_seed_scores


ROOT = Path("/groups/icecube/holgerkc/ML-ICE-PROJECT")
OUTDIR = ROOT / "PP"
OUT_PNG = OUTDIR / "physae_cross_region_roc.png"
OUT_SVG = OUTDIR / "physae_cross_region_roc.svg"
TITLE_SIZE = 23
LABEL_SIZE = 18
TICK_SIZE = 15
LEGEND_SIZE = 14
FIGSIZE = (7.8, 6.4)


def make_curve(pos_score: np.ndarray, neg_score: np.ndarray):
    y = np.concatenate([
        np.ones(len(pos_score), dtype=np.int8),
        np.zeros(len(neg_score), dtype=np.int8),
    ])
    s = np.concatenate([pos_score, neg_score])
    fpr, tpr, _ = roc_curve(y, s)
    return fpr, tpr, float(auc(fpr, tpr)), len(pos_score), len(neg_score)


def main() -> None:
    scores = load_seed_scores()

    curves = [
        (
            "Model B on region A",
            "#6bd6ff",
            *make_curve(scores["a_pos_score"], scores["a_neg_score"]),
        ),
        (
            "Model A on region B",
            "#ffb74d",
            *make_curve(scores["b_pos_score"], scores["b_neg_score"]),
        ),
    ]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    for label, color, fpr, tpr, roc_auc, n_pos, n_neg in curves:
        ax.plot(
            fpr,
            tpr,
            color=color,
            lw=2.6,
            label=f"{label}: AUC={roc_auc:.3f}  ({n_pos:,} out / {n_neg:,} in)",
        )

    ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color="#9aa4b2", alpha=0.75, label="random")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("False positive rate", color="white", fontsize=LABEL_SIZE)
    ax.set_ylabel("True positive rate", color="white", fontsize=LABEL_SIZE)
    ax.set_title(
        "Cross-region ROC (seeds only)",
        color="white",
        fontsize=TITLE_SIZE,
        pad=14,
    )
    ax.tick_params(colors="white", labelsize=TICK_SIZE)
    for spine in ax.spines.values():
        spine.set_color("white")
        spine.set_alpha(0.55)
    ax.grid(color="white", alpha=0.16, lw=0.8)

    leg = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        frameon=True,
        facecolor=(0, 0, 0, 0),
        edgecolor="#9aa4b2",
        fontsize=LEGEND_SIZE,
        borderaxespad=0.0,
    )
    leg.get_frame().set_alpha(0.0)
    for txt in leg.get_texts():
        txt.set_color("white")

    fig.subplots_adjust(left=0.13, right=0.98, top=0.86, bottom=0.34)
    fig.savefig(OUT_PNG, dpi=250, transparent=True)
    fig.savefig(OUT_SVG, transparent=True)
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_SVG}")
    for label, _, _, _, roc_auc, n_pos, n_neg in curves:
        print(f"{label}: AUC={roc_auc:.6f}, positives={n_pos:,}, inliers={n_neg:,}")


if __name__ == "__main__":
    main()
