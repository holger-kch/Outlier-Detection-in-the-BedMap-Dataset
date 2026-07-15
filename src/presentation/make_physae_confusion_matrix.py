#!/usr/bin/env python3
"""Cross-region confusion matrices for the current PhysAE/GNN best models."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from torch_geometric.data import Data


ROOT = Path("/groups/icecube/holgerkc/ML-ICE-PROJECT")
OUTDIR = ROOT / "PP"
MODEL_DIR = ROOT / "catalyst_pipeline/outputs"
FOLDS = ROOT / "catalyst_pipeline/outputs/physae_cross_region_folds_v4.npz"
OUT_PNG = OUTDIR / "physae_cross_region_confusion_matrix.png"
OUT_SVG = OUTDIR / "physae_cross_region_confusion_matrix.svg"

THRESHOLD = 0.5
EVAL_BATCH = 8192
NUM_WORKERS = 0

BG = "#0b111a"
WHITE = "#f7f8fb"
MUTED = "#aeb6c2"
GRID = "#ffffff"
CYAN = "#6bd6ff"
ORANGE = "#ffb74d"
GREEN = "#5ae0a0"
RED = "#ff6b6b"

sys.path.insert(0, str(ROOT / "catalyst_pipeline/scripts"))
from physae_gnn_v4 import PhysGNN, load_graph, make_loader, roc_auc  # noqa: E402


def load_best_model(fold: str, f_in: int, edge_dim: int, dev: str):
    path = MODEL_DIR / f"physae_best_v4_{fold}.pt"
    st = torch.load(path, map_location=dev)
    cfg = st.get("config", {})
    model = PhysGNN(
        f_in,
        cfg.get("hidden", 384),
        cfg.get("latent", 192),
        cfg.get("layers", 3),
        edge_dim,
        cfg.get("dropout", 0.11136),
    ).to(dev)
    model.load_state_dict(st["ema"] if "ema" in st else st["model"])
    model.eval()
    print(
        f"loaded best fold {fold}: ep {st.get('epoch')} "
        f"val_loss={st.get('best_val_loss', float('nan')):.4f}",
        flush=True,
    )
    return model, st


def score_cross_region_seed_logits(folds) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    fanout = [16, 8, 8]
    t0 = time.time()
    n, f_in, x, ei, ea, _ = load_graph()
    data = Data(x=x, edge_index=ei, edge_attr=ea, y=torch.zeros(n, dtype=torch.int8))
    model_a, meta_a = load_best_model("A", f_in, ea.shape[1], dev)
    model_b, meta_b = load_best_model("B", f_in, ea.shape[1], dev)

    out_a = folds["outliers_A"].astype(np.int64, copy=False)
    in_a = folds["inliers_A"].astype(np.int64, copy=False)
    out_b = folds["outliers_B"].astype(np.int64, copy=False)
    in_b = folds["inliers_B"].astype(np.int64, copy=False)

    rows = np.concatenate([out_a, in_a, out_b, in_b]).astype(np.int64, copy=False)
    # 1 = score with model B, 0 = score with model A.
    target_model = np.concatenate(
        [
            np.ones(len(out_a) + len(in_a), dtype=np.int8),
            np.zeros(len(out_b) + len(in_b), dtype=np.int8),
        ]
    )
    if len(np.unique(rows)) != len(rows):
        raise RuntimeError("seed row ids are not unique")

    sort_idx = np.argsort(rows)
    rows_sorted = rows[sort_idx]
    logits = np.empty(len(rows), dtype=np.float32)

    print(f"building seed NeighborLoader over {len(rows):,} seeds ...", flush=True)
    loader = make_loader(data, rows, fanout, EVAL_BATCH, False, NUM_WORKERS)
    print(f"loader ready; scoring best cross-region models ...", flush=True)

    done = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(dev)
            bs = batch.batch_size
            ids = batch.n_id[:bs].cpu().numpy()
            pos = np.searchsorted(rows_sorted, ids)
            if np.any(pos >= len(rows_sorted)) or np.any(rows_sorted[pos] != ids):
                raise RuntimeError("NeighborLoader returned an input node outside the seed set")
            original_pos = sort_idx[pos]
            target = target_model[original_pos]
            batch_logits = np.empty(bs, dtype=np.float32)

            mask_a = target == 0
            mask_b = target == 1
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                if mask_a.any():
                    logits_a = model_a(batch)[:bs].float().cpu().numpy()
                    batch_logits[mask_a] = logits_a[mask_a]
                if mask_b.any():
                    logits_b = model_b(batch)[:bs].float().cpu().numpy()
                    batch_logits[mask_b] = logits_b[mask_b]
            logits[original_pos] = batch_logits
            done += bs
            if done % (EVAL_BATCH * 20) < EVAL_BATCH:
                print(f"  scored {done:,}/{len(rows):,} seeds ({time.time() - t0:.0f}s)", flush=True)

    n_out_a, n_in_a, n_out_b, n_in_b = map(len, [out_a, in_a, out_b, in_b])
    splits = {
        "A_pos": logits[:n_out_a],
        "A_neg": logits[n_out_a:n_out_a + n_in_a],
        "B_pos": logits[n_out_a + n_in_a:n_out_a + n_in_a + n_out_b],
        "B_neg": logits[n_out_a + n_in_a + n_out_b:],
    }
    meta = {
        "best_A_epoch": int(meta_a.get("epoch", -1)),
        "best_B_epoch": int(meta_b.get("epoch", -1)),
        "best_A_val": float(meta_a.get("best_val_loss", np.nan)),
        "best_B_val": float(meta_b.get("best_val_loss", np.nan)),
    }
    return splits, meta


def confusion_from_logits(pos_logit: np.ndarray, neg_logit: np.ndarray) -> dict[str, int | float]:
    tp = int((pos_logit >= 0.0).sum())
    fn = int((pos_logit < 0.0).sum())
    fp = int((neg_logit >= 0.0).sum())
    tn = int((neg_logit < 0.0).sum())
    y = np.concatenate([np.ones(len(pos_logit), dtype=np.int8), np.zeros(len(neg_logit), dtype=np.int8)])
    s = np.concatenate([pos_logit, neg_logit])
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "auc": roc_auc(y, s)}


def draw_matrix(ax, title: str, counts: dict[str, int], accent: str) -> None:
    values = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])
    row_totals = values.sum(axis=1, keepdims=True)
    row_pct = values / row_totals * 100.0
    labels = np.array([["TN", "FP"], ["FN", "TP"]])
    fills = np.array([[GREEN, RED], [ORANGE, CYAN]])

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_facecolor((0, 0, 0, 0))

    for r in range(2):
        for c in range(2):
            is_error = labels[r, c] in {"FP", "FN"}
            rect = Rectangle(
                (c, r),
                1,
                1,
                facecolor=fills[r, c],
                alpha=0.18 if is_error else 0.24,
                edgecolor=fills[r, c],
                linewidth=2.2,
            )
            ax.add_patch(rect)
            ax.text(
                c + 0.5,
                r + 0.28,
                labels[r, c],
                ha="center",
                va="center",
                color=fills[r, c],
                fontsize=22,
                fontweight="bold",
            )
            ax.text(
                c + 0.5,
                r + 0.53,
                f"{values[r, c]:,}",
                ha="center",
                va="center",
                color=WHITE,
                fontsize=20,
                fontweight="bold",
            )
            ax.text(
                c + 0.5,
                r + 0.75,
                f"{row_pct[r, c]:.1f}% of row",
                ha="center",
                va="center",
                color=MUTED,
                fontsize=12.0,
            )

    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["pred inlier", "pred outlier"], color=WHITE, fontsize=12.5)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", pad=8, length=0)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["true inlier", "true outlier"], color=WHITE, fontsize=12.5)
    ax.tick_params(axis="y", pad=8, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(title, color=accent, fontsize=17, fontweight="bold", pad=28)

    total = values.sum()
    acc = (counts["tn"] + counts["tp"]) / total
    recall = counts["tp"] / max(counts["tp"] + counts["fn"], 1)
    fpr = counts["fp"] / max(counts["fp"] + counts["tn"], 1)
    ax.text(
        0.5,
        -0.18,
        f"accuracy={acc:.3f}   recall={recall:.3f}   FPR={fpr:.3f}   AUC={counts['auc']:.3f}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        color=MUTED,
        fontsize=11.5,
    )


def main() -> None:
    folds = np.load(FOLDS)
    logits, meta = score_cross_region_seed_logits(folds)
    matrices = [
        (
            "Model B on region A",
            CYAN,
            confusion_from_logits(logits["A_pos"], logits["A_neg"]),
        ),
        (
            "Model A on region B",
            ORANGE,
            confusion_from_logits(logits["B_pos"], logits["B_neg"]),
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.7))
    fig.patch.set_alpha(0.0)
    for ax, (title, accent, counts) in zip(axes, matrices):
        draw_matrix(ax, title, counts, accent)

    fig.suptitle(
        "Cross-region confusion matrix (seeds only)",
        color=WHITE,
        fontsize=24,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.905,
        f"threshold: p_outlier >= {THRESHOLD:.1f} (logit >= 0)  |  best epochs: A={meta['best_A_epoch']}, B={meta['best_B_epoch']}",
        ha="center",
        va="center",
        color=MUTED,
        fontsize=13,
        style="italic",
    )

    fig.subplots_adjust(left=0.08, right=0.98, top=0.74, bottom=0.18, wspace=0.26)
    fig.savefig(OUT_PNG, dpi=250, transparent=True)
    fig.savefig(OUT_SVG, transparent=True)
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_SVG}")
    for title, _, c in matrices:
        print(
            f"{title}: TN={c['tn']:,} FP={c['fp']:,} "
            f"FN={c['fn']:,} TP={c['tp']:,}"
        )


if __name__ == "__main__":
    main()
