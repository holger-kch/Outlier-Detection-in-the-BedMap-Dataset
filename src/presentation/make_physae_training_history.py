#!/usr/bin/env python3
"""Training-loss history figure for the two final GNN cross-region folds."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/groups/icecube/holgerkc/ML-ICE-PROJECT")
PP = ROOT / "PP"

LOGS = {
    "Fold A": ROOT / "catalyst_pipeline/logs/physae_train_60888143.out",
    "Fold B": ROOT / "catalyst_pipeline/logs/physae_train_60888144.out",
}
HISTORY = {
    "Fold A": ROOT / "catalyst_pipeline/outputs/physae_train_v4_A_history.json",
    "Fold B": ROOT / "catalyst_pipeline/outputs/physae_train_v4_B_history.json",
}
OUT_PNG = PP / "physae_training_history.png"
OUT_SVG = PP / "physae_training_history.svg"

BG = "#0b111a"
WHITE = "#f7f8fb"
MUTED = "#aeb6c2"
GRID = "#ffffff"
CYAN = "#6bd6ff"
ORANGE = "#ffb74d"
GREEN = "#5ae0a0"


def parse_history(path: Path):
    if not path.exists():
        return None
    import json

    rows = json.loads(path.read_text())
    if not rows:
        return None
    return {
        "epoch": np.array([r["epoch"] for r in rows], dtype=int),
        "loss": np.array([r["train_loss"] for r in rows], dtype=float),
        "val_loss": np.array([np.nan if r.get("val_loss") is None else r.get("val_loss", np.nan) for r in rows], dtype=float),
        "share": np.array([r.get("outlier_share", np.nan) for r in rows], dtype=float),
        "epoch_seconds": np.array([r.get("epoch_seconds", np.nan) for r in rows], dtype=float),
        "total_seconds": float(np.nansum([r.get("epoch_seconds", np.nan) for r in rows])),
    }


def parse_log(path: Path) -> dict[str, np.ndarray | float | str]:
    text = path.read_text(errors="ignore")
    rows = []
    for m in re.finditer(
        r"\[(?P<fold>[AB])\] ep (?P<epoch>\d+)/(?P<epochs>\d+) "
        r"loss=(?P<loss>[0-9.]+)(?: val_loss=(?P<val_loss>[0-9.]+|nan))? "
        r"outlier_share=(?P<share>[0-9.]+)% "
        r"\((?P<seconds>[0-9]+)s\)",
        text,
    ):
        rows.append(
            (
                int(m.group("epoch")),
                float(m.group("loss")),
                float(m.group("val_loss")) if m.group("val_loss") else np.nan,
                float(m.group("share")),
                int(m.group("seconds")),
            )
        )
    if not rows:
        raise RuntimeError(f"No epoch rows found in {path}")
    arr = np.array(rows, dtype=float)
    total = re.search(r"\[[AB]\] done; total (?P<seconds>[0-9]+)s", text)
    return {
        "epoch": arr[:, 0].astype(int),
        "loss": arr[:, 1],
        "val_loss": arr[:, 2],
        "share": arr[:, 3],
        "epoch_seconds": arr[:, 4],
        "total_seconds": float(total.group("seconds")) if total else float(arr[:, 4].sum()),
    }


def load_history(name: str):
    base = parse_log(LOGS[name])
    new = parse_history(HISTORY[name])
    if new is None:
        return base
    keep = base["epoch"] < new["epoch"][0]
    if not keep.any():
        return new
    return {
        "epoch": np.concatenate([base["epoch"][keep], new["epoch"]]),
        "loss": np.concatenate([base["loss"][keep], new["loss"]]),
        "val_loss": np.concatenate([base["val_loss"][keep], new["val_loss"]]),
        "share": np.concatenate([base["share"][keep], new["share"]]),
        "epoch_seconds": np.concatenate([base["epoch_seconds"][keep], new["epoch_seconds"]]),
        "total_seconds": float(base["epoch_seconds"][keep].sum() + new["epoch_seconds"].sum()),
    }


def validation_summary(histories) -> str:
    parts = []
    for name, hist in histories.items():
        valid = np.isfinite(hist["val_loss"])
        if not valid.any():
            continue
        epochs = hist["epoch"][valid]
        vals = hist["val_loss"][valid]
        best_epoch = int(epochs[int(np.argmin(vals))])
        stop_epoch = int(hist["epoch"][-1])
        short = name.replace("Fold ", "")
        parts.append(f"{short} stopped {stop_epoch} (best {best_epoch})")
    if parts:
        return "max 120 epochs, early stopping  |  " + " ;  ".join(parts)
    return "40 epochs  |  Fold A and Fold B"


def main() -> None:
    histories = {name: load_history(name) for name in LOGS}
    has_val = any(np.isfinite(hist["val_loss"]).any() for hist in histories.values())

    fig, ax = plt.subplots(figsize=(8.4, 5.15))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((0, 0, 0, 0))

    styles = {
        "Fold A": (CYAN, "o"),
        "Fold B": (ORANGE, "s"),
    }

    for name, hist in histories.items():
        color, marker = styles[name]
        ax.plot(
            hist["epoch"],
            hist["loss"],
            color=color,
            lw=2.8,
            marker=marker,
            ms=4.6,
            mfc=BG,
            mec=color,
            mew=1.2,
            label=f"{name} train loss",
        )
        valid = np.isfinite(hist["val_loss"])
        if valid.any():
            ax.plot(
                hist["epoch"][valid],
                hist["val_loss"][valid],
                color=color,
                lw=2.2,
                ls="--",
                marker="^",
                ms=4.4,
                mfc=BG,
                mec=color,
                mew=1.1,
                alpha=0.92,
                label=f"{name} validation loss",
            )

    title = "GNN training / validation loss" if has_val else "GNN training loss"
    ax.set_title(title, color=WHITE, fontsize=23, pad=28, weight="bold")
    ax.text(
        0.5,
        1.01,
        validation_summary(histories) if has_val else "40 epochs  |  Fold A and Fold B",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color=MUTED,
        fontsize=12.5,
        style="italic",
    )

    ax.set_xlabel("Epoch", color=WHITE, fontsize=17)
    ax.set_ylabel("Loss", color=WHITE, fontsize=17)

    max_epoch = int(max(hist["epoch"][-1] for hist in histories.values()))
    ax.set_xlim(0.5, max_epoch + 0.5)
    losses = []
    for hist in histories.values():
        losses.extend(hist["loss"])
        losses.extend(hist["val_loss"][np.isfinite(hist["val_loss"])])
    ax.set_ylim(max(0.0, float(np.nanmin(losses)) - 0.12), float(np.nanmax(losses)) + 0.18)
    ticks = list(range(1, max_epoch + 1, 4))
    if ticks[-1] != max_epoch:
        if max_epoch - ticks[-1] <= 2:
            ticks[-1] = max_epoch
        else:
            ticks.append(max_epoch)
    ax.set_xticks(ticks)
    ax.tick_params(colors=WHITE, labelsize=13)

    for spine in ax.spines.values():
        spine.set_color(WHITE)
        spine.set_alpha(0.55)

    ax.grid(color=GRID, alpha=0.15, lw=0.8)
    ax.set_axisbelow(True)

    leg = ax.legend(
        loc="upper right",
        ncol=2,
        frameon=True,
        facecolor=BG,
        edgecolor=WHITE,
        framealpha=0.72,
        fontsize=13,
        borderaxespad=0.8,
    )
    for txt in leg.get_texts():
        txt.set_color(WHITE)

    fig.subplots_adjust(left=0.11, right=0.98, top=0.78, bottom=0.18)
    fig.savefig(OUT_PNG, dpi=250, transparent=True)
    fig.savefig(OUT_SVG, transparent=True)
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_SVG}")
    for name, hist in histories.items():
        print(
            f"{name}: {len(hist['epoch'])} epochs, "
            f"loss {hist['loss'][0]:.4f}->{hist['loss'][-1]:.4f}"
            + (f", val {hist['val_loss'][np.isfinite(hist['val_loss'])][-1]:.4f}"
               if np.isfinite(hist["val_loss"]).any() else "")
        )


if __name__ == "__main__":
    main()
