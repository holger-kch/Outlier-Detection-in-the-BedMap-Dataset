#!/usr/bin/env python3
"""Ice-thickness map with PhysAE/GNN outliers overlaid in red."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from PIL import Image


ROOT = Path("/groups/icecube/holgerkc/ML-ICE-PROJECT")
CLEAN = Path("/groups/icecube/janikh/Bedmap/bedmap_clean.parquet")
SCORES = ROOT / "catalyst_pipeline/outputs/physae_scores_v4.parquet"
OUTDIR = ROOT / "PP"
DEFAULT_THRESHOLDS = [0.5]
FIGSIZE = (8, 7)
DPI = 250
RED_RADIUS_PX = 1


def read_map_columns(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pf = pq.ParquetFile(path)
    n_total = pf.metadata.num_rows
    east = np.empty(n_total, dtype=np.float32)
    north = np.empty(n_total, dtype=np.float32)
    ice = np.empty(n_total, dtype=np.float32)

    off = 0
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, columns=["east", "north", "ice_thickness"])
        n = table.num_rows
        east[off:off + n] = table.column("east").to_numpy(zero_copy_only=False)
        north[off:off + n] = table.column("north").to_numpy(zero_copy_only=False)
        ice[off:off + n] = table.column("ice_thickness").to_numpy(zero_copy_only=False)
        off += n
        if (rg + 1) % 12 == 0:
            print(f"read map {rg + 1}/{pf.num_row_groups} row groups", flush=True)

    return east, north, ice


def threshold_tag(threshold: float) -> str:
    return f"{threshold:g}".replace(".", "p")


def output_paths(threshold: float) -> tuple[Path, Path]:
    tag = threshold_tag(threshold)
    png = OUTDIR / f"ice_thickness_physae_outliers_thr{tag}.png"
    black = OUTDIR / f"ice_thickness_physae_outliers_thr{tag}_black.png"
    return png, black


def read_outlier_rows_by_threshold(
    path: Path,
    thresholds: list[float],
) -> dict[float, np.ndarray]:
    pf = pq.ParquetFile(path)
    chunks = {thr: [] for thr in thresholds}

    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, columns=["row_id", "p_outlier"])
        row_id = table.column("row_id").to_numpy(zero_copy_only=False)
        score = table.column("p_outlier").to_numpy(zero_copy_only=False)
        finite = np.isfinite(score)
        for thr in thresholds:
            mask = finite & (score >= thr)
            if mask.any():
                chunks[thr].append(row_id[mask].astype(np.int64, copy=False))
        if (rg + 1) % 12 == 0:
            seen = ", ".join(
                f">={thr:g}: {sum(len(c) for c in chunks[thr]):,}"
                for thr in thresholds
            )
            print(
                f"read scores {rg + 1}/{pf.num_row_groups} row groups | {seen}",
                flush=True,
            )

    return {
        thr: np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
        for thr, parts in chunks.items()
    }


def stamp_red_points(
    rgba: np.ndarray,
    display_xy: np.ndarray,
    axes_bbox,
    radius_px: int = RED_RADIUS_PX,
) -> None:
    height, width = rgba.shape[:2]
    x = np.rint(display_xy[:, 0]).astype(np.int32)
    y = np.rint(height - 1 - display_xy[:, 1]).astype(np.int32)

    in_axes = (
        (display_xy[:, 0] >= axes_bbox.x0)
        & (display_xy[:, 0] <= axes_bbox.x1)
        & (display_xy[:, 1] >= axes_bbox.y0)
        & (display_xy[:, 1] <= axes_bbox.y1)
    )
    x = x[in_axes]
    y = y[in_axes]

    offsets = [(0, 0)]
    if radius_px >= 1:
        offsets.extend([(-1, 0), (1, 0), (0, -1), (0, 1)])

    red = np.array([255, 45, 45, 255], dtype=np.uint8)
    for dx, dy in offsets:
        xx = x + dx
        yy = y + dy
        keep = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
        rgba[yy[keep], xx[keep]] = red


def save_black_background_copy(src: Path, dst: Path) -> None:
    im = Image.open(src).convert("RGBA")
    bg = Image.new("RGBA", im.size, (0, 0, 0, 255))
    bg.alpha_composite(im)
    bg.convert("RGB").save(dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--thr",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS,
        help="One or more p_outlier thresholds to plot.",
    )
    parser.add_argument(
        "--red-radius-px",
        type=int,
        default=RED_RADIUS_PX,
        help="Pixel radius for the red outlier overlay.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = sorted(set(args.thr))

    east, north, ice = read_map_columns(CLEAN)
    outliers_by_threshold = read_outlier_rows_by_threshold(SCORES, thresholds)

    finite = np.isfinite(east) & np.isfinite(north) & np.isfinite(ice)
    for thr in thresholds:
        outliers_by_threshold[thr] = outliers_by_threshold[thr][finite[outliers_by_threshold[thr]]]
        print(
            f"background finite points: {int(finite.sum()):,} | "
            f"model outliers (p_outlier >= {thr:g}): {len(outliers_by_threshold[thr]):,}",
            flush=True,
        )

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    s = ax.scatter(
        east[finite],
        north[finite],
        c=ice[finite],
        cmap="viridis",
        s=1,
        vmin=0,
        vmax=4000,
        linewidths=0,
        edgecolors="none",
        rasterized=True,
    )

    ax.set_aspect("equal")
    ax.set_title("Ice thickness + model outliers", color="white")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    cb = plt.colorbar(s, ax=ax, shrink=0.6)
    cb.ax.tick_params(colors="white")
    cb.outline.set_edgecolor("white")
    cb.set_label("ice thickness (m)", color="white")

    fig.tight_layout(pad=0.15)
    fig.set_dpi(DPI)
    print("rendering original scatter background", flush=True)
    fig.canvas.draw()

    base_rgba = np.asarray(fig.canvas.buffer_rgba()).copy()

    for thr in thresholds:
        out_png, black_png = output_paths(thr)
        print(f"adding small red model-outlier overlay for threshold {thr:g}", flush=True)
        outlier_rows = outliers_by_threshold[thr]
        display_xy = ax.transData.transform(np.column_stack((east[outlier_rows], north[outlier_rows])))
        rgba = base_rgba.copy()
        stamp_red_points(rgba, display_xy, ax.bbox, radius_px=args.red_radius_px)

        Image.fromarray(rgba, "RGBA").save(out_png)
        print(f"wrote {out_png}", flush=True)
        save_black_background_copy(out_png, black_png)
        print(f"wrote {black_png}", flush=True)


if __name__ == "__main__":
    main()
