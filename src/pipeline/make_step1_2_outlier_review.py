#!/usr/bin/env python3
"""Sample N random step1+2 OUTLIERS; show ONLY the APPROVED (cleaned) support + the outlier.

For each sampled outlier we re-run the model's own support establishment + cleaning
(via step1_2_candidates_support.clean_support) and draw ONLY the surviving,
independent (other-track) support that actually judged it — coloured by ice
thickness, with track-profile lines — plus the outlier as a red diamond.  Angular
sector spokes are drawn on the FLOOR of each panel as a surround guide.
A slider flips through the N outliers.

Output: catalyst_pipeline/review/step1_2_outliers_sample.html
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial import cKDTree
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step1_2_candidates_support as s12   # reuse the EXACT cleaning logic

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = "/groups/icecube/janikh/Bedmap/bedmap_clean.parquet"
LABELS = PROJECT / "outputs" / "step1_2_labels.parquet"
OUTHTML = PROJECT / "review" / "step1_2_outliers_sample.html"
COLS = ["east", "north", "ice_thickness", "track_id", "file_no"]


def log(m):
    print(m, flush=True)


def read_source(path):
    pf = pq.ParquetFile(path)
    tabs = [pf.read_row_group(i, columns=COLS) for i in range(pf.metadata.num_row_groups)]
    t = pa.concat_tables(tabs)
    return {c: t[c].to_numpy(zero_copy_only=False) for c in COLS}


def track_lines(ex, ny, iz, trk):
    xs, ys, zs = [], [], []
    for tid in np.unique(trk):
        sel = np.where(trk == tid)[0]
        if sel.size < 2:
            continue
        order = sel[np.argsort(ex[sel])]
        xs.extend(ex[order].tolist() + [None]); ys.extend(ny[order].tolist() + [None]); zs.extend(iz[order].tolist() + [None])
    return xs, ys, zs


def floor_grid(R, zf, n):
    """n radial spokes + a rim circle on the floor plane z=zf."""
    xs, ys, zs = [], [], []
    for a in range(n):
        th = a * 2 * np.pi / n
        xs += [0, R * np.cos(th), None]; ys += [0, R * np.sin(th), None]; zs += [zf, zf, None]
    th = np.linspace(0, 2 * np.pi, 60)
    xs += (R * np.cos(th)).tolist() + [None]; ys += (R * np.sin(th)).tolist() + [None]; zs += [zf] * len(th) + [None]
    return xs, ys, zs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(LABELS))
    ap.add_argument("--input", default=SOURCE)
    ap.add_argument("--out", default=str(OUTHTML))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--support-radius", type=float, default=2000.0)
    ap.add_argument("--cap", type=int, default=600)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-depth-change", type=float, default=200.0)
    ap.add_argument("--eta-fraction", type=float, default=0.01)
    ap.add_argument("--eta-floor", type=float, default=15.0)
    ap.add_argument("--min-points", type=int, default=100)
    ap.add_argument("--floor-sectors", type=int, default=8)
    ap.add_argument("--model-sectors", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ca = SimpleNamespace(k=args.k, max_depth_change=args.max_depth_change,
                         min_points=args.min_points)

    lab = pd.read_parquet(args.labels)
    out = lab[lab["final"] == 1]
    log(f"total outliers: {len(out):,}")
    n = min(args.n, len(out))
    sample = out.sample(n=n, random_state=args.seed).reset_index(drop=True)
    rows = sample["row_id"].to_numpy().astype(np.int64)

    log("reading source + building cKDTree …")
    src = read_source(args.input)
    east = src["east"].astype(np.float64); north = src["north"].astype(np.float64)
    ice = src["ice_thickness"].astype(np.float64); track = src["track_id"]; file_no = src["file_no"]
    tree = cKDTree(np.column_stack([east, north]))

    frames, steps = [], []
    first, first_title = None, ""
    for k, rid in enumerate(rows):
        cx, cy, cz = east[rid], north[rid], ice[rid]
        kk = min(args.cap, len(ice))
        dist, idx = tree.query([cx, cy], k=kk)
        idx = np.atleast_1d(idx); dist = np.atleast_1d(dist)
        m = (dist <= args.support_radius) & (idx != rid) & np.isfinite(ice[idx])
        sidx = idx[m]
        sx, sy, sH, strk, sfile = east[sidx], north[sidx], ice[sidx], track[sidx], file_no[sidx]

        # APPROVED support = the model's cleaning, then the independent (other-track) subset that judges it
        if len(sidx) >= 3:
            eta_l = max(args.eta_floor, args.eta_fraction * float(np.median(sH)))
            keep, _ = s12.clean_support(sx, sy, sH, strk, sfile, ca, eta_l)
        else:
            keep = np.zeros(len(sidx), dtype=bool)
        appr = keep & (strk != track[rid])
        ex, ny, iz, trk = sx[appr] - cx, sy[appr] - cy, sH[appr], strk[appr]

        # surround occupancy (model_sectors) of the approved support
        if len(ex):
            ang = np.arctan2(ny, ex) % (2 * np.pi)
            occ = len(np.unique(np.floor(ang / (2 * np.pi / args.model_sectors)).astype(int)))
            loc_med = float(np.median(iz)); ntr = int(len(np.unique(trk)))
            R = float(np.max(np.hypot(ex, ny))) * 1.05
            zf = float(min(iz.min(), cz)) - 0.06 * (max(iz.max(), cz) - min(iz.min(), cz) + 1)
        else:
            occ, loc_med, ntr, R, zf = 0, float("nan"), 0, args.support_radius, cz - 100
        title = (f"outlier {k+1}/{n} &nbsp; row_id={int(rid)} &nbsp; track={int(track[rid])}<br>"
                 f"H={cz:.0f} m &nbsp; approved other-track median={loc_med:.0f} m &nbsp; "
                 f"residual={cz-loc_med:+.0f} m &nbsp; approved support={len(ex)} ({ntr} tracks) &nbsp; "
                 f"sectors filled={occ}/{args.model_sectors}")

        sup = go.Scatter3d(x=ex, y=ny, z=iz, mode="markers",
                           marker=dict(size=3.0, color=iz, colorscale="Viridis",
                                       colorbar=dict(title="H [m]"), opacity=0.9),
                           text=[f"track={int(t)}<br>H={v:.0f}" for t, v in zip(trk, iz)],
                           hoverinfo="text", name="approved support")
        lx, ly, lz = track_lines(ex, ny, iz, trk)
        lines = go.Scatter3d(x=lx, y=ly, z=lz, mode="lines",
                             line=dict(color="rgba(60,60,60,0.35)", width=1.5),
                             hoverinfo="skip", name="track profiles")
        gx, gy, gz = floor_grid(R, zf, args.floor_sectors)
        grid = go.Scatter3d(x=gx, y=gy, z=gz, mode="lines",
                            line=dict(color="rgba(150,150,150,0.55)", width=1.2),
                            hoverinfo="skip", name=f"{args.floor_sectors} sectors (floor)")
        opt = go.Scatter3d(x=[0], y=[0], z=[cz], mode="markers",
                           marker=dict(size=7, color="red", symbol="diamond",
                                       line=dict(color="black", width=1)),
                           text=[f"OUTLIER H={cz:.0f}"], hoverinfo="text", name="OUTLIER")
        data = [sup, lines, grid, opt]
        frames.append(go.Frame(data=data, name=str(k), layout=go.Layout(title_text=title)))
        if k == 0:
            first, first_title = data, title
        steps.append(dict(method="animate", label=str(k + 1),
                          args=[[str(k)], dict(mode="immediate",
                                frame=dict(duration=0, redraw=True), transition=dict(duration=0))]))

    fig = go.Figure(data=first, frames=frames)
    fig.update_layout(
        title_text=first_title,
        scene=dict(xaxis_title="east − east₀ [m]", yaxis_title="north − north₀ [m]",
                   zaxis_title="ice thickness [m]"),
        sliders=[dict(active=0, currentvalue=dict(prefix="outlier #"), pad=dict(t=45), steps=steps)],
        width=1150, height=860, margin=dict(l=0, r=0, t=90, b=0),
        legend=dict(orientation="h", y=0.99),
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.out, include_plotlyjs="cdn")
    log(f"wrote {args.out}  ({n} outliers; approved support only + {args.floor_sectors}-sector floor)")


if __name__ == "__main__":
    main()
