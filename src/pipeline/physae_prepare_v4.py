#!/usr/bin/env python3
"""Step-4 GNN — input preparation (for the semi-supervised GraphSAGE node classifier).

Builds every static input the final-stage GNN needs, as separate Slurm-runnable
stages. NOTHING here uses BedMachine (ith_bm / delta_bm are NOT read). Node
features are FRESH intrinsic physics only; Step 1+2 supplies labels only (no
cone/octant/support statistics enter as features). Coordinates build EDGES only.

Graph: the full cached k=16 east/north kNN map (spatial_edge_index_v3_k16.npy is
reused directly as the [neighbour, node] edge_index; no k=8, no duplicate cache).

Stages (pick with --stage):
  features  source parquet   ->  standardized intrinsic-physics node feature matrix
  edge_attr per-edge [log1p(dist), signed-thickness-gradient] (geometry only), k=16
  folds     spatial cross-region 2-fold assignment of the seeds (region = folds only)

Identity convention: node index == row_id == row order of the clean parquet.

Leak fixes applied in `features`:
  * geom_resid (z - bed - thickness) is MASKED to "missing" where the bed was
    back-reconstructed (geom_resid identically 0 there -> it would leak a
    provenance flag). Detected from a `bed_reconstructed` column if present,
    else from |geom_resid| < 1e-3.
  * velocity magnitude `v` is NOT used (it equals hypot(vx,vy) exactly); we use
    log1p(speed) + unit flow direction instead.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

PROJECT = Path(__file__).resolve().parents[1]
OUTDIR = PROJECT / "outputs"
SOURCE = "/groups/icecube/janikh/Bedmap/bedmap_clean.parquet"

# kNN edge caches already built by the current pipeline (int32, length N*16, node-major, nearest-first)
SRC_NPY = OUTDIR / "spatial_edges_v3_src.npy"   # = node id, repeated 16x per node
DST_NPY = OUTDIR / "spatial_edges_v3_dst.npy"   # = neighbour id

K_FULL = 16         # we use the full cached k=16 map (maximum cross-track reach). No k=8.
SENTINEL = -9999.0

# physics columns we read (NO ith_bm, NO atd, NO identity/time)
PHYS_COLS = ["ice_thickness", "bed_evel", "z", "s", "vx", "vy", "smb", "temp"]

# final standardizable feature order (median/IQR robust-z); dir-x/dir-y and mask bits appended after
STD_FEATURES = ["sl_ice", "bed", "geom_resid", "surf_minus_bed", "log_speed", "slope", "surf", "smb", "temp"]
DIR_FEATURES = ["dir_x", "dir_y"]            # clipped [-1,1], NOT standardized
MASK_FEATURES = ["geom_resid", "smb", "temp"]  # finite/validity bit appended for each


def log(m):
    print(m, flush=True)


def signlog1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def to_nan_sentinel(a):
    a = a.astype(np.float64)
    a[~np.isfinite(a)] = np.nan
    a[a == SENTINEL] = np.nan
    return a


def _row_groups(pf):
    return range(pf.metadata.num_row_groups)


def _read_rg_cols(pf, rg, cols):
    t = pf.read_row_group(rg, columns=cols)
    return {c: t.column(c).to_numpy(zero_copy_only=False).astype(np.float64) for c in cols}


def derive_raw(cols, has_bedrecon, bedrecon=None):
    """From raw source columns -> the raw (pre-standardization) derived features
    and validity masks. Returns dict of float64 arrays."""
    ice = to_nan_sentinel(cols["ice_thickness"])
    bed = to_nan_sentinel(cols["bed_evel"])
    z = to_nan_sentinel(cols["z"])
    s = to_nan_sentinel(cols["s"])
    vx = to_nan_sentinel(cols["vx"])
    vy = to_nan_sentinel(cols["vy"])
    smb = to_nan_sentinel(cols["smb"])
    temp = to_nan_sentinel(cols["temp"])

    geom = z - bed - ice              # grounded-ice geometric consistency residual
    surf_minus_bed = z - bed          # geometric thickness from elevations
    speed = np.hypot(vx, vy)
    eps = 1e-6
    dir_x = np.clip(vx / (speed + eps), -1.0, 1.0)
    dir_y = np.clip(vy / (speed + eps), -1.0, 1.0)

    # leak fix: mask geom_resid where the bed was back-reconstructed (geom == 0 by construction)
    if has_bedrecon:
        recon = np.nan_to_num(bedrecon, nan=0.0) > 0.5
    else:
        recon = np.isfinite(geom) & (np.abs(geom) < 1e-3)
    geom_masked = geom.copy()
    geom_masked[recon] = np.nan      # treated as missing -> imputed + flagged by its mask bit

    raw = {
        "sl_ice": signlog1p(ice),
        "bed": bed,
        "geom_resid": geom_masked,
        "surf_minus_bed": surf_minus_bed,
        "log_speed": np.log1p(speed),
        "slope": s,
        "surf": z,
        "smb": smb,
        "temp": temp,
        "dir_x": dir_x,
        "dir_y": dir_y,
    }
    return raw


# ------------------------------------------------------------- stage: features
def stage_features(args):
    pf = pq.ParquetFile(SOURCE)
    N = pf.metadata.num_rows
    names = set(pf.schema_arrow.names)
    has_bedrecon = "bed_reconstructed" in names
    read_cols = list(PHYS_COLS) + (["bed_reconstructed"] if has_bedrecon else [])
    log(f"N={N:,}  bed_reconstructed column present: {has_bedrecon}")

    # ---- pass A: robust stats (median / IQR) from a strided subsample ----
    stride = max(1, N // args.stats_sample)
    samp = {f: [] for f in STD_FEATURES}
    t0 = time.time()
    for rg in _row_groups(pf):
        cols = _read_rg_cols(pf, rg, read_cols)
        br = cols.pop("bed_reconstructed", None) if has_bedrecon else None
        raw = derive_raw(cols, has_bedrecon, br)
        n = len(raw["bed"])
        sel = np.arange(0, n, stride)
        for f in STD_FEATURES:
            v = raw[f][sel]
            samp[f].append(v[np.isfinite(v)])
        if rg % 10 == 0:
            log(f"  stats pass rg {rg}/{pf.metadata.num_row_groups} ({time.time()-t0:.0f}s)")
    stats = {}
    for f in STD_FEATURES:
        v = np.concatenate(samp[f]) if samp[f] else np.array([0.0])
        med = float(np.median(v)) if v.size else 0.0
        q25, q75 = (float(np.percentile(v, 25)), float(np.percentile(v, 75))) if v.size else (0.0, 0.0)
        scale = (q75 - q25) / 1.349 + 1e-6
        stats[f] = {"median": med, "scale": float(scale), "n_sample": int(v.size)}
    log(f"robust stats computed ({time.time()-t0:.0f}s)")

    # ---- output layout ----
    n_std, n_dir, n_mask = len(STD_FEATURES), len(DIR_FEATURES), len(MASK_FEATURES)
    F_in = n_std + n_dir + n_mask
    x = np.lib.format.open_memmap(OUTDIR / "physae_node_features_v4.npy", mode="w+", dtype=np.float16, shape=(N, F_in))

    # ---- pass B: write standardized features + targets + masks ----
    off = 0
    for rg in _row_groups(pf):
        cols = _read_rg_cols(pf, rg, read_cols)
        br = cols.pop("bed_reconstructed", None) if has_bedrecon else None
        raw = derive_raw(cols, has_bedrecon, br)
        n = len(raw["bed"])
        block = np.zeros((n, F_in), np.float32)
        # standardized features
        for j, f in enumerate(STD_FEATURES):
            v = raw[f]
            zz = (v - stats[f]["median"]) / stats[f]["scale"]
            zz[~np.isfinite(zz)] = 0.0          # impute missing to the median (0 after z)
            block[:, j] = np.clip(zz, -12.0, 12.0)
        # direction features (already in [-1,1])
        for j, f in enumerate(DIR_FEATURES):
            v = raw[f]
            v[~np.isfinite(v)] = 0.0
            block[:, n_std + j] = v
        # mask bits (1 = present/valid)
        for j, f in enumerate(MASK_FEATURES):
            block[:, n_std + n_dir + j] = np.isfinite(raw[f]).astype(np.float32)
        x[off:off + n] = block.astype(np.float16)
        off += n
        if rg % 10 == 0:
            log(f"  write pass rg {rg}/{pf.metadata.num_row_groups} off={off:,} ({time.time()-t0:.0f}s)")
    x.flush()
    assert off == N, f"row mismatch {off} != {N}"

    meta = {
        "N": int(N), "F_in": int(F_in),
        "std_features": STD_FEATURES, "dir_features": DIR_FEATURES,
        "mask_features": MASK_FEATURES,
        "has_bedrecon_col": bool(has_bedrecon), "no_bedmachine": True,
        "dropped_v_collinear": True, "geom_resid_masked_where_bed_reconstructed": True,
        "stats": stats,
    }
    (OUTDIR / "physae_feature_stats_v4.json").write_text(json.dumps(meta, indent=2))
    log(f"=== features done === x{ x.shape } -> outputs/  ({time.time()-t0:.0f}s)")


# ------------------------------------------------------------ stage: edge_attr
def stage_edge_attr(args):
    # Built node-major from src/dst so the order matches BOTH the existing k=16 cache
    # (spatial_edge_index_v3_k16.npy) and our k<16 slice (physae_edge_index_v4_k*.npy).
    src = np.load(SRC_NPY, mmap_mode="r"); dst = np.load(DST_NPY, mmap_mode="r")
    E = len(src); N = E // K_FULL
    node = np.asarray(src).astype(np.int64)   # target (node-major, aligns with k=16 edge_index)
    nbr = np.asarray(dst).astype(np.int64)    # source (neighbour)
    pf = pq.ParquetFile(SOURCE)
    N = pf.metadata.num_rows
    log(f"edge_attr for {E:,} edges over {N:,} nodes")
    east = np.empty(N, np.float64); north = np.empty(N, np.float64); H = np.empty(N, np.float64)
    off = 0
    for rg in _row_groups(pf):
        t = pf.read_row_group(rg, columns=["east", "north", "ice_thickness"])
        e = t.column("east").to_numpy(zero_copy_only=False)
        n = len(e)
        east[off:off + n] = e
        north[off:off + n] = t.column("north").to_numpy(zero_copy_only=False)
        H[off:off + n] = to_nan_sentinel(t.column("ice_thickness").to_numpy(zero_copy_only=False))
        off += n
    # per-edge geometry (chunked)
    attr = np.lib.format.open_memmap(OUTDIR / "physae_edge_attr_v4_k16.npy", mode="w+",
                                     dtype=np.float16, shape=(E, 2))
    # robust scale for the signed gradient from a sample
    s0, s1 = 0, min(E, 5_000_000)
    d_s = np.hypot(east[node[s0:s1]] - east[nbr[s0:s1]], north[node[s0:s1]] - north[nbr[s0:s1]])
    g_s = (H[nbr[s0:s1]] - H[node[s0:s1]]) / np.maximum(d_s, 1.0)
    g_s = signlog1p(g_s); g_s = g_s[np.isfinite(g_s)]
    g_scale = (np.percentile(g_s, 75) - np.percentile(g_s, 25)) / 1.349 + 1e-6 if g_s.size else 1.0
    log(f"signed-gradient robust scale = {g_scale:.4f}")
    CH = 20_000_000
    for s in range(0, E, CH):
        e = min(s + CH, E)
        a = node[s:e]; b = nbr[s:e]
        d = np.hypot(east[a] - east[b], north[a] - north[b])
        grad = (H[b] - H[a]) / np.maximum(d, 1.0)
        c0 = np.log1p(d)                                   # distance (m) -> log1p
        c1 = signlog1p(grad) / g_scale                    # signed thickness gradient, robust-scaled
        c0[~np.isfinite(c0)] = 0.0; c1[~np.isfinite(c1)] = 0.0
        attr[s:e, 0] = np.clip(c0, 0.0, 20.0).astype(np.float16)
        attr[s:e, 1] = np.clip(c1, -12.0, 12.0).astype(np.float16)
        log(f"  edge_attr {e:,}/{E:,}")
    attr.flush()
    (OUTDIR / "physae_edge_attr_v4_k16_meta.json").write_text(
        json.dumps({"E": int(E), "k": 16, "cols": ["log1p_dist", "signed_grad_scaled"], "g_scale": float(g_scale)}, indent=2))
    log(f"=== edge_attr done === {attr.shape}")


# ---------------------------------------------------------------- stage: folds
def _read_clean_coords(ids):
    """east/north for a set of row_ids (sorted-merge over row groups)."""
    pf = pq.ParquetFile(SOURCE)
    ids = np.sort(np.asarray(ids, np.int64))
    out_e = np.empty(len(ids)); out_n = np.empty(len(ids))
    off = 0; cur = 0; fill = 0
    for rg in _row_groups(pf):
        n = pf.metadata.row_group(rg).num_rows
        s, e = off, off + n
        loc = []
        start = cur
        while cur < len(ids) and ids[cur] < e:
            if ids[cur] >= s:
                loc.append(int(ids[cur] - s))
            cur += 1
        if loc:
            t = pf.read_row_group(rg, columns=["east", "north"])
            ee = t.column("east").to_numpy(zero_copy_only=False)
            nn = t.column("north").to_numpy(zero_copy_only=False)
            loc = np.asarray(loc)
            out_e[fill:fill + len(loc)] = ee[loc]
            out_n[fill:fill + len(loc)] = nn[loc]
            fill += len(loc)
        off = e
    return ids, out_e[:fill], out_n[:fill]


def _kmeans2(xy, iters=50, seed=0):
    rng = np.random.default_rng(seed)
    c = xy[rng.choice(len(xy), 2, replace=False)]
    for _ in range(iters):
        d0 = ((xy - c[0]) ** 2).sum(1); d1 = ((xy - c[1]) ** 2).sum(1)
        lab = (d1 < d0).astype(np.int64)
        nc = np.stack([xy[lab == 0].mean(0) if (lab == 0).any() else c[0],
                       xy[lab == 1].mean(0) if (lab == 1).any() else c[1]])
        if np.allclose(nc, c):
            c = nc; break
        c = nc
    return lab, c


def _assign_region(e, n, cA, cB):
    """Nearest of two centroids (km). 0 = region A, 1 = region B."""
    xy = np.stack([e, n], 1) / 1000.0
    dA = ((xy - cA) ** 2).sum(1); dB = ((xy - cB) ** 2).sum(1)
    return (dB < dA).astype(np.int8)


def stage_folds(args):
    # True cross-region 2-fold: split ALL of Antarctica into two regions, so the held-out
    # region (outliers AND inliers AND unlabeled) is fully unseen during a fold's training.
    t = pq.read_table(OUTDIR / "step1_2_labels.parquet").to_pandas()
    out_ids = t.loc[t["final"] == 1, "row_id"].to_numpy(np.int64)
    in_ids = t.loc[t["final"] == 0, "row_id"].to_numpy(np.int64)
    log(f"seeds: outliers={len(out_ids):,}  inliers={len(in_ids):,}")

    # 2-region geographic split, defined by the outlier-seed clusters
    oid, oe, on = _read_clean_coords(out_ids)
    lab, c = _kmeans2(np.stack([oe, on], 1) / 1000.0, seed=args.seed)
    cA, cB = c[0], c[1]
    sep = float(np.hypot(*(cA - cB)))
    outliers_A = oid[lab == 0]; outliers_B = oid[lab == 1]

    # inliers assigned to the same two regions
    iid, ie, inn = _read_clean_coords(in_ids)
    ireg = _assign_region(ie, inn, cA, cB)
    inliers_A = iid[ireg == 0]; inliers_B = iid[ireg == 1]
    log(f"cross-region split (sep={sep:.0f} km): "
        f"A out={len(outliers_A):,} in={len(inliers_A):,} | B out={len(outliers_B):,} in={len(inliers_B):,}")

    # region label for ALL nodes (used to restrict the unlabeled pool per fold)
    pf = pq.ParquetFile(SOURCE); N = pf.metadata.num_rows
    region = np.lib.format.open_memmap(OUTDIR / "physae_region_v4.npy", mode="w+", dtype=np.int8, shape=(N,))
    off = 0; t0 = time.time()
    for rg in _row_groups(pf):
        tab = pf.read_row_group(rg, columns=["east", "north"])
        e = tab.column("east").to_numpy(zero_copy_only=False)
        n = tab.column("north").to_numpy(zero_copy_only=False)
        region[off:off + len(e)] = _assign_region(e, n, cA, cB)
        off += len(e)
    region.flush()
    nA = int((np.asarray(region) == 0).sum()); nB = N - nA
    log(f"all-node region: A={nA:,}  B={nB:,}  ({time.time()-t0:.0f}s)")

    np.savez(OUTDIR / "physae_cross_region_folds_v4.npz",
             outliers_A=outliers_A, outliers_B=outliers_B,
             inliers_A=inliers_A, inliers_B=inliers_B,
             centroidA=cA, centroidB=cB, separation_km=sep)
    (OUTDIR / "physae_cross_region_folds_v4.json").write_text(json.dumps(
        {"n_outliers_A": int(len(outliers_A)), "n_outliers_B": int(len(outliers_B)),
         "n_inliers_A": int(len(inliers_A)), "n_inliers_B": int(len(inliers_B)),
         "n_nodes_A": nA, "n_nodes_B": nB, "separation_km": sep,
         "note": "region defines folds ONLY, never a feature; held-out region fully unseen in training"},
        indent=2))
    log("=== folds done ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["features", "edge_attr", "folds"])
    ap.add_argument("--stats-sample", type=int, default=5_000_000)
    ap.add_argument("--seed", type=int, default=20260606)
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    {"features": stage_features, "edge_attr": stage_edge_attr,
     "folds": stage_folds}[args.stage](args)


if __name__ == "__main__":
    main()
