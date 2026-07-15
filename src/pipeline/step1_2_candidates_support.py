#!/usr/bin/env python3
"""STEP 1 + 2 (the ONLY approach) — implemented 1:1 from STEP1_2_SPEC.txt.

Starting point: two points within 25 m, from two different tracks = the candidates.

Step 1 - establish the support (2 km radius around the pair):
  early check: >=3 tracks, >=2 ids, >=10 points
  clean, in order:
    1. max ice-depth change to NN (k=3) <= 300 m
    2. s_max = 2*h / d_max,  h = P_max - P_min,  d_max = xy_distance(P_max, P_min)  (on survivors)
    3. max slope to (k=3) NN <= s_max*d + 2*eta
  accept check after cleaning: >=3 tracks, >=2 ids, >=10 points  else discard the double hit.

Step 2 - judge the candidates (cone method):
  for each candidate use only support from tracks OTHER than the candidate's own.
  candidate passes if within band from EVERY support point: |dH| <= s_max*d + 2*eta.
    one passes, one fails    -> inlier / outlier
    both fail                -> discard
    both pass and pair agrees (|H_c1-H_c2| <= 2*s_max*d12 + 2*eta) -> both inliers
    both pass but differ more -> discard
  eta = 50,  d = point-to-point distance,  d12 = distance between the two candidates.

No BedMachine, no embedding, no GNN. Just step 1+2 on every double hit.
Sharded (idx % num_shards == shard) so it parallelises across array tasks.
"""

import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

PROJECT = Path(__file__).resolve().parents[1]
OUTDIR = PROJECT / "outputs"
SOURCE = "/groups/icecube/janikh/Bedmap/bedmap_clean.parquet"
COLS = ["east", "north", "ice_thickness", "track_id", "file_no"]


def log(m):
    print(m, flush=True)


def read_source(path):
    pf = pq.ParquetFile(path)
    tabs = [pf.read_row_group(i, columns=COLS) for i in range(pf.metadata.num_row_groups)]
    t = pa.concat_tables(tabs)
    return {c: t[c].to_numpy(zero_copy_only=False) for c in COLS}


def find_pairs(coords, track, finite, tree, args):
    """All cross-track pairs (i<j) within pair_radius whose smaller index is in this shard."""
    n = len(track)
    I, J = [], []
    t0 = time.time()
    for start in range(0, n, args.chunk_size):
        end = min(start + args.chunk_size, n)
        idx = np.arange(start, end)
        idx = idx[(idx % args.num_shards) == args.shard]          # this shard's points
        idx = idx[finite[idx]]
        if len(idx) == 0:
            continue
        nbrs = tree.query_ball_point(coords[idx], r=args.pair_radius, workers=args.workers)
        for ii, neigh in zip(idx, nbrs):
            ti = track[ii]
            for j in neigh:
                if j <= ii:                                       # unordered pair once, smaller index owns it
                    continue
                if track[j] == ti or not finite[j]:
                    continue
                I.append(ii); J.append(j)
        if (start // args.chunk_size) % 20 == 0:
            log(f"  pair scan {end:,}/{n:,} ({100*end/n:.1f}%) pairs={len(I):,} {time.time()-t0:.0f}s")
        if args.limit and len(I) >= args.limit:                   # test mode: stop early
            log(f"  reached --limit ({args.limit}) pairs, stopping scan early")
            break
    return np.asarray(I, dtype=np.int64), np.asarray(J, dtype=np.int64)


def knn_maxabs(H, coords, k):
    """For each point, the MAX |dH| to its k nearest neighbours (xy). Returns array; needs len>k."""
    m = len(H)
    kk = min(k, m - 1)
    t = cKDTree(coords)
    dist, nn = t.query(coords, k=kk + 1)                          # incl self
    nn = nn[:, 1:]                                                # drop self
    return np.max(np.abs(H[:, None] - H[nn]), axis=1)


def knn_slope_ok(H, coords, k, s_max, eta):
    """Keep point if ALL its k nearest satisfy |dH| <= s_max*d + 2*eta."""
    m = len(H)
    kk = min(k, m - 1)
    t = cKDTree(coords)
    dist, nn = t.query(coords, k=kk + 1)
    dist = dist[:, 1:]; nn = nn[:, 1:]
    dH = np.abs(H[:, None] - H[nn])
    allowed = s_max * dist + 2.0 * eta
    return np.all(dH <= allowed, axis=1)                          # True = keep


def clean_support(sx, sy, sH, strk, sfile, args, eta):
    """Step 1 cleaning. Returns survivor mask over the support arrays, and s_max."""
    coords = np.column_stack([sx, sy])
    keep = np.ones(len(sH), dtype=bool)

    # 1. max ice-depth change to NN (k=3) <= 300 m
    maxabs = knn_maxabs(sH, coords, args.k)
    keep &= (maxabs <= args.max_depth_change)
    if keep.sum() < args.min_points:
        return keep, None

    # 2. s_max = 2*h / d_max on survivors
    sH1 = sH[keep]; c1 = coords[keep]
    imax = int(np.argmax(sH1)); imin = int(np.argmin(sH1))
    h = float(sH1[imax] - sH1[imin])
    d_max = float(np.hypot(c1[imax, 0] - c1[imin, 0], c1[imax, 1] - c1[imin, 1]))
    s_max = np.inf if d_max == 0.0 else 2.0 * h / d_max

    # 3. max slope to (k=3) NN <= s_max*d + 2*eta  (on survivors)
    ok = knn_slope_ok(sH1, c1, args.k, s_max, eta)
    surv = np.where(keep)[0][~ok]
    keep[surv] = False
    return keep, s_max


def passes_cone(Hc, cx, cy, sx, sy, sH, s_max, eta):
    """Candidate passes if within band from EVERY support point: |dH| <= s_max*d + 2*eta."""
    d = np.hypot(sx - cx, sy - cy)
    return bool(np.all(np.abs(Hc - sH) <= s_max * d + 2.0 * eta))


def process_pair(i, j, src, tree, args):
    """Return list of (row_id, label) verdicts. label 0=inlier, 1=outlier."""
    east, north = src["east"], src["north"]
    H, track, file_no = src["ice_thickness"], src["track_id"], src["file_no"]
    cx = 0.5 * (east[i] + east[j]); cy = 0.5 * (north[i] + north[j])

    # support = nearest points within support_radius (cap at knn for tractability), EXCLUDING the pair
    k = min(args.knn, len(H))
    dist, idx = tree.query([cx, cy], k=k)
    idx = np.atleast_1d(idx); dist = np.atleast_1d(dist)
    m = (dist <= args.support_radius) & (idx != i) & (idx != j) & np.isfinite(H[idx])
    sidx = idx[m]
    if len(sidx) < args.min_points:
        return []
    sx, sy, sH = east[sidx], north[sidx], H[sidx]
    strk, sfile = track[sidx], file_no[sidx]

    # early check (raw support)
    if (len(np.unique(strk)) < args.min_tracks or
            len(np.unique(sfile)) < args.min_ids or len(sidx) < args.min_points):
        return []

    # noise eta = eta_fraction * local ice depth (support median)
    eta = max(args.eta_floor, args.eta_fraction * float(np.median(sH)))
    # step 1 cleaning
    keep, s_max = clean_support(sx, sy, sH, strk, sfile, args, eta)
    # too much support pruned -> messy / contradictory neighbourhood -> discard
    if keep.size and (keep.size - int(keep.sum())) / keep.size > args.max_removed_frac:
        return []
    if s_max is None or keep.sum() < args.min_points:
        return []
    sx, sy, sH, strk, sfile = sx[keep], sy[keep], sH[keep], strk[keep], sfile[keep]

    # accept check after cleaning
    if (len(np.unique(strk)) < args.min_tracks or
            len(np.unique(sfile)) < args.min_ids or len(sH) < args.min_points):
        return []

    # step 2 verdict — each candidate judged on support from OTHER tracks only.
    # returns True/False (cone pass/fail) or None = "cannot judge" (not surrounded).
    def judge(c):
        tc = track[c]
        o = strk != tc
        if not o.any():
            return None
        dx = sx[o] - east[c]; dy = sy[o] - north[c]
        ang = np.arctan2(dy, dx) % (2.0 * np.pi)           # [0, 2pi)
        sec = np.floor(ang / (2.0 * np.pi / args.n_sectors)).astype(np.int64)
        counts = np.bincount(sec, minlength=args.n_sectors)
        if counts.min() < args.sector_min:                 # not surrounded in all n_sectors -> can't judge
            return None
        return passes_cone(H[c], east[c], north[c], sx[o], sy[o], sH[o], s_max, eta)

    p1, p2 = judge(i), judge(j)
    if p1 is None or p2 is None:                           # a candidate can't be judged -> discard
        return []
    d12 = float(np.hypot(east[i] - east[j], north[i] - north[j]))
    if p1 and p2:
        if abs(H[i] - H[j]) <= 2.0 * s_max * d12 + 2.0 * eta:
            return [(i, 0), (j, 0)]                               # both inliers
        return []                                                 # both pass but differ -> discard
    if p1 and not p2:
        return [(i, 0), (j, 1)]
    if p2 and not p1:
        return [(j, 0), (i, 1)]
    return []                                                     # both fail -> discard


def verify_point(c, src, tree, args):
    """Step 3 growth confirmation: judge a SINGLE grown candidate against its own
    support+cone (margin grow_slope_margin*s_max). Returns 1=confirmed outlier,
    0=inside band (reject), -1=cannot judge (discard)."""
    east, north = src["east"], src["north"]
    H, track, file_no = src["ice_thickness"], src["track_id"], src["file_no"]
    if not np.isfinite(H[c]):
        return -1
    cx, cy, Hc = east[c], north[c], H[c]
    k = min(args.knn, len(H))
    dist, idx = tree.query([cx, cy], k=k)
    idx = np.atleast_1d(idx); dist = np.atleast_1d(dist)
    m = (dist <= args.support_radius) & (idx != c) & np.isfinite(H[idx])
    sidx = idx[m]
    if len(sidx) < args.min_points:
        return -1
    sx, sy, sH, strk, sfile = east[sidx], north[sidx], H[sidx], track[sidx], file_no[sidx]
    if (len(np.unique(strk)) < args.min_tracks or
            len(np.unique(sfile)) < args.min_ids or len(sidx) < args.min_points):
        return -1
    eta = max(args.eta_floor, args.eta_fraction * float(np.median(sH)))
    keep, s_max = clean_support(sx, sy, sH, strk, sfile, args, eta)
    if keep.size and (keep.size - int(keep.sum())) / keep.size > args.max_removed_frac:
        return -1
    if s_max is None or keep.sum() < args.min_points:
        return -1
    sx, sy, sH, strk, sfile = sx[keep], sy[keep], sH[keep], strk[keep], sfile[keep]
    if (len(np.unique(strk)) < args.min_tracks or
            len(np.unique(sfile)) < args.min_ids or len(sH) < args.min_points):
        return -1
    o = strk != track[c]
    if not o.any():
        return -1
    dx = sx[o] - cx; dy = sy[o] - cy                       # 8-sector surround on other-track support
    sec = np.floor((np.arctan2(dy, dx) % (2.0 * np.pi)) / (2.0 * np.pi / args.n_sectors)).astype(np.int64)
    if np.bincount(sec, minlength=args.n_sectors).min() < args.sector_min:
        return -1
    inside = passes_cone(Hc, cx, cy, sx[o], sy[o], sH[o], s_max * args.grow_slope_margin, eta)
    return 0 if inside else 1


# --- multiprocessing: workers inherit the big tree + src via fork (copy-on-write, not copied) ---
_SRC = None
_TREE = None
_ARGS = None


def _worker(ij):
    return process_pair(int(ij[0]), int(ij[1]), _SRC, _TREE, _ARGS)


def _verify_worker(c):
    return verify_point(int(c), _SRC, _TREE, _ARGS)


def main():
    global _SRC, _TREE, _ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=SOURCE)
    ap.add_argument("--out", default=str(OUTDIR / "step1_2_labels.parquet"))
    ap.add_argument("--pair-radius", type=float, default=25.0)
    ap.add_argument("--pairs-cache", default="", help="npz cache of double-hit pairs (auto-named by radius if empty)")
    ap.add_argument("--rebuild-pairs", action="store_true", help="force re-scan even if a cache exists")
    ap.add_argument("--support-radius", type=float, default=2000.0)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--eta-fraction", type=float, default=0.01,
                    help="noise eta = eta_fraction * local ice depth (support median); 0.01 = 1%")
    ap.add_argument("--eta-floor", type=float, default=15.0, help="floor on eta [m]")
    ap.add_argument("--min-tracks", type=int, default=5)
    ap.add_argument("--min-ids", type=int, default=2)
    ap.add_argument("--min-points", type=int, default=100)
    ap.add_argument("--n-sectors", type=int, default=8,
                    help="candidate must be surrounded by support in EACH of this many angular sectors (8=octants)")
    ap.add_argument("--sector-min", type=int, default=1,
                    help="min other-track support points required in EACH sector")
    ap.add_argument("--max-depth-change", type=float, default=200.0)
    ap.add_argument("--max-removed-frac", type=float, default=0.30,
                    help="discard the double hit if cleaning removes more than this fraction of the support")
    # --- growth verify mode (step 3): confirm single grown candidates against the SAME support+cone ---
    ap.add_argument("--verify-input", default="",
                    help="parquet with row_id (filtered to final==1 if a 'final' col exists) to VERIFY as single points")
    ap.add_argument("--verify-output", default="", help="output parquet for verify mode")
    ap.add_argument("--grow-slope-margin", type=float, default=1.5,
                    help="verify cone uses grow_slope_margin*s_max (stricter, no coincident partner)")
    ap.add_argument("--knn", type=int, default=600, help="cap on support points (nearest within radius)")
    ap.add_argument("--chunk-size", type=int, default=200_000)
    ap.add_argument("--workers", type=int, default=-1)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--procs", type=int, default=0, help="CPU procs for the per-pair loop (0=auto)")
    ap.add_argument("--limit", type=int, default=0, help="process only first N pairs (test mode)")
    ap.add_argument("--progress-every", type=int, default=2000)
    args = ap.parse_args()
    if args.procs <= 0:
        args.procs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

    log(f"reading {args.input}")
    src = read_source(args.input)
    east, north, H = src["east"].astype(np.float64), src["north"].astype(np.float64), src["ice_thickness"].astype(np.float64)
    src["east"], src["north"], src["ice_thickness"] = east, north, H
    coords = np.column_stack([east, north])
    finite = np.isfinite(H)
    log(f"rows: {len(H):,}   building cKDTree …")
    tree = cKDTree(coords)

    # --- STEP 3 growth verify mode: confirm grown candidates as single points, then exit ---
    if args.verify_input:
        cand = pd.read_parquet(args.verify_input)
        if "final" in cand.columns:
            cand = cand[cand["final"] == 1]
        elif "pseudo_label" in cand.columns:
            cand = cand[cand["pseudo_label"] == 1]
        rows_v = cand["row_id"].to_numpy().astype(np.int64)
        log(f"verifying {len(rows_v):,} candidate points (grow margin {args.grow_slope_margin}x s_max) …")
        _SRC, _TREE, _ARGS = src, tree, args
        res = []
        t0 = time.time()
        with Pool(processes=args.procs) as pool:
            for nn, v in enumerate(pool.imap(_verify_worker, rows_v.tolist(), chunksize=256)):
                res.append(v)
                if (nn + 1) % 5000 == 0:
                    log(f"  verified {nn+1:,}/{len(rows_v):,} {time.time()-t0:.0f}s")
        res = np.asarray(res)
        outp = args.verify_output or args.verify_input.replace(".parquet", "_verified.parquet")
        pd.DataFrame({"row_id": rows_v, "verdict": res}).to_parquet(outp, index=False)
        log(f"=== VERIFY DONE === confirmed_outlier={int((res==1).sum()):,}  "
            f"rejected_inside={int((res==0).sum()):,}  discard={int((res==-1).sum()):,}")
        log(f"wrote {outp}")
        return

    # --- double-hit pairs: load from cache if available, else scan once and CACHE ---
    cache = args.pairs_cache or str(OUTDIR / f"step1_2_pairs_r{int(round(args.pair_radius))}m.npz")
    use_cache = (args.num_shards == 1 and not args.limit)        # cache only the full pair set
    if use_cache and (not args.rebuild_pairs) and Path(cache).exists():
        d = np.load(cache)
        I, J = d["I"], d["J"]
        log(f"loaded {len(I):,} double-hit pairs from cache {cache}  (skip scan)")
    else:
        log(f"finding double-hit pairs (shard {args.shard}/{args.num_shards}, r={args.pair_radius} m) …")
        I, J = find_pairs(coords, src["track_id"], finite, tree, args)
        log(f"double-hit pairs in this shard: {len(I):,}")
        if use_cache:
            np.savez_compressed(cache, I=I, J=J)
            log(f"cached double-hit pairs -> {cache}  (next qualification run skips the scan)")
    if args.limit and len(I) > args.limit:
        I, J = I[:args.limit], J[:args.limit]
        log(f"TEST MODE: limited to {len(I):,} pairs")

    verdicts = []
    stats = {"pairs": len(I), "inlier_outlier": 0, "both_inlier": 0, "discard": 0}
    t0 = time.time()

    def tally(v):
        if not v:
            stats["discard"] += 1
        elif len(v) == 2 and v[0][1] == 0 and v[1][1] == 0:
            stats["both_inlier"] += 1
        else:
            stats["inlier_outlier"] += 1
        verdicts.extend(v)

    if args.procs <= 1 or len(I) == 0:
        for n, (i, j) in enumerate(zip(I, J)):
            tally(process_pair(int(i), int(j), src, tree, args))
            if (n + 1) % args.progress_every == 0:
                log(f"  judged {n+1:,}/{len(I):,}  io={stats['inlier_outlier']:,} "
                    f"both_in={stats['both_inlier']:,} discard={stats['discard']:,} {time.time()-t0:.0f}s")
    else:
        # workers inherit src/tree/args via fork (copy-on-write); chunksize amortises IPC
        _SRC, _TREE, _ARGS = src, tree, args
        log(f"per-pair judging on {args.procs} CPU procs …")
        with Pool(processes=args.procs) as pool:
            for n, v in enumerate(pool.imap_unordered(_worker, zip(I.tolist(), J.tolist()), chunksize=256)):
                tally(v)
                if (n + 1) % args.progress_every == 0:
                    log(f"  judged {n+1:,}/{len(I):,}  io={stats['inlier_outlier']:,} "
                        f"both_in={stats['both_inlier']:,} discard={stats['discard']:,} {time.time()-t0:.0f}s")

    # aggregate per row_id across pairs: outlier if any-outlier & no-inlier; inlier if any-inlier & no-outlier; else conflict
    df = pd.DataFrame(verdicts, columns=["row_id", "label"]) if verdicts else pd.DataFrame(columns=["row_id", "label"])
    if len(df):
        g = df.groupby("row_id")["label"].agg(["max", "min", "count"]).reset_index()
        g["final"] = np.where(g["max"] == g["min"], g["max"], -1)   # -1 = conflict
        out = g.rename(columns={"count": "n_votes"})[["row_id", "final", "n_votes"]]
    else:
        out = pd.DataFrame(columns=["row_id", "final", "n_votes"])
    outp = args.out
    if args.num_shards > 1:
        outp = outp.replace(".parquet", f"_shard{args.shard:02d}of{args.num_shards}.parquet")
    out.to_parquet(outp, index=False)

    n_out = int((out["final"] == 1).sum()) if len(out) else 0
    n_in = int((out["final"] == 0).sum()) if len(out) else 0
    n_conf = int((out["final"] == -1).sum()) if len(out) else 0
    log(f"=== DONE shard {args.shard}/{args.num_shards} ===")
    log(f"pairs={stats['pairs']:,}  inlier/outlier={stats['inlier_outlier']:,}  "
        f"both_inlier={stats['both_inlier']:,}  discard={stats['discard']:,}")
    log(f"labelled points: inliers={n_in:,}  outliers={n_out:,}  conflict={n_conf:,}")
    log(f"wrote {outp}")


if __name__ == "__main__":
    main()
