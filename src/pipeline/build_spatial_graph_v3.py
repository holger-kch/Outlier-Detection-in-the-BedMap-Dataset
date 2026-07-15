#!/usr/bin/env python3
"""Build the full spatial k-NN graph over all 74.7M Bedmap points (on east/north).

cKDTree (2D, exact, CPU) -> chunked queries -> directed kNN edges saved to disk.
Node index == row_id (clean-parquet row order). Training symmetrizes the edges.
This is the physical-neighbour-consistency graph for the GNN; coordinates build
edges ONLY, never features.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

CLEAN = "/groups/icecube/janikh/Bedmap/bedmap_clean.parquet"
OUTDIR = Path("/lustre/hpc/icecube/holgerkc/ML-ICE-PROJECT/catalyst_pipeline/outputs")


def log(m): print(m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=2_000_000)
    ap.add_argument("--workers", type=int, default=-1)
    ap.add_argument("--src-out", default=str(OUTDIR / "spatial_edges_v3_src.npy"))
    ap.add_argument("--dst-out", default=str(OUTDIR / "spatial_edges_v3_dst.npy"))
    ap.add_argument("--meta-out", default=str(OUTDIR / "spatial_edges_v3_meta.json"))
    args = ap.parse_args()

    pf = pq.ParquetFile(CLEAN)
    N = pf.metadata.num_rows
    log(f"loading east/north for {N:,} nodes ...")
    east = np.empty(N, np.float32); north = np.empty(N, np.float32)
    off = 0; t0 = time.time()
    for rg in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(rg, columns=["east", "north"]).to_pandas()
        n = len(t)
        east[off:off + n] = t["east"].to_numpy(); north[off:off + n] = t["north"].to_numpy()
        off += n
    finite = np.isfinite(east) & np.isfinite(north)
    fin_idx = np.where(finite)[0].astype(np.int64)
    coords = np.stack([east[fin_idx], north[fin_idx]], 1).astype(np.float64)
    log(f"finite-coord nodes: {len(fin_idx):,} / {N:,} ({time.time()-t0:.0f}s); building cKDTree ...")
    tb = time.time()
    tree = cKDTree(coords)
    log(f"cKDTree built in {time.time()-tb:.0f}s; querying k={args.k} in chunks of {args.chunk:,}")

    M = len(fin_idx)
    E = M * args.k
    src = np.lib.format.open_memmap(args.src_out, mode="w+", dtype=np.int32, shape=(E,))
    dst = np.lib.format.open_memmap(args.dst_out, mode="w+", dtype=np.int32, shape=(E,))
    tq = time.time(); written = 0
    for s in range(0, M, args.chunk):
        e = min(s + args.chunk, M)
        _, nbr = tree.query(coords[s:e], k=args.k + 1, workers=args.workers)  # (chunk, k+1)
        nbr = nbr[:, 1:]  # drop self
        rows = np.repeat(np.arange(s, e, dtype=np.int64), args.k)
        src[written:written + rows.size] = fin_idx[rows].astype(np.int32)
        dst[written:written + rows.size] = fin_idx[nbr.reshape(-1)].astype(np.int32)
        written += rows.size
        log(f"  queried {e:,}/{M:,} ({time.time()-tq:.0f}s)")
    src.flush(); dst.flush()

    meta = {"n_nodes": int(N), "n_finite_nodes": int(M), "k": args.k,
            "n_directed_edges": int(written), "node_index_is_row_id": True,
            "src": args.src_out, "dst": args.dst_out, "runtime_seconds": time.time() - t0}
    Path(args.meta_out).write_text(json.dumps(meta, indent=2))
    log(f"wrote {written:,} directed edges -> {args.src_out}, {args.dst_out}")
    log(f"meta -> {args.meta_out}; total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
