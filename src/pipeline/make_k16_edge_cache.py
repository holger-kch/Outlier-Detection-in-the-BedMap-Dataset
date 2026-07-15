#!/usr/bin/env python3
"""Build the full k=16 oriented edge_index cache (neighbour->node, int64) from the
existing k=16 kNN edges. Standalone so it can run in the background while training
uses the k=8 cache."""
import time
import numpy as np
from pathlib import Path

OUTDIR = Path("/lustre/hpc/icecube/holgerkc/ML-ICE-PROJECT/catalyst_pipeline/outputs")
t0 = time.time()
src = np.load(OUTDIR / "spatial_edges_v3_src.npy", mmap_mode="r")
dst = np.load(OUTDIR / "spatial_edges_v3_dst.npy", mmap_mode="r")
print(f"edges={len(src):,}; stacking oriented int64 ...", flush=True)
ei = np.stack([np.asarray(dst, np.int64), np.asarray(src, np.int64)])  # neighbour -> node
out = OUTDIR / "spatial_edge_index_v3_k16.npy"
np.save(out, ei)
print(f"wrote {out} ({ei.shape[1]:,} edges) in {time.time()-t0:.0f}s", flush=True)
