#!/usr/bin/env python3
"""Analyze step1_2_labels.parquet: schema, seed distribution, source physics columns.
Read-only diagnostic. Run on slurm per project rule.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

LABELS = "/groups/icecube/holgerkc/ML-ICE-PROJECT/catalyst_pipeline/outputs/step1_2_labels.parquet"
SOURCE = "/groups/icecube/janikh/Bedmap/bedmap_clean.parquet"


def log(m):
    print(m, flush=True)


# 1. label parquet schema + counts
lab = pd.read_parquet(LABELS)
log("=== LABEL PARQUET ===")
log(f"path: {LABELS}")
log(f"shape: {lab.shape}")
log(f"columns + dtypes:\n{lab.dtypes}")
log(f"head:\n{lab.head(8).to_string()}")
if "final" in lab.columns:
    log(f"final value_counts:\n{lab['final'].value_counts().sort_index()}")
if "n_votes" in lab.columns:
    log(f"n_votes describe:\n{lab['n_votes'].describe()}")
log(f"row_id min/max: {lab['row_id'].min()} / {lab['row_id'].max()}")
log(f"row_id unique == len: {lab['row_id'].nunique() == len(lab)}")

# 2. source schema (intrinsic physics columns available)
pf = pq.ParquetFile(SOURCE)
log("\n=== SOURCE PARQUET SCHEMA ===")
log(f"path: {SOURCE}")
log(f"num_rows: {pf.metadata.num_rows:,}   num_row_groups: {pf.metadata.num_row_groups}")
log("schema:")
log(str(pf.schema_arrow))

# 3. seed distribution by track/file — join labels onto source identity columns
#    row_id = row number in source. Pull track_id, file_no, east, north for labelled rows only.
need = ["track_id", "file_no", "east", "north", "ice_thickness"]
have = [c for c in need if c in [f.name for f in pf.schema_arrow]]
log(f"\njoining on source cols: {have}")
# read full identity columns (cheap relative to full table) then index by row_id
tabs = [pf.read_row_group(i, columns=have) for i in range(pf.metadata.num_row_groups)]
import pyarrow as pa
t = pa.concat_tables(tabs)
src = {c: t[c].to_numpy(zero_copy_only=False) for c in have}
ridx = lab["row_id"].to_numpy().astype(np.int64)
seed = pd.DataFrame({"row_id": ridx, "final": lab["final"].to_numpy()})
for c in have:
    seed[c] = src[c][ridx]

log("\n=== SEED DISTRIBUTION ===")
for name, val in [("OUTLIER (1)", 1), ("INLIER (0)", 0), ("CONFLICT (-1)", -1)]:
    sub = seed[seed["final"] == val]
    log(f"\n-- {name}: {len(sub):,} points --")
    if "track_id" in sub.columns:
        log(f"distinct tracks: {sub['track_id'].nunique()}")
        log(f"top 15 tracks by count:\n{sub['track_id'].value_counts().head(15).to_string()}")
    if "file_no" in sub.columns:
        log(f"distinct files: {sub['file_no'].nunique()}")
    if "east" in sub.columns and len(sub):
        log(f"east  range: {sub['east'].min():.0f} .. {sub['east'].max():.0f}")
        log(f"north range: {sub['north'].min():.0f} .. {sub['north'].max():.0f}")
    if "ice_thickness" in sub.columns and len(sub):
        log(f"ice_thickness describe:\n{sub['ice_thickness'].describe().to_string()}")

# concentration: what fraction of outliers on the top-N tracks
out = seed[seed["final"] == 1]
if "track_id" in out.columns and len(out):
    vc = out["track_id"].value_counts()
    for N in [1, 3, 5, 9, 20]:
        frac = vc.head(N).sum() / len(out)
        log(f"\noutlier concentration: top {N} tracks hold {vc.head(N).sum():,}/{len(out):,} = {100*frac:.1f}%")

# coarse spatial regions (200 km bins) to gauge how many regions outliers span
if "east" in out.columns and len(out):
    bx = (out["east"] // 200000).astype(int)
    by = (out["north"] // 200000).astype(int)
    nreg = len(set(zip(bx, by)))
    log(f"\noutlier coarse 200km spatial cells occupied: {nreg}")
    ino = seed[seed["final"] == 0]
    bx2 = (ino["east"] // 200000).astype(int)
    by2 = (ino["north"] // 200000).astype(int)
    log(f"inlier  coarse 200km spatial cells occupied: {len(set(zip(bx2, by2)))}")

log("\n=== DONE ===")
