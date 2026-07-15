#!/usr/bin/env python3
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

LAB = "/groups/icecube/holgerkc/ML-ICE-PROJECT/catalyst_pipeline/outputs/step1_2_labels.parquet"
SRC = "/groups/icecube/janikh/Bedmap/bedmap_clean.parquet"

pf = pq.ParquetFile(LAB)
print("=== SCHEMA ===", flush=True)
print(pf.schema_arrow, flush=True)
print("num_rows:", pf.metadata.num_rows, flush=True)

lab = pd.read_parquet(LAB)
print("\n=== columns ===", list(lab.columns), flush=True)
print("\n=== head ===\n", lab.head(10).to_string(), flush=True)
print("\n=== dtypes ===\n", lab.dtypes, flush=True)

print("\n=== final value_counts ===\n", lab["final"].value_counts(dropna=False).to_string(), flush=True)
print("\n=== n_votes describe (all) ===\n", lab["n_votes"].describe().to_string(), flush=True)
for v, nm in [(1, "OUTLIER"), (0, "INLIER"), (-1, "CONFLICT")]:
    sub = lab[lab["final"] == v]
    print(f"\n=== n_votes describe ({nm} n={len(sub):,}) ===\n", sub["n_votes"].describe().to_string(), flush=True)

print("\nrow_id range:", lab["row_id"].min(), lab["row_id"].max(), flush=True)
print("row_id dups:", lab["row_id"].duplicated().sum(), flush=True)

# join track_id/file_no from source for spatial/track concentration
src = pq.ParquetFile(SRC)
N = src.metadata.num_rows
print("\nsource rows:", N, flush=True)
track = np.empty(N, np.int64)
fileno = np.empty(N, np.int64)
off = 0
for rg in range(src.metadata.num_row_groups):
    t = src.read_row_group(rg, columns=["track_id", "file_no"]).to_pandas()
    n = len(t)
    track[off:off+n] = t["track_id"].to_numpy()
    fileno[off:off+n] = t["file_no"].to_numpy()
    off += n

for v, nm in [(1, "OUTLIER"), (0, "INLIER")]:
    rid = lab.loc[lab["final"] == v, "row_id"].to_numpy()
    tr = track[rid]
    fn = fileno[rid]
    vc = pd.Series(tr).value_counts()
    print(f"\n===== {nm} (n={len(rid):,}) =====", flush=True)
    print(f"distinct tracks: {vc.size:,}", flush=True)
    print(f"distinct files : {pd.Series(fn).nunique():,}", flush=True)
    print("top 20 tracks by seed count:", flush=True)
    for tid, c in vc.head(20).items():
        print(f"  track {int(tid):>8}: {int(c):>7,} seeds ({100*c/len(rid):.1f}%)", flush=True)
    top1 = vc.iloc[0]
    top3 = vc.head(3).sum()
    top10 = vc.head(10).sum()
    print(f"top-1 track share: {100*top1/len(rid):.1f}%", flush=True)
    print(f"top-3 track share: {100*top3/len(rid):.1f}%", flush=True)
    print(f"top-10 track share: {100*top10/len(rid):.1f}%", flush=True)
print("DONE", flush=True)
