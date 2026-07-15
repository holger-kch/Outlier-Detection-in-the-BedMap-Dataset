#!/usr/bin/env python3
"""Antarctic map of the locked step1+2 OUTLIER seeds (final==1) in red, on top of
all points coloured by ice thickness. Shows WHERE the seeds lie + track clustering."""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CLEAN = "/groups/icecube/janikh/Bedmap/bedmap_clean.parquet"
PROJ = "/lustre/hpc/icecube/holgerkc/ML-ICE-PROJECT/catalyst_pipeline"

ap = argparse.ArgumentParser()
ap.add_argument("--labels", default=f"{PROJ}/outputs/step1_2_labels.parquet")
ap.add_argument("--out", default=f"{PROJ}/review/step1_2_seed_map.png")
args = ap.parse_args()

pf = pq.ParquetFile(CLEAN); N = pf.metadata.num_rows
east = np.empty(N, np.float32); north = np.empty(N, np.float32)
h = np.empty(N, np.float32); track = np.empty(N, np.int64)
off = 0
for rg in range(pf.metadata.num_row_groups):
    t = pf.read_row_group(rg, columns=["east", "north", "ice_thickness", "track_id"]).to_pandas()
    n = len(t); east[off:off+n] = t["east"]; north[off:off+n] = t["north"]
    h[off:off+n] = t["ice_thickness"]; track[off:off+n] = t["track_id"]; off += n
    if (rg+1) % 18 == 0: print(f"read {rg+1}/72", flush=True)

lab = pd.read_parquet(args.labels, columns=["row_id", "final"])
seed_rid = lab.loc[lab["final"] == 1, "row_id"].to_numpy()
red = np.zeros(N, bool); red[seed_rid] = True
fin = np.isfinite(east) & np.isfinite(north) & np.isfinite(h)
print(f"seeds: {int(red.sum()):,} | finite background: {int(fin.sum()):,}", flush=True)

# track clustering of seeds
st = track[red]
vc = pd.Series(st).value_counts()
print(f"seeds span {vc.size:,} distinct tracks", flush=True)
print("top 15 tracks by seed count:", flush=True)
for tid, c in vc.head(15).items():
    print(f"  track {int(tid):>7}: {int(c):,} seeds", flush=True)

fig, ax = plt.subplots(figsize=(14, 12))
sc = ax.scatter(east[fin & ~red], north[fin & ~red], c=h[fin & ~red], cmap="viridis",
                vmin=0, vmax=4000, s=0.10, linewidths=0, rasterized=True)
cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.01); cb.set_label("ice_thickness (m)")
ax.scatter(east[red], north[red], s=2.0, c="red", edgecolors="none", rasterized=True,
           label=f"step1+2 outlier seeds ({int(red.sum()):,})", zorder=5)
ax.set_aspect("equal")
ax.set_title("step1+2 outlier seeds (red) over all Bedmap points (ice thickness)")
ax.set_xlabel("east (m)"); ax.set_ylabel("north (m)")
ax.legend(loc="lower left", framealpha=0.9, markerscale=6)
fig.tight_layout(); fig.savefig(args.out, dpi=170)
print(f"wrote {args.out}", flush=True)
