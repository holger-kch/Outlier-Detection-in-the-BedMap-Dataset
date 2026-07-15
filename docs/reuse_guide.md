# Reuse Guide

This repository keeps the original cluster analysis code, not a packaged Python
library. The code is still useful outside the original project, but a new user
must point the scripts at their own BedMap-style table and output directory.

## Main Entry Points

| Goal | Start here | What it does |
|---|---|---|
| Build pseudo-label seeds | [step1_2_candidates_support.py](../src/pipeline/step1_2_candidates_support.py) | Finds double-hit candidates and applies support/cone tests. |
| Plot pseudo-label seed map | [make_seed_map.py](../src/pipeline/make_seed_map.py) | Draws the inlier/outlier seed map. |
| Build spatial graph | [build_spatial_graph_v3.py](../src/pipeline/build_spatial_graph_v3.py) | Builds the `k=16` nearest-neighbour graph. |
| Prepare GNN inputs | [physae_prepare_v4.py](../src/pipeline/physae_prepare_v4.py) | Creates node features, edge attributes, and cross-region folds. |
| Train or infer with the GNN | [physae_gnn_v4.py](../src/pipeline/physae_gnn_v4.py) | Trains fold models and scores all nodes. |
| Validate cross-region performance | [validate_physae_v4.py](../src/pipeline/validate_physae_v4.py) | Evaluates AUC, recall, and FPR on held-out seed regions. |
| Regenerate README/result figures | [src/presentation/](../src/presentation/) | Rebuilds the figures after the required outputs exist. |

## Inputs A New User Must Provide

The pipeline expects a BedMap-style parquet table with one row per measurement.
The main columns used across the GNN workflow are:

- coordinates: `east`, `north`,
- measured target: `ice_thickness`,
- pseudo-label construction: `track_id`, `file_no`,
- node features: `bed_evel`, `z`, `s`, `vx`, `vy`, `smb`, `temp`,
- optional provenance/leak-control column: `bed_reconstructed`.

Large generated artifacts are intentionally not tracked here. A full rerun must
recreate graph arrays, feature arrays, fold arrays, score parquets, and model
checkpoints locally.

## Paths To Change Before Running Elsewhere

The scripts preserve the original cluster paths so the repository documents the
actual run. For reuse, edit the path constants at the top of these files:

| Path type | Files |
|---|---|
| BedMap source parquet | [step1_2_candidates_support.py](../src/pipeline/step1_2_candidates_support.py), [physae_prepare_v4.py](../src/pipeline/physae_prepare_v4.py), [validate_physae_v4.py](../src/pipeline/validate_physae_v4.py), [make_seed_map.py](../src/pipeline/make_seed_map.py), [make_ice_thickness_physae_outlier_map.py](../src/presentation/make_ice_thickness_physae_outlier_map.py) |
| Pipeline output directory | [build_spatial_graph_v3.py](../src/pipeline/build_spatial_graph_v3.py), [make_k16_edge_cache.py](../src/pipeline/make_k16_edge_cache.py), [physae_prepare_v4.py](../src/pipeline/physae_prepare_v4.py), [physae_gnn_v4.py](../src/pipeline/physae_gnn_v4.py), [optuna_physae_v4.py](../src/pipeline/optuna_physae_v4.py), [validate_physae_v4.py](../src/pipeline/validate_physae_v4.py) |
| Slurm working directory and logs | [src/slurm/](../src/slurm/) |
| Presentation figure output directory | [src/presentation/](../src/presentation/) |

The quickest adaptation is to copy the relevant script group, replace the
source parquet path, set `OUTDIR` to a writable local directory, and run the
stages in the order shown in [reproduction_notes.md](reproduction_notes.md).

## What To Reuse First

For understanding or adapting the method, start with:

1. [step1_2_candidates_support.py](../src/pipeline/step1_2_candidates_support.py)
   for the pseudo-label logic.
2. [physae_prepare_v4.py](../src/pipeline/physae_prepare_v4.py)
   for feature construction and leak prevention.
3. [physae_gnn_v4.py](../src/pipeline/physae_gnn_v4.py)
   for the EdgeGated GraphSAGE model.
4. [validate_physae_v4.py](../src/pipeline/validate_physae_v4.py)
   for the cross-region evaluation design.

The README gives the research story; [code_map.md](code_map.md) gives the
workflow map; this file is the practical extraction guide.
