# Step 4 — semi-supervised GraphSAGE node classifier

The last stage of the catalyst pipeline. A GNN that is **trained directly on the
Step 1+2 seeds** and **generalizes the outlier score to all 74,747,031** Bedmap
points — beyond the surveyed cross-sections that produced the seeds.

Decision basis: domain knowledge + the *current* pipeline only (prior modelling
attempts were deliberately not used to choose the architecture).

## What it is

A **semi-supervised GraphSAGE node classifier** over the cached east/north **k=16**
kNN graph.

- **Trained on the seeds.** The loss *is* the seed classification: 38,881 outlier
  seeds = positives, 646,674 inlier seeds = trusted negatives. Both classes set
  the decision boundary; the inlier seeds also pin real-but-extreme physics (deep
  troughs, ice streams, shelves) as GOOD so the model does not flag real nature.
- **Generalizes by message passing.** Each node's prediction is built from its
  physical neighbourhood, so the learned function applies to *every* node —
  including the ~74M unlabeled and the ~75% with zero cross-track neighbours.
- **Semi-supervised / PU for the unlabeled.** The ~74M unlabeled points are
  **never** treated as negatives. They enter an **nnPU** (non-negative
  positive-unlabeled) risk estimator so the model does not overfit the seed
  regions and spreads the signal to new regions.

The GNN **learns local consistency by message passing** — it does *not* reuse the
Step 1+2 cone-method; the cone/octant/support statistics are not features or
mechanism (Step 1+2 supplies labels only).

## Hard rules honoured

- **No BedMachine.** `ith_bm`/`delta_bm` are never read.
- **Step 1+2 = labels only.** No cone/octant/support statistics as features or mechanism.
- **Physics-only, label-free, track-agnostic node features.** Forbidden:
  `track_id`, `point_id`, `flight_id`, `file`/`file_no`, `track_method`,
  coordinates (`east/north/lon/lat` — build EDGES only), time, `atd`, labels.
- **k = 16** (the full cached map). No k=8.

## Two verified data-leak fixes (built in)

1. **`geom_resid` provenance leak.** Where the bed was back-reconstructed,
   `geom_resid = z − bed − thickness ≡ 0`, so raw `geom_resid` leaks a provenance
   flag. It is **masked to "missing"** there (from a `bed_reconstructed` column if
   present, else `|geom_resid| < 1e-3`), with a validity-mask bit.
2. **`v` collinearity.** `v == hypot(vx, vy)` exactly → `v` is dropped; we use
   `log1p(speed)` + unit flow direction.

## Node features (14 channels)

Robust-standardized (median / IQR) in chunks; NaN/sentinel → imputed to median
with a mask bit where relevant.

| # | feature | transform | why |
|---|---------|-----------|-----|
| 1 | ice_thickness | signed-log1p, robust-z | the measured quantity under QC |
| 2 | bed_evel | robust-z | independent bed geometry |
| 3 | geom_resid (masked) | robust-z + mask bit | grounded-ice consistency z−bed−thick |
| 4 | surf_minus_bed = z−bed | robust-z | geometric thickness from elevations |
| 5 | log1p(speed) | robust-z | flow regime (replaces `v`) |
| 6 | dir_x = vx/\|v\| | clip[-1,1] | along- vs across-flow, no track_id |
| 7 | dir_y = vy/\|v\| | clip[-1,1] | " |
| 8 | s (surface slope) | robust-z | steep margins vs flat interior |
| 9 | z (surface elev) | robust-z | interior vs margin/shelf regime |
| 10 | smb | robust-z + mask | accumulation/firn regime |
| 11 | temp | robust-z + mask | thermal/shelf regime |
| +3 | mask bits {geom_resid, smb, temp} | 0/1 | flags imputed values |

## Graph / edges (k=16)

- Reuse `outputs/spatial_edge_index_v3_k16.npy` directly: int64 `[neighbour, node]`,
  1,195,952,496 directed edges, node index = row_id. mmap-loaded (not RAM-resident).
- **Edge features (geometry only):** `[log1p(distance), signed thickness gradient]`.
  The signed along-track gradient lets a sustained ramp (real trough/outlet) and a
  self-reversing spike (error) be weighted differently.
- **The ~75% zero-cross-track nodes** need no special case: every node has a
  complete intrinsic feature vector and a defined score; for a collinear node,
  message passing degenerates to a learned 1-D along-track consistency filter.

## Architecture

`EdgeGatedSAGE` (mean aggr, message = MLP(x_j)·σ(W·edge_attr)) × 3 layers,
LayerNorm + GELU + residuals, hidden 128, latent 96, JumpingKnowledge('cat'),
classification head 96→64→1→sigmoid. ~0.2–0.3 M params.

## Training objective

**Outlier-favouring (cost-sensitive) nnPU — the OUTLIER seeds dominate the loss:**
`L = w_pos·E_P[ℓ+] + max(0, E_U[ℓ−] − π·E_P[ℓ−]) + γ_N·E_N[ℓ−]`,
where P = outlier seeds, N = inlier seeds (trusted neg), U = unlabeled (PU mixture,
sampled each epoch). `w_pos` (default 8) makes the outlier term dominant; the negative
pressure comes mainly from the **unlabeled PU term** (representative of the whole map),
so `γ_N` (inlier weight, default 0.3) is small and the model does NOT lean on the inlier
seeds. `π` (default 3e-3) only enters the PU negative correction. AdamW, cosine LR, bf16
autocast, EMA weights, 40 epochs, temperature-scaling calibration, checkpoint every 5,
`--resume` default. (Watch FPR in cross-region validation: if `w_pos` is too high it
over-flags — `w_pos`/`γ_N` are the dials.)

## Scalability (k=16 fits ~56 GB)

mmap edge_index/edge_attr/features → resident cost is the PyG CSC (~10–20 GB) +
small per-batch tensors. `num_workers=0` (avoids edge-list copy-on-write OOM),
NeighborLoader fanout `16,8,8`, in-place chunked standardization, GPU peak a few GB.

## Inference — cross-fit / out-of-fold (no training-on-final-labels)

The final dataset-wide score is produced by **cross-labelling**: every point is scored by
the fold model trained on the OTHER region (region A → model B, region B → model A), so no
point is ever scored by a model that trained on its own region's labels. Requires both fold
models. Output `outputs/physae_scores_v4.parquet`
(`row_id, p_outlier` [out-of-fold], `score_A, score_B, agreement, region`);
`agreement = 1 − |score_A − score_B|` is the confidence signal. A `full`-region model is
deliberately NOT used for the final labelling (it would score points it trained on).

## Validation — 2-cross-fold (train on one half of Antarctica, predict on the other)

- The continent is split into **two regions** (k-means on the outlier-seed
  coordinates; every node, outlier and inlier, gets a region via `physae_region_v4.npy`).
  **Fold A** trains ONLY on region A (its outliers + inliers + unlabeled) and is tested
  on **region B**'s held-out outliers vs region B's inliers; **fold B** is the mirror.
  The held-out region is **fully unseen** in training (its outliers, inliers AND
  unlabeled are all excluded). Report cross-region AUC + recall@0.5/0.7 + FPR. Region
  is used ONLY to define folds, never as a feature.
- The final dataset-wide scoring is **cross-fit / out-of-fold** — each region is scored by
  the OTHER region's model (see Inference). No `full`-region model is used for final labels.
- **Degree-stratified** recall by cross-track-degree bucket {0,1-2,3-7,8+} — must
  score the zero-cross-track seeds correctly, proving transfer to the ~75% regime.
- A candidate is **not "caught"** until a 3D local cross-track spot-check confirms
  it; `outliers_caught_book.md` is unchanged by model output.

## Run order

```bash
cd /groups/icecube/holgerkc/ML-ICE-PROJECT
S=catalyst_pipeline/slurm
# 1) static inputs (CPU)
sbatch $S/run_physae_prepare_v4.slurm --stage features
sbatch $S/run_physae_prepare_v4.slurm --stage edge_attr
sbatch $S/run_physae_prepare_v4.slurm --stage folds
# 2) cross-region generalization folds (GPU, independent)
sbatch $S/run_physae_train_v4_gpu.slurm --fold A
sbatch $S/run_physae_train_v4_gpu.slurm --fold B
sbatch $S/run_validate_physae_v4_gpu.slurm
# 3) score ALL points cross-fit / out-of-fold (uses the two fold models; no full model)
sbatch $S/run_physae_infer_v4_gpu.slurm
```

## Files

- `scripts/physae_prepare_v4.py` — features / edge_attr / folds stages.
- `scripts/physae_gnn_v4.py` — model (`PhysGNN`) + train (`--fold`) + inference (`--infer`).
- `scripts/validate_physae_v4.py` — cross-region + degree-stratified validation.
- `slurm/run_physae_*` — the Slurm wrappers.

(Filenames keep the `physae_*` prefix from the pipeline's v4 cache naming; the model
itself is a semi-supervised GraphSAGE classifier, not an autoencoder.)

## Open items

- `π` prior sweep on cross-region AUC; consider per-degree-bucket temperature.
- Optional graph-Laplacian consistency term / self-training are deferred (drift
  risk) — only if cross-region recall plateaus.
- Inference uses sampled neighbourhoods (fanout 16,8,8); raise fanout for a
  lower-variance final score if needed.

## Operational limits on this cluster at k=16 — READ BEFORE RUNNING

Hard-won (06/06/2026). Honor these to avoid the crash cycle.

**Node inventory:** `node162` = 187 GB RAM, 6× L4 (24 GB). `node161`/`node071`/`node072`
= 62 GB RAM, 2 GPU. **Only node162 is big enough for a k=16 job** — the 62 GB nodes cannot
fit even ONE k=16 worker.

1. **k=16 CSC build is the RAM peak: ~60–70 GB per process.** PyG `NeighborLoader.to_csc`
   over the 1.2B-edge graph faults the edge_index mmap (~19 GB) + argsort temp (~10 GB) +
   row (~10 GB) + edge_attr perm (~10 GB). Steady RAM is ~47–57 GB, but the BUILD spike is
   the killer. **Give each job ≥ 85–90 GB.** 60 GB → OOM-kill (SIGKILL, exit 9).

2. **Only ONE graph-CSC alive at a time.** Holding the train loader + eval loader together =
   two ~20 GB CSCs → OOM/SIGBUS. Free (`del` + `gc.collect()`) the train loader BEFORE
   building the eval loader.

3. **DataLoader workers (`num_workers>0`) overflow /dev/shm with large k=16 batches:**
   - default (fd) strategy → **SIGBUS** (Bus error, core dumped)
   - `set_sharing_strategy('file_system')` → **ENOSPC** "No space left on device" on `/torch_*`
     (fills /dev/shm or /tmp)
   Worst with big configs (hidden 384, fanout 16,12,8); happens even for ONE job.
   **→ use `num_workers=0` (zero IPC, zero /dev/shm) for reliability.** Trades GPU util for
   safety. A single moderate-config job CAN use workers (the baseline ran `num_workers=8` at
   99–100% GPU), but it is NOT reliable across the whole HPO search space.

4. **Concurrency is RAM-bound, not GPU-bound.** With `num_workers=0` + 90 GB/job, node162 fits
   ~2 concurrent k=16 jobs (2×90 = 180 < 187). More requires the shared-CSC refactor
   (precompute `colptr`/`row` once, mmap read-only) so each worker is tiny.

5. Always `TMPDIR=/tmp/<user>_<job>`; single Slurm partition per job; read a live job's GPU with
   `srun --jobid=<id> --overlap nvidia-smi`.

Note: the archived v3 lessons already flagged `num_workers=0` + "system RAM is the killer" +
"k=8 halves graph RAM". k=16 (chosen for cross-track reach) doubles the RAM/IPC pressure, so
these limits bite harder — at k=16, `num_workers=0` is effectively mandatory.
