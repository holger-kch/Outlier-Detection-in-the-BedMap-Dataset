# Outlier Detection in the BedMap Dataset

## Semi-Supervised GNN Component

Repository: [Outlier-Detection-in-the-BedMap-Dataset](https://github.com/holger-kch/Outlier-Detection-in-the-BedMap-Dataset)

This repository contains the Graph Neural Network part of the BedMap outlier
detection project. The goal is to score every ice-thickness measurement in the
BedMap-style Antarctic survey dataset for whether it behaves like a measurement
outlier when compared with its physical variables and spatial neighbours.

The project starts from pseudo-labels rather than hand labels. A physics/support
selection identifies high-confidence outlier and inlier seeds. A semi-supervised
GraphSAGE model then learns from those seeds on a k-nearest-neighbour graph and
assigns an outlier probability to all `74,747,031` points.

Figure scope follows the final presentation: the repository explains the
context from slides 1-4 in text, and the tracked figures are restricted to the
GNN/pseudo-label material from slides 7-16 and Appendix 10-18.

<table>
  <tr>
    <td width="50%">
      <img src="figures/readme/ice_thickness_outlier_seeds.png" alt="Pseudo-label seeds on the ice-thickness map" width="100%">
    </td>
    <td width="50%">
      <img src="figures/readme/ice_thickness_gnn_outliers_thr0p7.png" alt="Ice thickness map with GNN outliers" width="100%">
    </td>
  </tr>
  <tr>
    <td>
      The GNN is trained from pseudo-labels: geometry/physics rules create
      high-confidence inlier and outlier seeds before any neural model is used.
    </td>
    <td>
      The final scoring pass marks high-probability model outliers in red on
      top of the ice-thickness map. At `p_outlier >= 0.7`, the GNN flags
      `702,675` points.
    </td>
  </tr>
</table>

## Project Context

BedMap collects Antarctic ice-thickness measurements from many survey campaigns
spanning decades. The core measurement comes from airborne radio echo sounding:
a VHF signal is sent through the ice, the bed echo is recorded, and travel time
is converted to ice thickness.

That history makes the dataset valuable, but also messy. The project targets
measurement errors such as sudden unphysical jumps along a track or local
disagreements between nearby independent tracks. Those errors matter because
ice-thickness fields are inputs to downstream Antarctic and climate modelling.

The GNN does not see only thickness. It uses a small set of physical variables
available for each point, including ice thickness, bed/surface geometry, flow,
surface slope, surface mass balance, and temperature. Coordinates are used to
build the graph edges, not as node features.

## Main Result

The GNN was evaluated by cross-region validation: train on one large spatial
region, test on the other, then mirror the split. Region membership is only used
to define the held-out folds; it is not a model feature.

| Test direction | Outlier seeds | Inlier seeds | AUC | Recall at 0.5 | FPR at 0.5 |
|---|---:|---:|---:|---:|---:|
| Model B on region A | 19,317 | 247,192 | 0.865 | 0.850 | 0.314 |
| Model A on region B | 19,564 | 399,482 | 0.858 | 0.519 | 0.055 |

Full-map inference:

| Threshold | Flagged points |
|---:|---:|
| `p_outlier >= 0.5` | 1,951,535 |
| `p_outlier >= 0.7` | 702,675 |
| `p_outlier >= 0.9` | 34,500 |

The useful outcome is not that every red point is automatically a confirmed
bad measurement. The useful outcome is a continent-wide prioritised candidate
list: the GNN transfers the pseudo-label signal from local seed regions to the
full survey graph.

## How The GNN Is Built

### 1. Pseudo-Labels

The model does not use manually labelled outliers. It uses seed labels produced
by a physics/support pipeline:

- `38,881` outlier seeds,
- `646,674` inlier seeds,
- roughly `74M` remaining unlabeled nodes.

The seed-selection code is in [src/pipeline/step1_2_candidates_support.py](src/pipeline/step1_2_candidates_support.py),
with inspection and summary helpers in [src/pipeline/analyze_step1_2_labels.py](src/pipeline/analyze_step1_2_labels.py)
and [src/pipeline/make_seed_map.py](src/pipeline/make_seed_map.py).

<p>
  <img src="figures/readme/double_hit.png" alt="Double-hit pseudo-label candidate" width="49%">
  <img src="figures/readme/support_relation.png" alt="Support relation for pseudo-labels" width="49%">
</p>

<img src="figures/readme/support_qualifications.png" alt="Support qualifications for pseudo-labels" width="760">

<p>
  <img src="figures/readme/cone.png" alt="Cone verdict for pseudo-labels" width="49%">
  <img src="figures/readme/ice_thickness_outlier_seeds.png" alt="Resulting pseudo-label seeds" width="49%">
</p>

### 2. Physics-Only Node Features

The GNN input deliberately avoids coordinates, track identifiers, time, and
survey metadata that could become shortcuts. It also avoids BedMachine
thickness/residual columns as model inputs. Node features are built from
physics/geophysical variables available in the BedMap-style table:

- measured ice thickness,
- bed elevation,
- geometric residuals,
- surface-minus-bed thickness,
- log speed and flow direction,
- surface slope and elevation,
- surface mass balance,
- temperature,
- missing-value mask bits.

The feature construction lives in [src/pipeline/physae_prepare_v4.py](src/pipeline/physae_prepare_v4.py).
The exact feature statistics from the run are kept in
[results/physae_feature_stats_v4.json](results/physae_feature_stats_v4.json).

<img src="figures/readme/gnn_features.png" alt="GNN physics feature table" width="760">

### 3. k-NN Spatial Graph

The graph connects each point to `k = 16` spatial neighbours. The node index is
the source row id in the original BedMap table. Edge attributes are:

- `log1p_dist`, the log-scaled spatial distance,
- `signed_grad_scaled`, the signed ice-thickness gradient across the edge.

The graph build scripts are [src/pipeline/build_spatial_graph_v3.py](src/pipeline/build_spatial_graph_v3.py)
and [src/pipeline/make_k16_edge_cache.py](src/pipeline/make_k16_edge_cache.py).
The run metadata are in [results/spatial_edges_v3_meta.json](results/spatial_edges_v3_meta.json)
and [results/physae_edge_attr_v4_k16_meta.json](results/physae_edge_attr_v4_k16_meta.json).

<img src="figures/readme/knn_map.png" alt="k-nearest-neighbour map concept" width="520">

<img src="figures/readme/edge_table.png" alt="GNN edge table with distance and signed gradient" width="760">

### 4. Edge-Gated GraphSAGE

The model is an edge-aware GraphSAGE classifier. Message passing lets the model
compare each measurement with nearby measurements, while edge attributes tell it
how far the neighbour is and whether the thickness change looks like a smooth
gradient or a sharp jump.

The implementation is in [src/pipeline/physae_gnn_v4.py](src/pipeline/physae_gnn_v4.py).
Hyperparameter search code is in [src/pipeline/optuna_physae_v4.py](src/pipeline/optuna_physae_v4.py),
and the selected configuration is stored in [results/optuna_best_v4.json](results/optuna_best_v4.json).

<img src="figures/readme/gnn_model.png" alt="Edge-gated GraphSAGE model diagram" width="860">

### 5. Cross-Region Validation

The validation asks whether a model trained in one spatial region can recover
seeds in the other region. This is harder and more meaningful than testing on
nearby points from the same survey area.

Validation code:

- [src/pipeline/validate_physae_v4.py](src/pipeline/validate_physae_v4.py)
- [src/presentation/physae_cross_region_seed_scores.py](src/presentation/physae_cross_region_seed_scores.py)
- [src/presentation/make_physae_cross_region_roc.py](src/presentation/make_physae_cross_region_roc.py)
- [src/presentation/make_physae_logit_distribution.py](src/presentation/make_physae_logit_distribution.py)
- [src/presentation/make_physae_confusion_matrix.py](src/presentation/make_physae_confusion_matrix.py)

<p>
  <img src="figures/readme/physae_training_history.png" alt="GNN training and validation loss" width="49%">
  <img src="figures/readme/physae_cross_region_roc.png" alt="Cross-region ROC" width="49%">
</p>

<p>
  <img src="figures/readme/physae_cross_region_logit_distribution.png" alt="Cross-region logit distributions" width="49%">
  <img src="figures/readme/physae_cross_region_confusion_matrix.png" alt="Cross-region confusion matrix" width="49%">
</p>

### 6. Full-Map Scoring

The final inference script scores all `74,747,031` nodes and writes a
point-level `p_outlier`. The plotting script overlays the high-score points on
the ice-thickness map.

Code:

- [src/pipeline/physae_gnn_v4.py](src/pipeline/physae_gnn_v4.py) for inference mode,
- [src/presentation/make_ice_thickness_physae_outlier_map.py](src/presentation/make_ice_thickness_physae_outlier_map.py)
  for the map.

<img src="figures/readme/ice_thickness_gnn_outliers_thr0p7.png" alt="Final GNN outlier map at threshold 0.7" width="760">

## Repository Layout

```text
.
├── src/
│   ├── pipeline/       # seed, graph, feature, GNN, validation code
│   ├── presentation/   # scripts that generated the GNN figures
│   └── slurm/          # cluster job wrappers for the GNN run
├── figures/
│   ├── pseudolabels/   # pseudo-label figures from slides/appendix
│   ├── gnn/            # original GNN diagrams
│   ├── results/        # result figures
│   └── readme/         # dark-background previews for GitHub
├── results/            # small JSON metadata and metrics
└── docs/               # navigation and reproduction notes
```

The full BedMap table, derived graph arrays, score parquets, model checkpoints,
logs, PowerPoint files, decorative backgrounds, and presentation build products
are intentionally not tracked.

## Navigation

- [docs/code_map.md](docs/code_map.md) maps each scientific step to code.
- [docs/figure_index.md](docs/figure_index.md) maps each figure to the script
  that generated it.
- [docs/results_summary.md](docs/results_summary.md) collects the key numbers.
- [docs/reproduction_notes.md](docs/reproduction_notes.md) explains what is
  needed to rerun the analysis on the original cluster environment.

## Scope

This repository focuses only on the GNN component. Other project components
from the group presentation, such as along-track kNN spike detection, PCA,
latent-space carving, graph autoencoder work, ensemble slides, and decorative
PowerPoint backgrounds, are deliberately excluded.
