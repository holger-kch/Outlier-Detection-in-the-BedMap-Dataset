# Project Coverage

This repository is meant to be readable without the original presentation deck.
The deck/PDF is not tracked. The material has been rewritten as a standalone
GitHub project with figures, code links, result metadata, and reuse notes.

## Covered Here

| Project part | Repository coverage |
|---|---|
| BedMap context and outlier motivation | Explained in the README project context. |
| Pseudo-label seed construction | [step1_2_candidates_support.py](../src/pipeline/step1_2_candidates_support.py), [double_hit.png](../figures/pseudolabels/double_hit.png), [support_relation.png](../figures/pseudolabels/support_relation.png), [cone.png](../figures/pseudolabels/cone.png). |
| Resulting pseudo-label seeds | [ice_thickness_outlier_seeds.png](../figures/pseudolabels/ice_thickness_outlier_seeds.png), [make_seed_map.py](../src/pipeline/make_seed_map.py). |
| GNN node features | [gnn_features.png](../figures/gnn/gnn_features.png), [physae_prepare_v4.py](../src/pipeline/physae_prepare_v4.py), [physae_feature_stats_v4.json](../results/physae_feature_stats_v4.json). |
| Spatial `k=16` graph and edge attributes | [knn_map.png](../figures/gnn/knn_map.png), [edge_table.png](../figures/gnn/edge_table.png), [build_spatial_graph_v3.py](../src/pipeline/build_spatial_graph_v3.py), [make_k16_edge_cache.py](../src/pipeline/make_k16_edge_cache.py), [physae_prepare_v4.py](../src/pipeline/physae_prepare_v4.py). |
| Edge-gated GraphSAGE model | [gnn_thumb.png](../figures/gnn/gnn_thumb.png), [gnn_model.png](../figures/gnn/gnn_model.png), [physae_gnn_v4.py](../src/pipeline/physae_gnn_v4.py), [optuna_physae_v4.py](../src/pipeline/optuna_physae_v4.py). |
| Training and cross-region validation | [physae_training_history.png](../figures/results/physae_training_history.png), [physae_cross_region_roc.png](../figures/results/physae_cross_region_roc.png), [physae_cross_region_logit_distribution.png](../figures/results/physae_cross_region_logit_distribution.png), [physae_cross_region_confusion_matrix.png](../figures/results/physae_cross_region_confusion_matrix.png), [validate_physae_v4.py](../src/pipeline/validate_physae_v4.py). |
| Full-map GNN scoring | [ice_thickness_gnn_outliers_thr0p7.png](../figures/results/ice_thickness_gnn_outliers_thr0p7.png), [physae_gnn_v4.py](../src/pipeline/physae_gnn_v4.py), [make_ice_thickness_physae_outlier_map.py](../src/presentation/make_ice_thickness_physae_outlier_map.py). |
| Code extraction for reuse | [reuse_guide.md](reuse_guide.md). |

## Deliberately Excluded

The repository is focused on Holger's GNN work. It does not include the full
presentation deck, decorative presentation backgrounds, raw BedMap data, graph
arrays, checkpoints, score parquets, or other group-method material such as
along-track kNN spike detection, PCA, latent-space carving, graph autoencoder
work, or ensemble material.

The absence of the presentation is intentional: the README and docs should carry
the explanation without requiring a reader to open external slides.
