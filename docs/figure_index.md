# Figure Index

This index maps each tracked figure to the part of the analysis it explains and
to the code or metadata behind it. The original presentation deck is not tracked,
so the table is written to be useful without opening external slides.

## Pseudo-Label Figures

| Figure | Analysis step | Purpose | Code/source |
|---|---|---|---|
| [double_hit.png](../figures/pseudolabels/double_hit.png) | Pseudo-label candidate construction | Shows how two nearby independent tracks can nominate a pseudo-label candidate. | [make_pseudolabel_figs.py](../src/presentation/make_pseudolabel_figs.py), [step1_2_candidates_support.py](../src/pipeline/step1_2_candidates_support.py) |
| [support_relation.png](../figures/pseudolabels/support_relation.png) | Pseudo-label support test | Shows the local support relation used before assigning a pseudo-label. | [make_pseudolabel_figs.py](../src/presentation/make_pseudolabel_figs.py), [step1_2_candidates_support.py](../src/pipeline/step1_2_candidates_support.py) |
| [cone.png](../figures/pseudolabels/cone.png) | Pseudo-label cone verdict | Shows the cone verdict used to decide whether a candidate is physically consistent with support. | [make_pseudolabel_figs.py](../src/presentation/make_pseudolabel_figs.py), [step1_2_candidates_support.py](../src/pipeline/step1_2_candidates_support.py) |
| [ice_thickness_outlier_seeds.png](../figures/pseudolabels/ice_thickness_outlier_seeds.png) | Pseudo-label seed result | Shows the resulting inlier/outlier pseudo-label seeds on the ice-thickness map. | [make_seed_map.py](../src/pipeline/make_seed_map.py) |

The support requirements are documented in the README and implemented in
[step1_2_candidates_support.py](../src/pipeline/step1_2_candidates_support.py).
There is no separate artificial support-requirements figure in this repository.

## GNN Figures

| Figure | Analysis step | Purpose | Code/source |
|---|---|---|---|
| [gnn_thumb.png](../figures/gnn/gnn_thumb.png) | Semi-supervised GNN setup | Shows labelled seed nodes, unlabeled graph nodes, and message passing toward a scored node. | [make_gnn_thumb.py](../src/presentation/make_gnn_thumb.py), [physae_gnn_v4.py](../src/pipeline/physae_gnn_v4.py) |
| [gnn_features.png](../figures/gnn/gnn_features.png) | Node-feature construction | Lists the physics-only node features used by the GNN. | [make_gnn_features_fig.py](../src/presentation/make_gnn_features_fig.py), [physae_prepare_v4.py](../src/pipeline/physae_prepare_v4.py) |
| [gnn_model.png](../figures/gnn/gnn_model.png) | Edge-gated GraphSAGE model | Shows message passing, seed classes, and the training objective. | [make_gnn_model_fig.py](../src/presentation/make_gnn_model_fig.py), [physae_gnn_v4.py](../src/pipeline/physae_gnn_v4.py) |
| [knn_map.png](../figures/gnn/knn_map.png) | Spatial graph construction | Shows the `k=16` graph idea. | [make_knn_figs.py](../src/presentation/make_knn_figs.py), [build_spatial_graph_v3.py](../src/pipeline/build_spatial_graph_v3.py) |
| [edge_table.png](../figures/gnn/edge_table.png) | Edge-attribute construction | Shows the two edge attributes: relative distance and signed gradient. | [make_edge_table_fig.py](../src/presentation/make_edge_table_fig.py), [physae_prepare_v4.py](../src/pipeline/physae_prepare_v4.py) |

## Result Figures

| Figure | Analysis step | Purpose | Code/source |
|---|---|---|---|
| [physae_training_history.png](../figures/results/physae_training_history.png) | Fold training | Fold A/B training and validation loss. | [make_physae_training_history.py](../src/presentation/make_physae_training_history.py) |
| [physae_cross_region_roc.png](../figures/results/physae_cross_region_roc.png) | Cross-region validation | Cross-region ROC on held-out seed labels. | [make_physae_cross_region_roc.py](../src/presentation/make_physae_cross_region_roc.py), [physae_cross_region_seed_scores.py](../src/presentation/physae_cross_region_seed_scores.py) |
| [physae_cross_region_logit_distribution.png](../figures/results/physae_cross_region_logit_distribution.png) | Cross-region validation | Cross-region logit separation for outlier and inlier seeds. | [make_physae_logit_distribution.py](../src/presentation/make_physae_logit_distribution.py) |
| [physae_cross_region_confusion_matrix.png](../figures/results/physae_cross_region_confusion_matrix.png) | Cross-region validation | Held-out seed confusion matrices at threshold `p_outlier = 0.5`. | [make_physae_confusion_matrix.py](../src/presentation/make_physae_confusion_matrix.py) |
| [ice_thickness_gnn_outliers_thr0p7.png](../figures/results/ice_thickness_gnn_outliers_thr0p7.png) | Full-map scoring | Full-map ice-thickness view with `p_outlier >= 0.7` points in red. | [make_ice_thickness_physae_outlier_map.py](../src/presentation/make_ice_thickness_physae_outlier_map.py) |

`figures/readme/` contains dark-background PNG previews of the same allowed
figures so they render clearly on GitHub.
