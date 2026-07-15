# Figure Index

This index maps the tracked figures to the code or slide source behind them.
Figure scope is restricted to Holger's GNN/pseudo-label material inside the
requested presentation window: main slides 7-16 and Appendix 10-18. Context
from slides 1-4 is explained in the README as text. Slide 7 is not tracked
because it covers the group's along-track kNN spike-detection method, not the
GNN component.

## Pseudo-Label Figures

| Figure | Slide source | Purpose | Code/source |
|---|---|---|---|
| [double_hit.png](../figures/pseudolabels/double_hit.png) | Slide 9 / Appendix 11 | Shows how two nearby independent tracks can nominate a pseudo-label candidate. | [make_pseudolabel_figs.py](../src/presentation/make_pseudolabel_figs.py), [step1_2_candidates_support.py](../src/pipeline/step1_2_candidates_support.py) |
| [support_relation.png](../figures/pseudolabels/support_relation.png) | Appendix 13 | Shows the local support relation used before assigning a pseudo-label. | [make_pseudolabel_figs.py](../src/presentation/make_pseudolabel_figs.py), [step1_2_candidates_support.py](../src/pipeline/step1_2_candidates_support.py) |
| [cone.png](../figures/pseudolabels/cone.png) | Appendix 14 | Shows the cone verdict used to decide whether a candidate is physically consistent with support. | [make_pseudolabel_figs.py](../src/presentation/make_pseudolabel_figs.py), [step1_2_candidates_support.py](../src/pipeline/step1_2_candidates_support.py) |
| [ice_thickness_outlier_seeds.png](../figures/pseudolabels/ice_thickness_outlier_seeds.png) | Slide 10 / Appendix 15 | Shows the resulting inlier/outlier pseudo-label seeds on the ice-thickness map. | [make_seed_map.py](../src/pipeline/make_seed_map.py) |

Appendix 12 is a text/formula slide. Its support requirements are documented
in the README and implemented in
[step1_2_candidates_support.py](../src/pipeline/step1_2_candidates_support.py);
there is no separate tracked figure for that slide.

## GNN Figures

| Figure | Slide source | Purpose | Code/source |
|---|---|---|---|
| [gnn_thumb.png](../figures/gnn/gnn_thumb.png) | Slide 8 | Shows the semi-supervised node-classification setup: labelled seed nodes, unlabeled graph nodes, and message passing toward a scored node. | [make_gnn_thumb.py](../src/presentation/make_gnn_thumb.py), [physae_gnn_v4.py](../src/pipeline/physae_gnn_v4.py) |
| [gnn_features.png](../figures/gnn/gnn_features.png) | Slide 11 | Lists the physics-only node features used by the GNN. | [make_gnn_features_fig.py](../src/presentation/make_gnn_features_fig.py), [physae_prepare_v4.py](../src/pipeline/physae_prepare_v4.py) |
| [gnn_model.png](../figures/gnn/gnn_model.png) | Slide 12 | Shows message passing, seed classes, and the training objective. | [make_gnn_model_fig.py](../src/presentation/make_gnn_model_fig.py), [physae_gnn_v4.py](../src/pipeline/physae_gnn_v4.py) |
| [knn_map.png](../figures/gnn/knn_map.png) | Slide 8 / Appendix 16 | Shows the `k=16` graph idea. | [make_knn_figs.py](../src/presentation/make_knn_figs.py), [build_spatial_graph_v3.py](../src/pipeline/build_spatial_graph_v3.py) |
| [edge_table.png](../figures/gnn/edge_table.png) | Appendix 17 | Shows the two edge attributes: relative distance and signed gradient. | [make_edge_table_fig.py](../src/presentation/make_edge_table_fig.py), [physae_prepare_v4.py](../src/pipeline/physae_prepare_v4.py) |

## Result Figures

| Figure | Slide source | Purpose | Code/source |
|---|---|---|---|
| [physae_training_history.png](../figures/results/physae_training_history.png) | Slide 13 / Appendix 18 | Fold A/B training and validation loss. | [make_physae_training_history.py](../src/presentation/make_physae_training_history.py) |
| [physae_cross_region_roc.png](../figures/results/physae_cross_region_roc.png) | Slide 14 | Cross-region ROC on held-out seed labels. | [make_physae_cross_region_roc.py](../src/presentation/make_physae_cross_region_roc.py), [physae_cross_region_seed_scores.py](../src/presentation/physae_cross_region_seed_scores.py) |
| [physae_cross_region_logit_distribution.png](../figures/results/physae_cross_region_logit_distribution.png) | Slide 15 | Cross-region logit separation for outlier and inlier seeds. | [make_physae_logit_distribution.py](../src/presentation/make_physae_logit_distribution.py) |
| [physae_cross_region_confusion_matrix.png](../figures/results/physae_cross_region_confusion_matrix.png) | Slide 14 | Held-out seed confusion matrices at threshold `p_outlier = 0.5`. | [make_physae_confusion_matrix.py](../src/presentation/make_physae_confusion_matrix.py) |
| [ice_thickness_gnn_outliers_thr0p7.png](../figures/results/ice_thickness_gnn_outliers_thr0p7.png) | Slide 16 | Full-map ice-thickness view with `p_outlier >= 0.7` points in red. | [make_ice_thickness_physae_outlier_map.py](../src/presentation/make_ice_thickness_physae_outlier_map.py) |

`figures/readme/` contains dark-background PNG previews of the same allowed
figures so they render clearly on GitHub.
