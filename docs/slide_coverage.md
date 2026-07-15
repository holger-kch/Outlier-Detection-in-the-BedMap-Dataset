# Slide Coverage

This repository is scoped to Holger's GNN component. The final presentation
context from slides 1-4 is summarized in the README, while the scientific GNN
material is covered from the requested slide window and appendix.

PowerPoint backgrounds, logos, stock photos, decorative crops, and group-method
figures outside the GNN component are not tracked.

## Main Slides

| Presentation material | Repository coverage |
|---|---|
| Slides 1-4: BedMap context, measurement setting, and why outliers matter | Explained in the README's project context section. |
| Slide 07/32: along-track kNN spike detection | Excluded. This is the group's along-track kNN method, not the GNN component. |
| Slide 08/32: semi-supervised GNN overview | [gnn_thumb.png](../figures/gnn/gnn_thumb.png), [knn_map.png](../figures/gnn/knn_map.png), README GNN overview, [make_gnn_thumb.py](../src/presentation/make_gnn_thumb.py), [make_knn_figs.py](../src/presentation/make_knn_figs.py). |
| Slide 09/32: candidates for pseudo-labels | [double_hit.png](../figures/pseudolabels/double_hit.png), README support-rule explanation, [step1_2_candidates_support.py](../src/pipeline/step1_2_candidates_support.py). |
| Slide 10/32: resulting pseudo-labels | [ice_thickness_outlier_seeds.png](../figures/pseudolabels/ice_thickness_outlier_seeds.png), [make_seed_map.py](../src/pipeline/make_seed_map.py). |
| Slide 11/32: GNN input features | [gnn_features.png](../figures/gnn/gnn_features.png), [physae_prepare_v4.py](../src/pipeline/physae_prepare_v4.py), [physae_feature_stats_v4.json](../results/physae_feature_stats_v4.json). |
| Slide 12/32: message passing and model objective | [gnn_model.png](../figures/gnn/gnn_model.png), [physae_gnn_v4.py](../src/pipeline/physae_gnn_v4.py). |
| Slide 13/32: training | [physae_training_history.png](../figures/results/physae_training_history.png), [physae_train_v4_A.json](../results/physae_train_v4_A.json), [physae_train_v4_B.json](../results/physae_train_v4_B.json), [optuna_best_v4.json](../results/optuna_best_v4.json). |
| Slide 14/32: ROC and confusion matrices | [physae_cross_region_roc.png](../figures/results/physae_cross_region_roc.png), [physae_cross_region_confusion_matrix.png](../figures/results/physae_cross_region_confusion_matrix.png), validation and plotting scripts in [src/](../src). |
| Slide 15/32: logit distribution | [physae_cross_region_logit_distribution.png](../figures/results/physae_cross_region_logit_distribution.png), [make_physae_logit_distribution.py](../src/presentation/make_physae_logit_distribution.py). |
| Slide 16/32: final GNN outlier map | [ice_thickness_gnn_outliers_thr0p7.png](../figures/results/ice_thickness_gnn_outliers_thr0p7.png), [make_ice_thickness_physae_outlier_map.py](../src/presentation/make_ice_thickness_physae_outlier_map.py). |

## Appendix

| Appendix material | Repository coverage |
|---|---|
| Appendix 10: overview of pseudo-labels, kNN-16 map, GNN, and Optuna | Covered through the README workflow and links below. |
| Appendix 11: candidates for pseudo-labels | [double_hit.png](../figures/pseudolabels/double_hit.png), [step1_2_candidates_support.py](../src/pipeline/step1_2_candidates_support.py). |
| Appendix 12: support requirements | Text/formula slide. Requirements are explained in README and implemented in [step1_2_candidates_support.py](../src/pipeline/step1_2_candidates_support.py). No separate figure is tracked for this slide. |
| Appendix 13: support relation | [support_relation.png](../figures/pseudolabels/support_relation.png), [make_pseudolabel_figs.py](../src/presentation/make_pseudolabel_figs.py). |
| Appendix 14: cone verdict | [cone.png](../figures/pseudolabels/cone.png), [make_pseudolabel_figs.py](../src/presentation/make_pseudolabel_figs.py). |
| Appendix 15: resulting pseudo-labels | [ice_thickness_outlier_seeds.png](../figures/pseudolabels/ice_thickness_outlier_seeds.png), [make_seed_map.py](../src/pipeline/make_seed_map.py). |
| Appendix 16: kNN-16 map | [knn_map.png](../figures/gnn/knn_map.png), [build_spatial_graph_v3.py](../src/pipeline/build_spatial_graph_v3.py), [make_k16_edge_cache.py](../src/pipeline/make_k16_edge_cache.py). |
| Appendix 17: edge table | [edge_table.png](../figures/gnn/edge_table.png), [physae_prepare_v4.py](../src/pipeline/physae_prepare_v4.py). |
| Appendix 18: training and Optuna | [physae_training_history.png](../figures/results/physae_training_history.png), [optuna_physae_v4.py](../src/pipeline/optuna_physae_v4.py), [optuna_best_v4.json](../results/optuna_best_v4.json). |
