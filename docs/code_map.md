# Code Map

This map follows the GNN workflow from seed labels to full-map scoring.

## 1. Pseudo-Label Seeds

- [step1_2_candidates_support.py](../src/pipeline/step1_2_candidates_support.py) builds the physics/support pseudo-label candidates.
- [analyze_step1_2_labels.py](../src/pipeline/analyze_step1_2_labels.py) summarizes the resulting label sets.
- [inspect_step1_2.py](../src/pipeline/inspect_step1_2.py) inspects candidate structure.
- [make_seed_map.py](../src/pipeline/make_seed_map.py) plots the seed map.
- [make_step1_2_outlier_review.py](../src/pipeline/make_step1_2_outlier_review.py) produces local review material.

## 2. Spatial Graph

- [build_spatial_graph_v3.py](../src/pipeline/build_spatial_graph_v3.py) builds the spatial k-nearest-neighbour graph.
- [make_k16_edge_cache.py](../src/pipeline/make_k16_edge_cache.py) converts the edge cache into the `k=16` oriented `edge_index`.
- [spatial_edges_v3_meta.json](../results/spatial_edges_v3_meta.json) records the graph size and node convention.
- [physae_edge_attr_v4_k16_meta.json](../results/physae_edge_attr_v4_k16_meta.json) records the edge-attribute columns.

## 3. Node Features And Folds

- [physae_prepare_v4.py](../src/pipeline/physae_prepare_v4.py) prepares node features, edge attributes, and cross-region folds.
- [physae_feature_stats_v4.json](../results/physae_feature_stats_v4.json) records the feature list, robust statistics, and leak-prevention flags.
- [physae_cross_region_folds_v4.json](../results/physae_cross_region_folds_v4.json) records the fold sizes and separation.

## 4. Model Training

- [physae_gnn_v4.py](../src/pipeline/physae_gnn_v4.py) implements the GNN model, training loop, and inference mode.
- [optuna_physae_v4.py](../src/pipeline/optuna_physae_v4.py) runs the hyperparameter search.
- [optuna_best_v4.json](../results/optuna_best_v4.json) stores the selected configuration.
- [physae_train_v4_A.json](../results/physae_train_v4_A.json) and [physae_train_v4_B.json](../results/physae_train_v4_B.json) store fold training summaries.

## 5. Validation And Figures

- [validate_physae_v4.py](../src/pipeline/validate_physae_v4.py) evaluates cross-region performance.
- [physae_cross_region_seed_scores.py](../src/presentation/physae_cross_region_seed_scores.py) scores held-out seed nodes with the opposite-region model.
- [make_physae_training_history.py](../src/presentation/make_physae_training_history.py) makes the training-history figure.
- [make_physae_cross_region_roc.py](../src/presentation/make_physae_cross_region_roc.py) makes the ROC figure.
- [make_physae_logit_distribution.py](../src/presentation/make_physae_logit_distribution.py) makes the logit-distribution figure.
- [make_physae_confusion_matrix.py](../src/presentation/make_physae_confusion_matrix.py) makes the confusion-matrix figure.

## 6. Full-Map Inference

- [physae_gnn_v4.py](../src/pipeline/physae_gnn_v4.py) runs full-map inference.
- [make_ice_thickness_physae_outlier_map.py](../src/presentation/make_ice_thickness_physae_outlier_map.py) overlays GNN outliers on the ice-thickness map.

