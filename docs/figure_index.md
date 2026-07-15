# Figure Index

This index maps the tracked GNN figures to the code that generated them.

| Figure | Purpose | Code |
|---|---|---|
| [gnn_cover.png](../figures/gnn/gnn_cover.png) | Concept drawing: survey points as a graph with outliers lighting up. | [make_gnn_cover.py](../src/presentation/make_gnn_cover.py) |
| [gnn_inputs.png](../figures/gnn/gnn_inputs.png) | Shows the three GNN inputs: node features, graph, and seed labels. | [make_gnn_figs.py](../src/presentation/make_gnn_figs.py) |
| [gnn_features.png](../figures/gnn/gnn_features.png) | Lists the physics-only feature construction. | [make_gnn_features_fig.py](../src/presentation/make_gnn_features_fig.py), [physae_prepare_v4.py](../src/pipeline/physae_prepare_v4.py) |
| [edge_table.png](../figures/gnn/edge_table.png) | Explains edge attributes: distance and signed gradient. | [make_edge_table_fig.py](../src/presentation/make_edge_table_fig.py) |
| [knn_edge.png](../figures/gnn/knn_edge.png) | Explains one k-nearest-neighbour edge. | [make_knn_figs.py](../src/presentation/make_knn_figs.py) |
| [knn_map.png](../figures/gnn/knn_map.png) | Shows the k-NN graph idea on a map-like point cloud. | [make_knn_figs.py](../src/presentation/make_knn_figs.py) |
| [gnn_message.png](../figures/gnn/gnn_message.png) | Shows neighbourhood message passing and point-level outlier score. | [make_gnn_figs.py](../src/presentation/make_gnn_figs.py) |
| [gnn_model.png](../figures/gnn/gnn_model.png) | Full GNN model/training schematic. | [make_gnn_model_fig.py](../src/presentation/make_gnn_model_fig.py), [physae_gnn_v4.py](../src/pipeline/physae_gnn_v4.py) |
| [gnn_result.png](../figures/gnn/gnn_result.png) | Compact result-card figure from the presentation. | [make_gnn_figs.py](../src/presentation/make_gnn_figs.py) |
| [physae_training_history.png](../figures/results/physae_training_history.png) | Fold A/B training and validation loss. | [make_physae_training_history.py](../src/presentation/make_physae_training_history.py) |
| [physae_cross_region_roc.png](../figures/results/physae_cross_region_roc.png) | Cross-region ROC on held-out seed labels. | [make_physae_cross_region_roc.py](../src/presentation/make_physae_cross_region_roc.py), [physae_cross_region_seed_scores.py](../src/presentation/physae_cross_region_seed_scores.py) |
| [physae_cross_region_logit_distribution.png](../figures/results/physae_cross_region_logit_distribution.png) | Cross-region logit separation for outlier and inlier seeds. | [make_physae_logit_distribution.py](../src/presentation/make_physae_logit_distribution.py) |
| [physae_cross_region_confusion_matrix.png](../figures/results/physae_cross_region_confusion_matrix.png) | Held-out seed confusion matrices at threshold `p_outlier = 0.5`. | [make_physae_confusion_matrix.py](../src/presentation/make_physae_confusion_matrix.py) |
| [ice_thickness_gnn_outliers_thr0p7.png](../figures/results/ice_thickness_gnn_outliers_thr0p7.png) | Full-map ice-thickness view with `p_outlier >= 0.7` points in red. | [make_ice_thickness_physae_outlier_map.py](../src/presentation/make_ice_thickness_physae_outlier_map.py) |

`figures/readme/` contains dark-background PNG previews of the same figures so
they render clearly on GitHub.

