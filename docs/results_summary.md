# Results Summary

## Dataset And Labels

| Quantity | Value |
|---|---:|
| Total BedMap points scored | 74,747,031 |
| Outlier seed labels | 38,881 |
| Inlier seed labels | 646,674 |
| Remaining unlabeled points | about 74M |
| Graph neighbours per node | 16 |
| Directed graph edges | 1,195,952,496 |
| Node features | 14 |
| Edge features | 2 |

## Cross-Region Validation

| Test direction | Outlier seeds | Inlier seeds | AUC | Recall at 0.5 | FPR at 0.5 | Confusion matrix |
|---|---:|---:|---:|---:|---:|---|
| Model B on region A | 19,317 | 247,192 | 0.8653 | 0.8499 | 0.3135 | TN 169,695 · FP 77,497 · FN 2,899 · TP 16,418 |
| Model A on region B | 19,564 | 399,482 | 0.8585 | 0.5185 | 0.0553 | TN 377,395 · FP 22,087 · FN 9,420 · TP 10,144 |

## Full-Map Inference

| Score threshold | Flagged points | Fraction of all points |
|---:|---:|---:|
| `p_outlier >= 0.5` | 1,951,535 | 2.611% |
| `p_outlier >= 0.7` | 702,675 | 0.940% |
| `p_outlier >= 0.9` | 34,500 | 0.046% |

The final map in [figures/results/ice_thickness_gnn_outliers_thr0p7.png](../figures/results/ice_thickness_gnn_outliers_thr0p7.png)
uses the `0.7` threshold.

