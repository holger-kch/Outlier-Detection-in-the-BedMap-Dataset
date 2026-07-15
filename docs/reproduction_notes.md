# Reproduction Notes

The repository contains the GNN source code, Slurm wrappers, small JSON
metadata, and result figures. It does not contain the BedMap data table or the
heavy generated artifacts.

## Not Tracked

- Raw or processed BedMap parquet files.
- Full graph arrays such as `spatial_edge_index_v3_k16.npy`.
- Node-feature arrays.
- Score parquets.
- Model checkpoints.
- Slurm logs.
- PowerPoint decks, slide backgrounds, and decorative photos.

## Original Run Environment

The scripts were developed on the NBI/HEP cluster and many still contain
absolute paths to the original project and data locations. A full rerun expects
the BedMap source table and the same Python/GPU environment.

Main dependencies:

- Python 3
- numpy, pandas, pyarrow, scipy, scikit-learn
- matplotlib, Pillow
- PyTorch
- PyTorch Geometric
- Optuna
- Slurm for the provided job scripts

## Run Order

From the original cluster project root, the GNN flow is:

```bash
# 1. Build or inspect pseudo-label seeds
sbatch src/slurm/run_step1_2_cpu.slurm
sbatch src/slurm/run_seed_map_cpu.slurm

# 2. Prepare GNN arrays
sbatch src/slurm/run_physae_prepare_v4.slurm --stage features
sbatch src/slurm/run_physae_prepare_v4.slurm --stage edge_attr
sbatch src/slurm/run_physae_prepare_v4.slurm --stage folds

# 3. Optional hyperparameter search
sbatch src/slurm/run_optuna_physae_v4_gpu.slurm

# 4. Train both cross-region models
sbatch src/slurm/run_physae_train_v4_gpu.slurm --fold A
sbatch src/slurm/run_physae_train_v4_gpu.slurm --fold B

# 5. Validate and infer
sbatch src/slurm/run_validate_physae_v4_gpu.slurm
sbatch src/slurm/run_physae_infer_v4_gpu.slurm
```

The plotting scripts in [src/presentation](../src/presentation/) regenerate the
figures after the score and fold artifacts exist.

For adapting the scripts to a different directory structure or dataset copy,
see [reuse_guide.md](reuse_guide.md).
