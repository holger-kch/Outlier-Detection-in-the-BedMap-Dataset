#!/bin/bash
#SBATCH --job-name=physae_infer
#SBATCH --partition=icecube_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=06:00:00
#SBATCH --output=/groups/icecube/holgerkc/ML-ICE-PROJECT/catalyst_pipeline/logs/physae_infer_%j.out
#SBATCH --error=/groups/icecube/holgerkc/ML-ICE-PROJECT/catalyst_pipeline/logs/physae_infer_%j.err

# Score all 74,747,031 points -> outputs/physae_scores_v4.parquet  (CROSS-FIT / out-of-fold)
# Requires BOTH fold A and B models. Each point is scored by the model trained on the OTHER
# region (region A -> model B, region B -> model A): no training-on-final-labels.
#   sbatch run_physae_infer_v4_gpu.slurm
set -euo pipefail
cd /groups/icecube/holgerkc/ML-ICE-PROJECT
mkdir -p catalyst_pipeline/logs catalyst_pipeline/outputs
export TMPDIR="/tmp/${USER}_${SLURM_JOB_ID}"; mkdir -p "${TMPDIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}" NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "physae infer on $(hostname) at $(date)  args: $*"
nvidia-smi || true
python -u catalyst_pipeline/scripts/physae_gnn_v4.py --infer "$@"
echo "finished at $(date)"
