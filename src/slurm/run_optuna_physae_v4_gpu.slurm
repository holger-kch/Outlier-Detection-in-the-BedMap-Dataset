#!/bin/bash
#SBATCH --job-name=physae_optuna
#SBATCH --partition=icecube_gpu
#SBATCH --nodelist=node162
#SBATCH --array=0-0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=/groups/icecube/holgerkc/ML-ICE-PROJECT/catalyst_pipeline/logs/physae_optuna_%A_%a.out
#SBATCH --error=/groups/icecube/holgerkc/ML-ICE-PROJECT/catalyst_pipeline/logs/physae_optuna_%A_%a.err

# 4 parallel Optuna workers on node162 (6 GPUs / 187 GB), sharing one JournalStorage on lustre.
# Cheap proxy (8 epochs + subsampled pool); objective = cross-region AP (train A -> eval held-out B).
set -euo pipefail
cd /groups/icecube/holgerkc/ML-ICE-PROJECT
mkdir -p catalyst_pipeline/logs catalyst_pipeline/outputs
export TMPDIR="/tmp/${USER}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; mkdir -p "${TMPDIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}" NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
ulimit -n 65536 || true   # file_system sharing strategy opens many fds

echo "optuna worker ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date)"
python -u catalyst_pipeline/scripts/optuna_physae_v4.py --worker --timeout "${OPTUNA_TIMEOUT:-3600}"
echo "worker ${SLURM_ARRAY_TASK_ID} finished at $(date)"
