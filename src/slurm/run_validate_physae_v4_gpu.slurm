#!/bin/bash
#SBATCH --job-name=physae_val
#SBATCH --partition=icecube_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=03:00:00
#SBATCH --output=/groups/icecube/holgerkc/ML-ICE-PROJECT/catalyst_pipeline/logs/physae_val_%j.out
#SBATCH --error=/groups/icecube/holgerkc/ML-ICE-PROJECT/catalyst_pipeline/logs/physae_val_%j.err

# Cross-region + degree-stratified validation -> outputs/physae_validation_v4.json
# Requires physae_model_v4_A.pt and physae_model_v4_B.pt.
set -euo pipefail
cd /groups/icecube/holgerkc/ML-ICE-PROJECT
mkdir -p catalyst_pipeline/logs catalyst_pipeline/outputs
export TMPDIR="/tmp/${USER}_${SLURM_JOB_ID}"; mkdir -p "${TMPDIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}" NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "physae validation on $(hostname) at $(date)"
nvidia-smi || true
python -u catalyst_pipeline/scripts/validate_physae_v4.py
echo "finished at $(date)"
