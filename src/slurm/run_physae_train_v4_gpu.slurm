#!/bin/bash
#SBATCH --job-name=physae_train
#SBATCH --partition=icecube_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=12:00:00
#SBATCH --output=/groups/icecube/holgerkc/ML-ICE-PROJECT/catalyst_pipeline/logs/physae_train_%j.out
#SBATCH --error=/groups/icecube/holgerkc/ML-ICE-PROJECT/catalyst_pipeline/logs/physae_train_%j.err

# Train one model. Examples:
#   sbatch run_physae_train_v4_gpu.slurm --fold A      # cross-region fold (south -> rest)
#   sbatch run_physae_train_v4_gpu.slurm --fold B      # cross-region fold (rest -> south)
#   sbatch run_physae_train_v4_gpu.slurm --fold full   # all seeds, for final inference
# Graph is the full cached k=16 map. --resume is default.
# For a fresh train/validation-loss curve, add: --no-resume --val-every 1
set -euo pipefail
cd /groups/icecube/holgerkc/ML-ICE-PROJECT
mkdir -p catalyst_pipeline/logs catalyst_pipeline/outputs
export TMPDIR="/tmp/${USER}_${SLURM_JOB_ID}"; mkdir -p "${TMPDIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}" NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "physae train on $(hostname) at $(date)  args: $*"
nvidia-smi || true
python -u catalyst_pipeline/scripts/physae_gnn_v4.py "$@"
echo "finished at $(date)"
