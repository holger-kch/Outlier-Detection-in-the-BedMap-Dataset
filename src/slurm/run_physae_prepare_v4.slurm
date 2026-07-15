#!/bin/bash
#SBATCH --job-name=physae_prep
#SBATCH --partition=icecube_long
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/groups/icecube/holgerkc/ML-ICE-PROJECT/catalyst_pipeline/logs/physae_prep_%j.out
#SBATCH --error=/groups/icecube/holgerkc/ML-ICE-PROJECT/catalyst_pipeline/logs/physae_prep_%j.err

# Run one prepare stage (graph = the full cached k=16 map; no k=8). Examples:
#   sbatch run_physae_prepare_v4.slurm --stage features
#   sbatch run_physae_prepare_v4.slurm --stage edge_attr
#   sbatch run_physae_prepare_v4.slurm --stage folds
set -euo pipefail
cd /groups/icecube/holgerkc/ML-ICE-PROJECT
mkdir -p catalyst_pipeline/logs catalyst_pipeline/outputs
export TMPDIR="/tmp/${USER}_${SLURM_JOB_ID}"; mkdir -p "${TMPDIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}" NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "physae prepare on $(hostname) at $(date)  args: $*"
python -u catalyst_pipeline/scripts/physae_prepare_v4.py "$@"
echo "finished at $(date)"
