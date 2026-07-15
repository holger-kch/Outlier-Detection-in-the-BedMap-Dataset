#!/bin/bash
#SBATCH --job-name=seed_map
#SBATCH --partition=icecube_long
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/groups/icecube/holgerkc/ML-ICE-PROJECT/catalyst_pipeline/logs/seed_map_%j.out
#SBATCH --error=/groups/icecube/holgerkc/ML-ICE-PROJECT/catalyst_pipeline/logs/seed_map_%j.err

set -euo pipefail
cd /groups/icecube/holgerkc/ML-ICE-PROJECT
mkdir -p catalyst_pipeline/logs catalyst_pipeline/review
export TMPDIR="/tmp/${USER}_${SLURM_JOB_ID}"; mkdir -p "${TMPDIR}"
export MPLCONFIGDIR="${TMPDIR}/mpl"; mkdir -p "${MPLCONFIGDIR}"

echo "seed map on $(hostname) at $(date)"
python -u catalyst_pipeline/scripts/make_seed_map.py "$@"
echo "Finished at $(date)"
