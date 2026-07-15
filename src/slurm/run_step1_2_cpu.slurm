#!/bin/bash
#SBATCH --job-name=step1_2
#SBATCH --partition=icecube_long
#SBATCH --cpus-per-task=48
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
#SBATCH --output=/groups/icecube/holgerkc/ML-ICE-PROJECT/catalyst_pipeline/logs/step1_2_%j.out
#SBATCH --error=/groups/icecube/holgerkc/ML-ICE-PROJECT/catalyst_pipeline/logs/step1_2_%j.err

set -euo pipefail
cd /groups/icecube/holgerkc/ML-ICE-PROJECT
mkdir -p catalyst_pipeline/logs catalyst_pipeline/outputs
export TMPDIR="/tmp/${USER}_${SLURM_JOB_ID}"; mkdir -p "${TMPDIR}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1  # per-proc 1 thread; we parallelise over pairs

echo "STEP 1+2 on $(hostname) at $(date)  (cpus=${SLURM_CPUS_PER_TASK})"

# Args: pass extra flags through, e.g.  sbatch run_step1_2_cpu.slurm --limit 2000   (quick test)
python -u catalyst_pipeline/scripts/step1_2_candidates_support.py \
  --procs "${SLURM_CPUS_PER_TASK}" \
  "$@"

echo "Finished STEP 1+2 at $(date)"
