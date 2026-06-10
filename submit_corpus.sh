#!/bin/bash
#SBATCH --account=def-zhijing
#SBATCH --job-name=cfm-corpus
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/corpus_%j.out
#SBATCH --error=logs/corpus_%j.err

# CPU-only job: pre-generates a corpus of 10K paired-outcome tasks to
# $SCRATCH/CFM-for-Decision-Makers/outputs_corpus/.
#
# Why CPU-only:
#   - SCM sampling is CPU-bound (no GPU needed)
#   - GPU partition has long queue; CPU partition is fast
#   - Saves GPU-hours for actual training
#
# Resumes automatically: re-running just fills in any missing samples.
# So if a worker dies and loses 100 tasks, re-submit and they regenerate.

set -e
cd $SCRATCH/CFM-for-Decision-Makers
mkdir -p logs outputs_corpus

export UWYK_SRC=$SCRATCH/g4cfm/src
if [ ! -d "$UWYK_SRC" ]; then
    echo "ERROR: UWYK_SRC missing — clone on login node first"
    exit 1
fi

module purge
module load python/3.11
module load scipy-stack
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export N_SAMPLES=10000
export N_WORKERS=16
export OUT_DIR=$SCRATCH/CFM-for-Decision-Makers/outputs_corpus
export SEED_BASE=42
export CHUNK=50

echo "=== Node info ==="
hostname
date
echo "CPUs: $(nproc)"
echo "================="

time python generate_corpus.py
