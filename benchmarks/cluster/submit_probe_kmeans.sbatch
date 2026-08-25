#!/bin/bash
#SBATCH --job-name=uwyk-probe-km
#SBATCH --output=logs_probe_kmeans/probe_%j.out
#SBATCH --error=logs_probe_kmeans/probe_%j.err
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

# One-shot diagnostic: run UWYK's PreprocessingGraphConditionedPFN on CPS
# and PSID realization 0 with three different KMeans random_state values
# and print the resulting PEHE. See probe_kmeans.py header for how to read
# the three-way comparison.

set -euo pipefail
mkdir -p logs_probe_kmeans

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    if   [ -f "$(dirname "$DEPLOY_ROOT")/venv/bin/activate" ]; then
        DEPLOY_ROOT="$(dirname "$DEPLOY_ROOT")"
    elif [ -f "$(dirname "$(dirname "$DEPLOY_ROOT")")/venv/bin/activate" ]; then
        DEPLOY_ROOT="$(dirname "$(dirname "$DEPLOY_ROOT")")"
    fi
fi
echo "[bootstrap] DEPLOY_ROOT=$DEPLOY_ROOT  PWD=$PWD"
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }

REPO="$DEPLOY_ROOT/R-PFN"
UWYK="$DEPLOY_ROOT/external/uwyk"
CKPT="$UWYK/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0"
CAUSALPFN="$DEPLOY_ROOT/external/causalpfn"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

export UWYK CKPT CAUSALPFN

echo "[$(date '+%Y-%m-%d %H:%M:%S')] launching probe"
python -u "$REPO/benchmarks/uwyk_table1/probe_kmeans.py"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] done"
