#!/bin/bash
#SBATCH --job-name=rpfn-rho-backfill
#SBATCH --output=logs_rho_backfill/task_%A_%a.out
#SBATCH --error=logs_rho_backfill/task_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-4%5
# ─────────────────────────────────────────────────────────────────────────
# One array task per dataset. Each task backfills per-query ρ and Var(τ|X)
# into every Table-3 npz file for that dataset. Payload is tiny (a few
# hundred floats per npz), so re-running is cheap.
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail
mkdir -p logs_rho_backfill

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
[ ! -d "$REPO/benchmarks" ] && REPO="$DEPLOY_ROOT"
DOPFN="${DOPFN:-$DEPLOY_ROOT/external/dopfn}"
CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
CHECKPOINT="${CHECKPOINT:-$REPO/checkpoints/step_50000_final.pt}"
RESULTS_DIR="${RESULTS_DIR:-$DEPLOY_ROOT/results}"

DATASETS=(IHDP ACIC CPS PSID PSIDbal)
TASK=${SLURM_ARRAY_TASK_ID:-0}
DATASET="${DATASETS[$TASK]}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] TASK=$TASK  DATASET=$DATASET"
echo "  REPO=$REPO   DOPFN=$DOPFN   CAUSALPFN=$CAUSALPFN"
echo "  CHECKPOINT=$CHECKPOINT   RESULTS_DIR=$RESULTS_DIR"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

python "$REPO/benchmarks/backfill_rho_v.py" \
    --results-dir "$RESULTS_DIR" \
    --repo        "$REPO" \
    --checkpoint  "$CHECKPOINT" \
    --dopfn       "$DOPFN" \
    --causalpfn   "$CAUSALPFN" \
    --datasets    "$DATASET"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE task=$TASK"
