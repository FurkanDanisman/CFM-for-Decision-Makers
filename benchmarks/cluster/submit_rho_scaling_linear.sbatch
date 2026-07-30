#!/bin/bash
#SBATCH --job-name=rpfn-rho-linear
#SBATCH --output=logs_rho_linear/task_%A_%a.out
#SBATCH --error=logs_rho_linear/task_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-5%6

set -euo pipefail
mkdir -p logs_rho_linear

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
[ ! -d "$REPO/benchmarks" ] && REPO="$DEPLOY_ROOT"
UWYK_SRC="${UWYK_SRC:-$DEPLOY_ROOT/external/uwyk/src}"
UWYK_CKPT_DIR="${UWYK_CKPT_DIR:-$DEPLOY_ROOT/external/uwyk/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0}"
CHECKPOINT50="${CHECKPOINT50:-$REPO/checkpoints/step_50000_final.pt}"
K="${K:-15}"
N_CONTEXT="${N_CONTEXT:-200}"
N_TEST="${N_TEST:-50}"
X_DIM="${X_DIM:-5}"
SIGMA_EPS="${SIGMA_EPS:-1.0}"
OUT="${OUT:-$DEPLOY_ROOT/rho_scaling_linear.png}"
RHO_INDEX=${SLURM_ARRAY_TASK_ID:-0}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] RHO_INDEX=$RHO_INDEX  K=$K  N=$N_CONTEXT  x_dim=$X_DIM"
echo "  REPO=$REPO  OUT=$OUT"

source "$DEPLOY_ROOT/venv/bin/activate"
export UWYK_SRC="$UWYK_SRC"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

python "$REPO/benchmarks/empirical_tests/rho_scaling_linear.py" \
    --repo           "$REPO" \
    --checkpoint50   "$CHECKPOINT50" \
    --uwyk-src       "$UWYK_SRC" \
    --uwyk-ckpt-dir  "$UWYK_CKPT_DIR" \
    --K              "$K" \
    --N-context      "$N_CONTEXT" \
    --N-test         "$N_TEST" \
    --x-dim          "$X_DIM" \
    --sigma-eps      "$SIGMA_EPS" \
    --out            "$OUT" \
    --rho-index      "$RHO_INDEX"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE task=$RHO_INDEX"
