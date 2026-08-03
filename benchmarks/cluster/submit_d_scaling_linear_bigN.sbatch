#!/bin/bash
#SBATCH --job-name=rpfn-d-bigN
#SBATCH --output=logs_d_bigN/task_%A_%a.out
#SBATCH --error=logs_d_bigN/task_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-4%5

set -euo pipefail
mkdir -p logs_d_bigN

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
[ ! -d "$REPO/benchmarks" ] && REPO="$DEPLOY_ROOT"
UWYK_SRC="${UWYK_SRC:-$DEPLOY_ROOT/external/uwyk/src}"
UWYK_CKPT_DIR="${UWYK_CKPT_DIR:-$DEPLOY_ROOT/external/uwyk/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0}"
CHECKPOINT50="${CHECKPOINT50:-$REPO/checkpoints/step_50000_final.pt}"
K="${K:-15}"
N_CONTEXT="${N_CONTEXT:-10000}"
N_TEST="${N_TEST:-50}"
SIGMA_EPS="${SIGMA_EPS:-1.0}"
OUT="${OUT:-$DEPLOY_ROOT/d_scaling_linear_bigN.png}"
D_INDEX=${SLURM_ARRAY_TASK_ID:-0}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] D_INDEX=$D_INDEX  K=$K  N=$N_CONTEXT"

source "$DEPLOY_ROOT/venv/bin/activate"
export UWYK_SRC="$UWYK_SRC"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

python "$REPO/benchmarks/empirical_tests/d_scaling_linear.py" \
    --repo           "$REPO" \
    --checkpoint50   "$CHECKPOINT50" \
    --uwyk-src       "$UWYK_SRC" \
    --uwyk-ckpt-dir  "$UWYK_CKPT_DIR" \
    --K              "$K" \
    --N-context      "$N_CONTEXT" \
    --N-test         "$N_TEST" \
    --sigma-eps      "$SIGMA_EPS" \
    --out            "$OUT" \
    --d-index        "$D_INDEX"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE task=$D_INDEX"
