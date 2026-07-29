#!/bin/bash
#SBATCH --job-name=rpfn-rho-test
#SBATCH --output=logs_rho_test/task_%A_%a.out
#SBATCH --error=logs_rho_test/task_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-5%6
# ────────────────────────────────────────────────────────────────────────
# Test 1 as an array: one task per ρ ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 0.95}.
# Each task runs K SCMs at its assigned ρ with workers=8 MALC parallel,
# writes rho_scaling_test.png.rho<idx>.npz. Resume-aware: if a shard
# already contains N rows, only computes K−N remaining SCMs.
#
# Wallclock estimate: 15 SCMs × ~5 min = 75 min per task, comfortable
# inside the 2h wall.
#
# After all six tasks finish, run once:
#   python R-PFN/benchmarks/empirical_tests/rho_scaling_test.py \
#       ... --plot --out $ROOT/rho_scaling_test.png
# to aggregate the shards into the final theory-vs-empirics figure.
# ────────────────────────────────────────────────────────────────────────
set -euo pipefail
mkdir -p logs_rho_test

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
[ ! -d "$REPO/benchmarks" ] && REPO="$DEPLOY_ROOT"
DOPFN="${DOPFN:-$DEPLOY_ROOT/external/dopfn}"
UWYK_SRC="${UWYK_SRC:-$DEPLOY_ROOT/external/uwyk/src}"
UWYK_CKPT_DIR="${UWYK_CKPT_DIR:-$DEPLOY_ROOT/external/uwyk/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0}"
CHECKPOINT50="${CHECKPOINT50:-$REPO/checkpoints/step_50000_final.pt}"
CHECKPOINT10="${CHECKPOINT10:-$REPO/checkpoints_dopfn/step_50000_final.pt}"
K="${K:-15}"
N_CONTEXT="${N_CONTEXT:-200}"
N_TEST="${N_TEST:-50}"
OUT="${OUT:-$DEPLOY_ROOT/rho_scaling_test.png}"
RHO_INDEX=${SLURM_ARRAY_TASK_ID:-0}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] TASK=$RHO_INDEX  (ρ_index=$RHO_INDEX)"
echo "  REPO=$REPO   OUT=$OUT   K=$K"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

python "$REPO/benchmarks/empirical_tests/rho_scaling_test.py" \
    --repo           "$REPO" \
    --checkpoint50   "$CHECKPOINT50" \
    --checkpoint10   "$CHECKPOINT10" \
    --uwyk-src       "$UWYK_SRC" \
    --uwyk-ckpt-dir  "$UWYK_CKPT_DIR" \
    --dopfn          "$DOPFN" \
    --K              "$K" \
    --N-context      "$N_CONTEXT" \
    --N-test         "$N_TEST" \
    --out            "$OUT" \
    --rho-index      "$RHO_INDEX"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE task=$RHO_INDEX"
