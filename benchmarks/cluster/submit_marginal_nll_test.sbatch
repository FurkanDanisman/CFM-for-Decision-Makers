#!/bin/bash
#SBATCH --job-name=rpfn-marg-nll
#SBATCH --output=logs_marg_nll/task_%A.out
#SBATCH --error=logs_marg_nll/task_%A.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
mkdir -p logs_marg_nll

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
[ ! -d "$REPO/benchmarks" ] && REPO="$DEPLOY_ROOT"
DOPFN="${DOPFN:-$DEPLOY_ROOT/external/dopfn}"
UWYK_SRC="${UWYK_SRC:-$DEPLOY_ROOT/external/uwyk/src}"
UWYK_CKPT_DIR="${UWYK_CKPT_DIR:-$DEPLOY_ROOT/external/uwyk/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0}"
CHECKPOINT50="${CHECKPOINT50:-$REPO/checkpoints/step_50000_final.pt}"
CHECKPOINT10="${CHECKPOINT10:-$REPO/checkpoints_dopfn/step_50000_final.pt}"
K="${K:-20}"
N_CONTEXT="${N_CONTEXT:-200}"
N_TEST="${N_TEST:-50}"
OUT="${OUT:-$DEPLOY_ROOT/marginal_nll_test.png}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] REPO=$REPO"
echo "  K=$K   N_CONTEXT=$N_CONTEXT   N_TEST=$N_TEST   OUT=$OUT"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

python "$REPO/benchmarks/empirical_tests/marginal_nll_test.py" \
    --repo           "$REPO" \
    --checkpoint50   "$CHECKPOINT50" \
    --checkpoint10   "$CHECKPOINT10" \
    --uwyk-src       "$UWYK_SRC" \
    --uwyk-ckpt-dir  "$UWYK_CKPT_DIR" \
    --dopfn          "$DOPFN" \
    --K              "$K" \
    --N-context      "$N_CONTEXT" \
    --N-test         "$N_TEST" \
    --out            "$OUT"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE"
