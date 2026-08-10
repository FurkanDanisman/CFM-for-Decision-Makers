#!/bin/bash
#SBATCH --job-name=rpfn-ihdp-l2
#SBATCH --output=logs_ihdp_l2/task_%A_%a.out
#SBATCH --error=logs_ihdp_l2/task_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-99%20

# Per-realization L2 evaluation on IHDP: one task per realization,
# throttled to 20 concurrent tasks. Adjust --array=0-99%N to change
# concurrency. Outputs $OUT.r{r:03d}.npz per task; run
# aggregate_and_plot.py after the array completes.

set -euo pipefail
mkdir -p logs_ihdp_l2

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
[ ! -d "$REPO/benchmarks" ] && REPO="$DEPLOY_ROOT"
DOPFN="${DOPFN:-$DEPLOY_ROOT/external/dopfn}"
CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
UWYK_SRC="${UWYK_SRC:-$DEPLOY_ROOT/external/uwyk/src}"
UWYK_CKPT_DIR="${UWYK_CKPT_DIR:-$DEPLOY_ROOT/external/uwyk/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0}"
CHECKPOINT50="${CHECKPOINT50:-$REPO/checkpoints/step_50000_final.pt}"
CHECKPOINT10="${CHECKPOINT10:-$REPO/checkpoints_dopfn/step_50000_final.pt}"
N_CONTEXT="${N_CONTEXT:-0}"                    # 0 = use full training set
MALC_B="${MALC_B:-60}"
MALC_MAX_K="${MALC_MAX_K:-3}"
N_EVAL="${N_EVAL:-200}"
UWYK_N_SAMPLES="${UWYK_N_SAMPLES:-1024}"
METHODS="${METHODS:-ours_fn50,ours_fn10,uwyk_noanc,dopfn}"
OUT="${OUT:-$DEPLOY_ROOT/l2_ihdp/out}"

REALIZATION="${SLURM_ARRAY_TASK_ID:-0}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] r=$REALIZATION  methods=$METHODS"
echo "  REPO=$REPO  OUT=$OUT"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

mkdir -p "$(dirname "$OUT")"

python "$REPO/benchmarks/l2_ihdp/eval_realization.py" \
    --realization    "$REALIZATION" \
    --out            "$OUT" \
    --repo           "$REPO" \
    --causalpfn      "$CAUSALPFN" \
    --dopfn          "$DOPFN" \
    --uwyk-src       "$UWYK_SRC" \
    --uwyk-ckpt-dir  "$UWYK_CKPT_DIR" \
    --checkpoint50   "$CHECKPOINT50" \
    --checkpoint10   "$CHECKPOINT10" \
    --n-context      "$N_CONTEXT" \
    --malc-B         "$MALC_B" \
    --malc-max-K     "$MALC_MAX_K" \
    --n-eval         "$N_EVAL" \
    --uwyk-n-samples "$UWYK_N_SAMPLES" \
    --methods        "$METHODS"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE r=$REALIZATION"
