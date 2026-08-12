#!/bin/bash
#SBATCH --job-name=rpfn-acic-l2
#SBATCH --output=logs_acic_l2/task_%A_%a.out
#SBATCH --error=logs_acic_l2/task_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-9

# Per-realization density L2 evaluation on ACIC 2016 (10 realizations).
# Reuses l2_ihdp's method runners; only truth + dataset load differ.
# Output: $OUT.r{r:03d}.npz per task; run summary_acic.py after.

set -euo pipefail
mkdir -p logs_acic_l2

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
[ ! -d "$REPO/benchmarks" ] && REPO="$DEPLOY_ROOT"
DOPFN="${DOPFN:-$DEPLOY_ROOT/external/dopfn}"
CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
UWYK_SRC="${UWYK_SRC:-$DEPLOY_ROOT/external/uwyk/src}"
UWYK_CKPT_DIR="${UWYK_CKPT_DIR:-$DEPLOY_ROOT/external/uwyk/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0}"
CHECKPOINT50="${CHECKPOINT50:-$REPO/checkpoints/step_50000_final.pt}"
CHECKPOINT_DOPFN_BB="${CHECKPOINT_DOPFN_BB:-$DEPLOY_ROOT/checkpoints_dopfn_backbone_j10/step_200000.pt}"
N_CONTEXT="${N_CONTEXT:-0}"
MALC_B="${MALC_B:-500}"
MALC_MAX_K="${MALC_MAX_K:-3}"
N_EVAL="${N_EVAL:-200}"
UWYK_N_SAMPLES="${UWYK_N_SAMPLES:-1024}"
METHODS="${METHODS:-ours_fn50,uwyk_noanc,uwyk_anc,dopfn,ours_dopfn_bb}"
OUT="${OUT:-$DEPLOY_ROOT/l2_acic/out}"
ACIC_CACHE_DIR="${ACIC_CACHE_DIR:-}"

REALIZATION="${SLURM_ARRAY_TASK_ID:-0}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] r=$REALIZATION  methods=$METHODS"
echo "  REPO=$REPO  OUT=$OUT"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

mkdir -p "$(dirname "$OUT")"

python "$REPO/benchmarks/l2_acic/eval_realization.py" \
    --realization        "$REALIZATION" \
    --out                "$OUT" \
    --repo               "$REPO" \
    --causalpfn          "$CAUSALPFN" \
    --dopfn              "$DOPFN" \
    --uwyk-src           "$UWYK_SRC" \
    --uwyk-ckpt-dir      "$UWYK_CKPT_DIR" \
    --checkpoint50       "$CHECKPOINT50" \
    --checkpoint-dopfn-bb "$CHECKPOINT_DOPFN_BB" \
    --n-context          "$N_CONTEXT" \
    --malc-B             "$MALC_B" \
    --malc-max-K         "$MALC_MAX_K" \
    --n-eval             "$N_EVAL" \
    --uwyk-n-samples     "$UWYK_N_SAMPLES" \
    --methods            "$METHODS" \
    --acic-cache-dir     "$ACIC_CACHE_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE r=$REALIZATION"
