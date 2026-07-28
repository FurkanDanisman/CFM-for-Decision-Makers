#!/bin/bash
#SBATCH --job-name=rpfn-semireal
#SBATCH --output=logs_semi_real/task_%A_%a.out
#SBATCH --error=logs_semi_real/task_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-9%10
# ── Array layout ──────────────────────────────────────────────────────────
# 2 datasets × N_SEEDS (default 5) = 10 tasks.
# Datasets: sales, law_race  (Do-PFN's semi-real benchmarks).
#
#   DS_IDX  = TASK / N_SEEDS
#   SEED    = TASK % N_SEEDS
#
# Two invocations recommended (paired with two `CHECKPOINT` values):
#   pass 1 (fn=50, baselines + Ours) →
#     CHECKPOINT=.../checkpoints/step_50000_final.pt \
#     OUTDIR=.../results_semi_real \
#     sbatch --account=... submit_semi_real.sbatch
#
#   pass 2 (fn=10, Ours-only) →
#     CHECKPOINT=.../checkpoints_dopfn/step_50000_final.pt \
#     OUTDIR=.../results_semi_real_fn10 \
#     OURS_ONLY=1 \
#     sbatch --account=... submit_semi_real.sbatch

set -euo pipefail
mkdir -p logs_semi_real

# ── Path layout (all env-overridable) ───────────────────────────────────────
DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
if [ ! -d "$REPO/benchmarks" ]; then REPO="$DEPLOY_ROOT"; fi
DOPFN="${DOPFN:-$DEPLOY_ROOT/external/dopfn}"
UWYK_SRC="${UWYK_SRC:-$DEPLOY_ROOT/external/uwyk/src}"
UWYK_CKPT_DIR="${UWYK_CKPT_DIR:-$DEPLOY_ROOT/external/uwyk/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0}"
CHECKPOINT="${CHECKPOINT:-$REPO/checkpoints/step_50000_final.pt}"
OUTDIR="${OUTDIR:-$DEPLOY_ROOT/results_semi_real}"
VENV="${VENV:-$DEPLOY_ROOT/venv}"
[ ! -f "$VENV/bin/activate" ] && [ -f "$DEPLOY_ROOT/.venv/bin/activate" ] && VENV="$DEPLOY_ROOT/.venv"

DATASETS=(sales law_race)
N_SEEDS="${N_SEEDS:-5}"
OURS_ONLY_FLAG=""
if [ "${OURS_ONLY:-0}" = "1" ]; then OURS_ONLY_FLAG="--ours-only"; fi

TASK=${SLURM_ARRAY_TASK_ID:-0}
DS_IDX=$(( TASK / N_SEEDS ))
SEED=$(( TASK % N_SEEDS ))
DATASET="${DATASETS[$DS_IDX]}"
WORKERS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p "$OUTDIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] TASK=$TASK  DATASET=$DATASET  SEED=$SEED  OURS_ONLY=${OURS_ONLY:-0}"
echo "  REPO=$REPO"
echo "  DOPFN=$DOPFN"
echo "  UWYK_SRC=$UWYK_SRC   UWYK_CKPT_DIR=$UWYK_CKPT_DIR"
echo "  CHECKPOINT=$CHECKPOINT"
echo "  OUTDIR=$OUTDIR"

source "$VENV/bin/activate"
export OMP_NUM_THREADS="$WORKERS"
export MKL_NUM_THREADS="$WORKERS"
export OPENBLAS_NUM_THREADS="$WORKERS"
export PYTHONUNBUFFERED=1

python "$REPO/benchmarks/semi_real/run_one.py" \
    --dataset        "$DATASET" \
    --seed           "$SEED" \
    --outdir         "$OUTDIR" \
    --repo           "$REPO" \
    --dopfn          "$DOPFN" \
    --uwyk-src       "$UWYK_SRC" \
    --uwyk-ckpt-dir  "$UWYK_CKPT_DIR" \
    --checkpoint     "$CHECKPOINT" \
    --workers        "$WORKERS" \
    $OURS_ONLY_FLAG

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE task=$TASK"
