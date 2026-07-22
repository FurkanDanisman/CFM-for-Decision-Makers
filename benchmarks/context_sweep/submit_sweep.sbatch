#!/bin/bash
#SBATCH --job-name=rpfn-ctxsweep
#SBATCH --output=logs_sweep/task_%A_%a.out
#SBATCH --error=logs_sweep/task_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-119%50
# ── Array layout ─────────────────────────────────────────────────────────────
# Alliance Canada limits array size to 1000. To cover 500 SCMs × 6 N × 2 sources
# = 6000 SCM-N-source configurations, we batch N_SCMS_PER_TASK SCMs per SLURM
# task. Total tasks = 2 sources × 6 contexts × N_BATCHES.
#
# Layout:
#   src_idx   = TASK / (N_CONTEXTS × N_BATCHES)   ∈ {0=prior, 1=poly}
#   remain    = TASK % (N_CONTEXTS × N_BATCHES)
#   N_idx     = remain / N_BATCHES                ∈ {0..N_CONTEXTS-1}
#   batch_idx = remain % N_BATCHES                ∈ {0..N_BATCHES-1}
#   seeds     = batch_idx * N_SCMS_PER_TASK .. (batch_idx+1) * N_SCMS_PER_TASK - 1

set -euo pipefail
mkdir -p logs_sweep results_sweep

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
REPO="$DEPLOY_ROOT/R-PFN"
DOPFN="$DEPLOY_ROOT/external/dopfn"
UWYK_SRC="$DEPLOY_ROOT/external/uwyk/src"
CAUSALPFN="$DEPLOY_ROOT/external/causalpfn"
CHECKPOINT="$REPO/checkpoints/step_50000_final.pt"
OUTDIR="$DEPLOY_ROOT/results_sweep"

N_SCMS_TOTAL="${N_SCMS_TOTAL:-500}"
N_SCMS_PER_TASK="${N_SCMS_PER_TASK:-50}"
N_BATCHES=$(( N_SCMS_TOTAL / N_SCMS_PER_TASK ))    # 500 / 50 = 10
CONTEXTS=(50 250 500 1000 5000 10000)
N_CONTEXTS=${#CONTEXTS[@]}                          # 6
SOURCES=(prior poly)
N_SOURCES=${#SOURCES[@]}                            # 2
WORKERS="${SLURM_CPUS_PER_TASK:-8}"

# Total tasks = 2 * 6 * 10 = 120 → array=0-119
TASK=${SLURM_ARRAY_TASK_ID:-0}
SRC_IDX=$(( TASK / (N_CONTEXTS * N_BATCHES) ))
REMAIN=$(( TASK % (N_CONTEXTS * N_BATCHES) ))
N_IDX=$(( REMAIN / N_BATCHES ))
BATCH_IDX=$(( REMAIN % N_BATCHES ))
SOURCE="${SOURCES[$SRC_IDX]}"
N_CTX="${CONTEXTS[$N_IDX]}"
SEED_START=$(( BATCH_IDX * N_SCMS_PER_TASK ))
SEED_END=$(( SEED_START + N_SCMS_PER_TASK - 1 ))

echo "[$(date '+%Y-%m-%d %H:%M:%S')] TASK=$TASK  SOURCE=$SOURCE  N=$N_CTX  seeds=[$SEED_START..$SEED_END]"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="$WORKERS"
export MKL_NUM_THREADS="$WORKERS"
export OPENBLAS_NUM_THREADS="$WORKERS"
export PYTHONUNBUFFERED=1

cd "$DOPFN"
for SEED in $(seq "$SEED_START" "$SEED_END"); do
    OUT_FILE="$OUTDIR/${SOURCE}_seed$(printf '%04d' "$SEED")_N$(printf '%06d' "$N_CTX").npz"
    if [ -f "$OUT_FILE" ]; then
        echo "[skip] $OUT_FILE exists"
        continue
    fi
    python "$REPO/benchmarks/context_sweep/run_one.py" \
        --source     "$SOURCE" \
        --seed       "$SEED" \
        --n-context  "$N_CTX" \
        --outdir     "$OUTDIR" \
        --repo       "$REPO" \
        --uwyk-src   "$UWYK_SRC" \
        --causalpfn  "$CAUSALPFN" \
        --checkpoint "$CHECKPOINT" \
        --workers    "$WORKERS" || echo "[warn] seed=$SEED failed, continuing"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE task=$TASK"
