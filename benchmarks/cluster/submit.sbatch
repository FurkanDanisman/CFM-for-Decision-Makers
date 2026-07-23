#!/bin/bash
#SBATCH --job-name=rpfn-table3
#SBATCH --output=logs/task_%A_%a.out
#SBATCH --error=logs/task_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --array=0-499%50
# ── Array layout ──────────────────────────────────────────────────────────
# 5 datasets × 100 realizations = 500 tasks. Matches paper protocol exactly.
# NO subsampling — full train context, full test queries.
# %50 caps concurrency at 50 running jobs; adjust to your queue limit.
# Runtime per task: highly dataset-dependent (~5 min IHDP → ~2 hr CPS full).

set -euo pipefail
mkdir -p logs results

# ── Configuration (edit these) ────────────────────────────────────────────
DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"        # where the rsync landed
REPO="$DEPLOY_ROOT/R-PFN"
DOPFN="$DEPLOY_ROOT/external/dopfn"
UWYK="$DEPLOY_ROOT/external/uwyk"
CAUSALPFN="$DEPLOY_ROOT/external/causalpfn"
CHECKPOINT="$REPO/checkpoints/step_50000_final.pt"
UWYK_CKPT="$UWYK/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0"
UWYK_BASELINE_CKPT="${UWYK_BASELINE_CKPT:-$UWYK/experiments/checkpoints/no_graph_conditioning/unconditional}"
OUTDIR="$DEPLOY_ROOT/results"

# Toggle: emit pehe_uwyk_baseline column if the checkpoint exists.
# Override with EMIT_UWYK_BASELINE=0 to skip.
EMIT_UWYK_BASELINE="${EMIT_UWYK_BASELINE:-1}"

N_REAL="${N_REAL:-100}"                    # paper: 100 realizations
WORKERS="${SLURM_CPUS_PER_TASK:-16}"

DATASETS=(IHDP ACIC CPS PSID PSIDbal)

# ── Decode array index → (dataset, realization) ──────────────────────────
TASK=${SLURM_ARRAY_TASK_ID:-0}
DS_IDX=$(( TASK / N_REAL ))
REAL_IDX=$(( TASK % N_REAL ))
DATASET="${DATASETS[$DS_IDX]}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] TASK=$TASK  DATASET=$DATASET  REAL=$REAL_IDX  WORKERS=$WORKERS"

# ── Activate env & run ────────────────────────────────────────────────────
source "$DEPLOY_ROOT/venv/bin/activate"
# Main process (torch inference) gets all threads. MALC workers set to 1 in _init_worker.
export OMP_NUM_THREADS="$WORKERS"
export MKL_NUM_THREADS="$WORKERS"
export OPENBLAS_NUM_THREADS="$WORKERS"
export PYTHONUNBUFFERED=1

cd "$DOPFN"   # Do-PFN needs to run from its own root (for datasets/ import)

BASELINE_ARGS=()
if [ "$EMIT_UWYK_BASELINE" = "1" ] && [ -f "$UWYK_BASELINE_CKPT/best_model.pt" ]; then
    BASELINE_ARGS=(--uwyk-baseline-ckpt-dir "$UWYK_BASELINE_CKPT")
    echo "  UWYK_BASELINE_CKPT=$UWYK_BASELINE_CKPT"
fi

python "$REPO/benchmarks/run_one.py" \
    --dataset       "$DATASET" \
    --realization   "$REAL_IDX" \
    --outdir        "$OUTDIR" \
    --repo          "$REPO" \
    --dopfn         "$DOPFN" \
    --uwyk          "$UWYK" \
    --causalpfn     "$CAUSALPFN" \
    --checkpoint    "$CHECKPOINT" \
    --uwyk-ckpt-dir "$UWYK_CKPT" \
    "${BASELINE_ARGS[@]}" \
    --workers       "$WORKERS"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE task=$TASK"
