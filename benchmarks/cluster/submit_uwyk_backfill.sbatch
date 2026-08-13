#!/bin/bash
#SBATCH --job-name=rpfn-t3-uwyk-noanc
#SBATCH --output=logs_uwyk_noanc/task_%A_%a.out
#SBATCH --error=logs_uwyk_noanc/task_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-499%50
# ─────────────────────────────────────────────────────────────────────────────
# Backfill UWYK No-Ancestral (ancestral ckpt + zero adjacency at inference)
# into existing RealCause Table 3 npz files WITHOUT touching Do-PFN /
# UWYK-Ancestral / UWYK-Baseline / OURS fields.
#
# Array layout matches submit.sbatch: 5 datasets × 100 realizations = 500 tasks;
# ACIC has only 10 realizations, so the run_one.py guard skips r>=10 for ACIC.
#
# Usage on killarney:
#   sbatch --account=aip-rgrosse R-PFN/benchmarks/cluster/submit_uwyk_noanc_backfill.sbatch
#
# Env overrides:
#   DEPLOY_ROOT — root that contains R-PFN/, external/, venv/, results/
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
mkdir -p logs_uwyk_noanc results

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
REPO="$DEPLOY_ROOT/R-PFN"
DOPFN="$DEPLOY_ROOT/external/dopfn"
UWYK="$DEPLOY_ROOT/external/uwyk"
CAUSALPFN="$DEPLOY_ROOT/external/causalpfn"
CHECKPOINT="$REPO/checkpoints/step_50000_final.pt"
UWYK_CKPT="$UWYK/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0"
OUTDIR="${OUTDIR:-$DEPLOY_ROOT/results}"

if [ ! -f "$UWYK_CKPT/best_model.pt" ]; then
    echo "[error] UWYK_CKPT ($UWYK_CKPT) missing best_model.pt" >&2
    exit 3
fi

N_REAL="${N_REAL:-100}"
WORKERS="${SLURM_CPUS_PER_TASK:-8}"
DATASETS=(IHDP ACIC CPS PSID PSIDbal)

# Which UWYK variant(s) to backfill. Space-separated list, e.g. "noanc anc".
# Defaults to noanc to preserve the historic behavior of this sbatch.
VARIANTS="${VARIANTS:-noanc}"
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-0}"       # 1 → overwrite existing pehe_uwyk_* fields
UWYK_MATCH_PAPER="${UWYK_MATCH_PAPER:-0}"     # 1 → paper-protocol PSID (fixed seed=42 balance)

TASK=${SLURM_ARRAY_TASK_ID:-0}
DS_IDX=$(( TASK / N_REAL ))
REAL_IDX=$(( TASK % N_REAL ))
DATASET="${DATASETS[$DS_IDX]}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] TASK=$TASK  DATASET=$DATASET  REAL=$REAL_IDX  VARIANTS=$VARIANTS  FORCE=$FORCE_RECOMPUTE  MATCH_PAPER=$UWYK_MATCH_PAPER"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="$WORKERS"
export MKL_NUM_THREADS="$WORKERS"
export OPENBLAS_NUM_THREADS="$WORKERS"
export PYTHONUNBUFFERED=1

cd "$DOPFN"

FORCE_FLAG=""
[ "$FORCE_RECOMPUTE" = "1" ] && FORCE_FLAG="--force-uwyk-recompute"
MATCH_FLAG=""
[ "$UWYK_MATCH_PAPER" = "1" ] && MATCH_FLAG="--uwyk-match-paper"

for VARIANT in $VARIANTS; do
    case "$VARIANT" in
        noanc) MODE_FLAG="--only-uwyk-noanc" ;;
        anc)   MODE_FLAG="--only-uwyk-anc" ;;
        *) echo "[error] unknown VARIANT=$VARIANT (expected noanc|anc)" >&2; exit 2 ;;
    esac
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] --- variant=$VARIANT ---"
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
        --workers       "$WORKERS" \
        $MODE_FLAG $FORCE_FLAG $MATCH_FLAG
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE task=$TASK"
