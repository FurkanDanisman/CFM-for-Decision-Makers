#!/bin/bash
#SBATCH --job-name=rpfn-t3-rerun
#SBATCH --output=logs_t3_rerun/task_%A_%a.out
#SBATCH --error=logs_t3_rerun/task_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --array=100-399%50

# Table 3 re-run: fn=50 raw mean + UWYK-anc + UWYK-noanc ONLY, for
# ACIC / CPS / PSID (NO IHDP, NO PSIDbal, NO baseline, NO MALC).
#
# Array slice (matches submit.sbatch's index scheme: TASK = DS_IDX * 100 + REAL):
#   100..199  → ACIC r=0..99
#   200..299  → CPS  r=0..99
#   300..399  → PSID r=0..99
#
# Everything else — invocation path, workers, memory, env — is identical
# to submit.sbatch. Only these two behavior toggles differ:
#   EMIT_UWYK_BASELINE=0   → skip pehe_uwyk_baseline column
#   --skip-malc            → skip MALC pool; ours writes only pehe_ours_mean
#
# Output lands in a fresh dir so we don't overwrite the existing Table 3
# npz files. Change OUTDIR if you want to merge into results/.

set -euo pipefail
mkdir -p logs_t3_rerun

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
# Auto-climb if submitted from inside R-PFN (venv/external live one level up).
for _ in 1 2 3; do
    [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
    DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
done
REPO="$DEPLOY_ROOT/R-PFN"
DOPFN="$DEPLOY_ROOT/external/dopfn"
UWYK="$DEPLOY_ROOT/external/uwyk"
CAUSALPFN="$DEPLOY_ROOT/external/causalpfn"
CHECKPOINT="${CHECKPOINT:-$REPO/checkpoints/step_50000_final.pt}"
UWYK_CKPT="$UWYK/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0"
OUTDIR="${OUTDIR:-$DEPLOY_ROOT/results_table3_rerun}"
mkdir -p "$OUTDIR"

N_REAL="${N_REAL:-100}"
WORKERS="${SLURM_CPUS_PER_TASK:-8}"

DATASETS=(IHDP ACIC CPS PSID PSIDbal)

TASK=${SLURM_ARRAY_TASK_ID:-0}
DS_IDX=$(( TASK / N_REAL ))
REAL_IDX=$(( TASK % N_REAL ))
DATASET="${DATASETS[$DS_IDX]}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] TASK=$TASK  DATASET=$DATASET  REAL=$REAL_IDX  WORKERS=$WORKERS"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="$WORKERS"
export MKL_NUM_THREADS="$WORKERS"
export OPENBLAS_NUM_THREADS="$WORKERS"
export PYTHONUNBUFFERED=1

cd "$DOPFN"

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
    --skip-malc \
    --workers       "$WORKERS"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE task=$TASK"
