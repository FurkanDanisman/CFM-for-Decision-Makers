#!/bin/bash
#SBATCH --job-name=rpfn-ours-only
#SBATCH --output=logs/ours_only_%A_%a.out
#SBATCH --error=logs/ours_only_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --array=0-499%50
# ── Array layout ──────────────────────────────────────────────────────────
# Same 5 datasets × 100 realizations layout as submit.sbatch, but runs ONLY
# the OURS pipeline (no Do-PFN, no UWYK). Use to sweep OURS point-estimate
# variants at large MALC_B without paying baseline compute again.
# %50 caps concurrency at 50; adjust as needed.

set -euo pipefail
mkdir -p logs

# ── Configuration (env-overridable) ──────────────────────────────────────
DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
DOPFN="${DOPFN:-$DEPLOY_ROOT/external/dopfn}"
UWYK="${UWYK:-$DEPLOY_ROOT/external/uwyk}"
CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
CHECKPOINT="${CHECKPOINT:-$REPO/checkpoints/step_50000_final.pt}"
# UWYK ckpt dir is required by the argparser but unused in --only-ours mode.
UWYK_CKPT="${UWYK_CKPT:-$UWYK/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0}"
OUTDIR="${OUTDIR:-$DEPLOY_ROOT/results_ours_only}"
MALC_B="${MALC_B:-500}"
MALC_MAX_K="${MALC_MAX_K:-3}"
BACKBONE="${BACKBONE:-ipfn}"                # ipfn | dopfn_bb
Y_POWER_TRANSFORM="${Y_POWER_TRANSFORM:-0}"  # 1 → apply Yeo-Johnson on Y at inference
Y_WINSORIZE="${Y_WINSORIZE:-0}"              # e.g. 5 → use (5%, 95%) Y quantiles for scaling
SKIP_MALC="${SKIP_MALC:-0}"                  # 1 → skip MALC pool entirely (raw-mean only)

N_REAL="${N_REAL:-100}"
WORKERS="${SLURM_CPUS_PER_TASK:-8}"

DATASETS=(IHDP ACIC CPS PSID PSIDbal)

TASK=${SLURM_ARRAY_TASK_ID:-0}
DS_IDX=$(( TASK / N_REAL ))
REAL_IDX=$(( TASK % N_REAL ))
DATASET="${DATASETS[$DS_IDX]}"

mkdir -p "$OUTDIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] TASK=$TASK  DATASET=$DATASET  REAL=$REAL_IDX  "\
"MALC_B=$MALC_B  MALC_MAX_K=$MALC_MAX_K  OUTDIR=$OUTDIR"

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
    --malc-B        "$MALC_B" \
    --malc-max-K    "$MALC_MAX_K" \
    --backbone      "$BACKBONE" \
    $( [ "$Y_POWER_TRANSFORM" = "1" ] && echo "--y-power-transform" ) \
    --y-winsorize   "$Y_WINSORIZE" \
    $( [ "$SKIP_MALC" = "1" ] && echo "--skip-malc" ) \
    --workers       "$WORKERS" \
    --only-ours

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE task=$TASK"
