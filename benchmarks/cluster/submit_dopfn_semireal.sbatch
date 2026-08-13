#!/bin/bash
#SBATCH --job-name=rpfn-semireal
#SBATCH --output=logs_semireal/task_%A_%a.out
#SBATCH --error=logs_semireal/task_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-3

# Runs OURS on Do-PFN's semi-real datasets (sales, law_race).
# Array: 4 tasks = 2 datasets × 2 variants (baseline, log-Y).
#
# Env vars (all optional):
#   BACKBONE=dopfn_bb (default)
#   CHECKPOINT=$DEPLOY_ROOT/checkpoints_dopfn_backbone_j10/step_200000.pt (default)
#   OUTDIR=$DEPLOY_ROOT/results_dopfn_semireal (default)
#   SPLIT_NUMBER=1
#   N_SPLITS=5
#   ALSO_DOPFN=1  → also run DoPFN's own DoPFNRegressor for paired reference

set -euo pipefail
mkdir -p logs_semireal

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
[ ! -d "$REPO/benchmarks" ] && REPO="$DEPLOY_ROOT"
DOPFN="${DOPFN:-$DEPLOY_ROOT/external/dopfn}"
CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
CHECKPOINT="${CHECKPOINT:-$DEPLOY_ROOT/checkpoints_dopfn_backbone_j10/step_200000.pt}"
BACKBONE="${BACKBONE:-dopfn_bb}"
OUTDIR="${OUTDIR:-$DEPLOY_ROOT/results_dopfn_semireal}"
SPLIT_NUMBER="${SPLIT_NUMBER:-1}"
N_SPLITS="${N_SPLITS:-5}"
ALSO_DOPFN_FLAG=""
[ "${ALSO_DOPFN:-0}" = "1" ] && ALSO_DOPFN_FLAG="--also-dopfn"

DATASETS=(sales law_race)
VARIANTS=(baseline logy)                   # baseline = plain, logy = --y-log-transform

TASK=${SLURM_ARRAY_TASK_ID:-0}
DS_IDX=$(( TASK / 2 ))
VR_IDX=$(( TASK % 2 ))
DATASET="${DATASETS[$DS_IDX]}"
VARIANT="${VARIANTS[$VR_IDX]}"
YFLAG=""
[ "$VARIANT" = "logy" ] && YFLAG="--y-log-transform"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] TASK=$TASK  DATASET=$DATASET  VARIANT=$VARIANT"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

python "$REPO/benchmarks/eval_dopfn_semireal.py" \
    --dataset       "$DATASET" \
    --split-number  "$SPLIT_NUMBER" \
    --n-splits      "$N_SPLITS" \
    --repo          "$REPO" \
    --dopfn         "$DOPFN" \
    --causalpfn     "$CAUSALPFN" \
    --checkpoint    "$CHECKPOINT" \
    --backbone      "$BACKBONE" \
    --skip-malc \
    $YFLAG \
    $ALSO_DOPFN_FLAG \
    --outdir        "$OUTDIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE $DATASET / $VARIANT"
