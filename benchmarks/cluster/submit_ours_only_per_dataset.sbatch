#!/bin/bash
#SBATCH --job-name=ours-per-ds
#SBATCH --output=logs/ours_per_ds_%A_%a.out
#SBATCH --error=logs/ours_per_ds_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --array=0-4

# 5-task array — one task per RealCause dataset. Each task loops over
# ALL realizations of its dataset sequentially. No 500-task arrays.
#
# Task 0: IHDP    (100 realizations)
# Task 1: ACIC    (10 realizations)
# Task 2: CPS     (100 realizations)
# Task 3: PSID    (100 realizations)
# Task 4: PSIDbal (100 realizations)
#
# Env vars:
#   DEPLOY_ROOT  cluster deploy root (default $PWD)
#   BACKBONE     ipfn (fn=50) | dopfn_bb (Do-PFN-bb)          — required
#   CHECKPOINT   path to matching .pt                          — required
#   OUTDIR       per-realization npzs land here                — required
#   MALC_B       MALC bootstrap draws (default 500)
#   MALC_MAX_K   MALC max K (default 3)

set -euo pipefail
mkdir -p logs

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
DOPFN="${DOPFN:-$DEPLOY_ROOT/external/dopfn}"
UWYK="${UWYK:-$DEPLOY_ROOT/external/uwyk}"
CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT required}"
UWYK_CKPT="${UWYK_CKPT:-$UWYK/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0}"
OUTDIR="${OUTDIR:?OUTDIR required}"
MALC_B="${MALC_B:-500}"
MALC_MAX_K="${MALC_MAX_K:-3}"
BACKBONE="${BACKBONE:?BACKBONE required — ipfn | dopfn_bb}"

DATASETS=(IHDP ACIC CPS PSID PSIDbal)
N_REAL_PER=(100 10 100 100 100)

TASK=${SLURM_ARRAY_TASK_ID:-0}
DATASET="${DATASETS[$TASK]}"
N_REAL="${N_REAL_PER[$TASK]}"

WORKERS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p "$OUTDIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] TASK=$TASK  DATASET=$DATASET  "\
"N_REAL=$N_REAL  BACKBONE=$BACKBONE  OUTDIR=$OUTDIR"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="$WORKERS"
export MKL_NUM_THREADS="$WORKERS"
export OPENBLAS_NUM_THREADS="$WORKERS"
export PYTHONUNBUFFERED=1

cd "$DOPFN"

for (( REAL_IDX=0; REAL_IDX<N_REAL; REAL_IDX++ )); do
    echo "[$(date '+%H:%M:%S')] r=$REAL_IDX/$N_REAL"
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
        --workers       "$WORKERS" \
        --only-ours
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE dataset=$DATASET"
