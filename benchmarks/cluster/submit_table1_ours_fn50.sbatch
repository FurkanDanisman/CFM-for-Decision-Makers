#!/bin/bash
#SBATCH --job-name=ours-t1-fn50
#SBATCH --output=logs_table1_ours_fn50/task_%A_%a.out
#SBATCH --error=logs_table1_ours_fn50/task_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --array=0-3

# Evaluate our fn=50 checkpoint on IHDP / ACIC / CPS / PSID(unbal) under
# UWYK's reproduce-realcause-results protocol (single forward, no clustering).
# One array task per dataset.
#
# Produces one pickle per realization under:
#   $DEPLOY_ROOT/results_table1_ours_fn50/table1_ours_fn50_<jobid>/ours_fn50_<DATASET>_<r>
# Aggregate however you like; per-realization {pehe, ate_rel_err} + preds
# are already inside.

set -euo pipefail
mkdir -p logs_table1_ours_fn50

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    if   [ -f "$(dirname "$DEPLOY_ROOT")/venv/bin/activate" ]; then
        DEPLOY_ROOT="$(dirname "$DEPLOY_ROOT")"
    elif [ -f "$(dirname "$(dirname "$DEPLOY_ROOT")")/venv/bin/activate" ]; then
        DEPLOY_ROOT="$(dirname "$(dirname "$DEPLOY_ROOT")")"
    fi
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
export CAUSALPFN_ROOT="${CAUSALPFN_ROOT:-$DEPLOY_ROOT/external/causalpfn}"
export OURS_CKPT="${OURS_CKPT:-$REPO/checkpoints/step_50000_final.pt}"
export TABLE1_OUT_ROOT="${TABLE1_OUT_ROOT:-$DEPLOY_ROOT/results_table1_ours_fn50}"

[ -f "$OURS_CKPT" ] || { echo "FATAL: fn=50 checkpoint not found at $OURS_CKPT" >&2; exit 1; }

DATASETS=(IHDP ACIC CPS PSID)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] launching  dataset=$DATASET"
python -u "$REPO/benchmarks/uwyk_table1/ours_fn50_no_clustering.py" \
    --dataset "$DATASET" --model ours_fn50 --exp_name "table1_ours_fn50_${SLURM_ARRAY_JOB_ID}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] done"
