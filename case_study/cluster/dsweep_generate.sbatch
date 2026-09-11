#!/bin/bash
#SBATCH --job-name=dvar-gen
#SBATCH --output=logs_case_study/dvargen_%A_%a.out
#SBATCH --error=logs_case_study/dvargen_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-3

# Regenerate the d_variation data on the cluster (CPU) — one array task per shift.
# 96k npz total (4 shifts x 8 d x 6 cases x 5 N x 100), deterministic (seed_base=0),
# so this reproduces the local copy exactly without pushing 4.2GB through git.
#
# Env: DATA_ROOT (default $DEPLOY_ROOT/case_study_data/d_variation),
#      D_VALUES (default "2 3 5 10 20 30 40 50"),
#      CONTEXTS (default "50 100 250 500 1000"), N_REAL (default 100).
set -euo pipefail
mkdir -p logs_case_study

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
DATA_ROOT="${DATA_ROOT:-$DEPLOY_ROOT/case_study_data/d_variation}"
D_VALUES="${D_VALUES:-2 3 5 10 20 30 40 50}"
CONTEXTS="${CONTEXTS:-50 100 250 500 1000}"
N_REAL="${N_REAL:-100}"

case "$SLURM_ARRAY_TASK_ID" in
    0) SHIFT=+2 ;; 1) SHIFT=-2 ;; 2) SHIFT=+5 ;; 3) SHIFT=-5 ;;
    *) echo "bad task $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1
echo "[$(date '+%F %T')] regen shift$SHIFT  d=[$D_VALUES]  contexts=[$CONTEXTS] -> $DATA_ROOT/shift$SHIFT"

python -u "$REPO/case_study/d_variation/generation_d.py" \
    --out-dir       "$DATA_ROOT/shift$SHIFT" \
    --n-covariates  $D_VALUES \
    --context-sizes $CONTEXTS \
    --n-realizations "$N_REAL" \
    --seed-base     0 \
    --cate-shift    "$SHIFT"

echo "[$(date '+%F %T')] done shift$SHIFT"
