#!/bin/bash
#SBATCH --job-name=cs-generate
#SBATCH --output=logs_case_study/gen_%j.out
#SBATCH --error=logs_case_study/gen_%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Regenerate the case-study npz on the cluster (CPU only — deterministic, fast).
# The data itself is not in git; this reproduces it from the committed
# case_study/generation.py so nothing large has to be pushed.
#
# Env vars:
#   DATA_ROOT   where the npz land (default: $DEPLOY_ROOT/case_study_data/shift<TAG>)
#   CATE_SHIFT  constant additive effect beta (direct-T->Y cases only). Default 2.
#   CONTEXTS    context sizes to generate. Default "200 500 1000".
#   N_REAL      realizations per (case, N) cell. Default 100.
#   SEED_BASE   base seed. Default 0.
set -euo pipefail
mkdir -p logs_case_study

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found under $DEPLOY_ROOT" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
CATE_SHIFT="${CATE_SHIFT:-2}"
CONTEXTS="${CONTEXTS:-200 500 1000}"
N_REAL="${N_REAL:-100}"
SEED_BASE="${SEED_BASE:-0}"
_tag="$(printf '%+g' "$CATE_SHIFT")"          # e.g. +2, -5, +0
DATA_ROOT="${DATA_ROOT:-$DEPLOY_ROOT/case_study_data/shift${_tag}}"

source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1

echo "[$(date '+%F %T')] generate case studies  beta=$CATE_SHIFT  contexts=[$CONTEXTS]"
echo "  DATA_ROOT=$DATA_ROOT  n_real=$N_REAL  seed_base=$SEED_BASE"

python -u "$REPO/case_study/generation.py" \
    --out-dir        "$DATA_ROOT" \
    --context-sizes  $CONTEXTS \
    --n-realizations "$N_REAL" \
    --seed-base      "$SEED_BASE" \
    --cate-shift     "$CATE_SHIFT"

echo "[$(date '+%F %T')] done -> $DATA_ROOT"
