#!/bin/bash
#SBATCH --job-name=cs-datagen
#SBATCH --output=logs_reproduce/cs_datagen_%A_%a.out
#SBATCH --error=logs_reproduce/cs_datagen_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-2
#
# One task per shift. Writes <shift>/d<K>/<Case>/N<ctx>/ plus a manifest.json
# recording n_covariates, cases, context sizes, n_realizations, seed_base and
# cate_shift -- that manifest is what makes the tree reproducible.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/env.sh"
mkdir -p logs_reproduce
DEPLOY_ROOT="${DEPLOY_ROOT:-$SCRATCH/rpfn_bench_kit}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
DATA="${DATA:-$DEPLOY_ROOT/case_study_data/d_variation}"
SHIFTS=(0 +2 -2)
SHIFT="${SHIFTS[$SLURM_ARRAY_TASK_ID]}"
source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1
export DOPFN_ROOT="${DOPFN_ROOT:-$DEPLOY_ROOT/external/dopfn}"
python -u "$REPO/case_study/d_variation/generation_d.py" \
    --out-root "$DATA/shift$SHIFT" \
    --ds "${DS:-2 3 5 10 20 30 40 50}" \
    --contexts "${CONTEXTS:-50 100 250 500 1000}" \
    --n-realizations "${NREAL:-100}" \
    --cate-shift "$SHIFT"
