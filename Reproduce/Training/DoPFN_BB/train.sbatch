#!/bin/bash
#SBATCH --job-name=train-dopfn-bb
#SBATCH --output=logs_reproduce/train_dopfnbb_%j.out
#SBATCH --error=logs_reproduce/train_dopfnbb_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail
mkdir -p logs_reproduce
DEPLOY_ROOT="${DEPLOY_ROOT:-$SCRATCH/rpfn_bench_kit}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
export DOPFN_ROOT="${DOPFN_ROOT:-$DEPLOY_ROOT/external/dopfn}"
source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
python -u "$REPO/training_dopfn_base/train.py" \
    --out-dir "${OUT_DIR:-$DEPLOY_ROOT/dopfn_bb_output}" \
    --J "${J:-10}" "$@"
