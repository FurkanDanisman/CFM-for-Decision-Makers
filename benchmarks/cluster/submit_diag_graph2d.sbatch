#!/bin/bash
#SBATCH --job-name=graph2d-diag
#SBATCH --output=logs_graph2d_ihdp/diag_%j.out
#SBATCH --error=logs_graph2d_ihdp/diag_%j.err
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail
mkdir -p logs_graph2d_ihdp

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    if   [ -f "$(dirname "$DEPLOY_ROOT")/venv/bin/activate" ]; then
        DEPLOY_ROOT="$(dirname "$DEPLOY_ROOT")"
    elif [ -f "$(dirname "$(dirname "$DEPLOY_ROOT")")/venv/bin/activate" ]; then
        DEPLOY_ROOT="$(dirname "$(dirname "$DEPLOY_ROOT")")"
    fi
fi

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
export UWYK="${UWYK:-$DEPLOY_ROOT/external/uwyk}"
export CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
export CKPT="${CKPT:-$REPO/checkpoints_graph2d/step_10000.pt}"

source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1
python -u "$REPO/benchmarks/eval_graph2d/diag_graph2d.py"
