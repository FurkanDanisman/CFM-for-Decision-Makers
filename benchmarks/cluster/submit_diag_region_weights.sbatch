#!/bin/bash
#SBATCH --job-name=diag-region-w
#SBATCH --output=logs_diag_region_weights/eval_%j.out
#SBATCH --error=logs_diag_region_weights/eval_%j.err
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Report the mean 9-region weight distribution for a cpfn2d checkpoint.
# Env: CKPT (required), CAUSALPFN, [N_TASKS]

set -euo pipefail
mkdir -p logs_diag_region_weights

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
export CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
export CAUSALPFN_ROOT="$CAUSALPFN"
export CKPT="${CKPT:?CKPT required}"
export N_TASKS="${N_TASKS:-20}"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

python -u "$REPO/benchmarks/eval_causalpfn2d/diag_region_weights.py"
