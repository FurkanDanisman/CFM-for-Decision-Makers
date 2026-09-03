#!/bin/bash
#SBATCH --job-name=diag-binwidth
#SBATCH --output=logs_diag_binwidth/eval_%j.out
#SBATCH --error=logs_diag_binwidth/eval_%j.err
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# What fraction of |Ê_t - μ_t| is under one bin-width δ?
# Env:
#   CKPT      (required)
#   HEAD      2d | 1d   (default 2d)
#   CAUSALPFN, MAX_REAL

set -euo pipefail
mkdir -p logs_diag_binwidth

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
export CKPT="${CKPT:?CKPT required}"
export MAX_REAL="${MAX_REAL:-}"
HEAD="${HEAD:-2d}"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

if [ "$HEAD" = "1d" ]; then
    SCRIPT="$REPO/benchmarks/eval_causalpfn2d/diag_within_binwidth_1d.py"
else
    SCRIPT="$REPO/benchmarks/eval_causalpfn2d/diag_within_binwidth.py"
fi
python -u "$SCRIPT"
