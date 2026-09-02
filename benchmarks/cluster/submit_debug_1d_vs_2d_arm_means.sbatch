#!/bin/bash
#SBATCH --job-name=debug-1d-vs-2d
#SBATCH --output=logs_debug_1d_vs_2d/eval_%j.out
#SBATCH --error=logs_debug_1d_vs_2d/eval_%j.err
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Per-query E[Y0]/E[Y1] side-by-side, 1D CausalPFN vs our 2D cpfn2d.
# Env: CKPT_1D, CKPT_2D, CAUSALPFN, [REALIZATIONS], [N_QUERY]

set -euo pipefail
mkdir -p logs_debug_1d_vs_2d

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
export CKPT_1D="${CKPT_1D:?required}"
export CKPT_2D="${CKPT_2D:?required}"
export REALIZATIONS="${REALIZATIONS:-0,8,12,27,83}"
export N_QUERY="${N_QUERY:-5}"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

python -u "$REPO/benchmarks/eval_causalpfn2d/debug_1d_vs_2d_arm_means.py"
