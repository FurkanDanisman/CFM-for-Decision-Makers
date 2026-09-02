#!/bin/bash
#SBATCH --job-name=diag-ystd
#SBATCH --output=logs_diag_ystd/eval_%j.out
#SBATCH --error=logs_diag_ystd/eval_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Histogram of y_std across the training data generator.
# Env: N_TASKS (default 200), N_SAMPLES (default 1024)

set -euo pipefail
mkdir -p logs_diag_ystd

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
export N_TASKS="${N_TASKS:-200}"
export N_SAMPLES="${N_SAMPLES:-1024}"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

python -u "$REPO/benchmarks/eval_causalpfn2d/diag_ystd_hist.py"
