#!/bin/bash
#SBATCH --job-name=graph2d-ihdp-matched
#SBATCH --output=logs_graph2d_ihdp_matched/eval_%j.out
#SBATCH --error=logs_graph2d_ihdp_matched/eval_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Evaluate a graph-conditioned 2D-head checkpoint on IHDP under the
# TRAINING-matched adjacency convention: {-1, +1} true signed matrix,
# then apply random hide_mask at hide_frac in {0.0, 0.25, 0.5, 0.75, 0.9}
# and average over multiple random masks per realization.

set -euo pipefail
mkdir -p logs_graph2d_ihdp_matched

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
export UWYK="${UWYK:-$DEPLOY_ROOT/external/uwyk}"
export CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
export CKPT="${CKPT:-$REPO/checkpoints_graph2d/step_42000.pt}"
export OUT="${OUT:-$DEPLOY_ROOT/results_graph2d_ihdp_matched}"
export N_MASKS="${N_MASKS:-8}"

[ -f "$CKPT" ] || { echo "FATAL: checkpoint not found at $CKPT" >&2; exit 1; }

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] launching training-matched graph2d IHDP eval"
python -u "$REPO/benchmarks/eval_graph2d/eval_graph2d_ihdp_matched.py"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] done"
