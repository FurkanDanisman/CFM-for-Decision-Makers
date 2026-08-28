#!/bin/bash
#SBATCH --job-name=diag-adj-sens
#SBATCH --output=logs_diag_adj_sens/diag_%j.out
#SBATCH --error=logs_diag_adj_sens/diag_%j.err
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Test whether the graph2d model actually responds to adjacency at
# inference. Load a trained checkpoint, run 5 IHDP realizations through
# 4 different adjacency matrices (noanc / anc_full / anc_flipped /
# anc_all1) and report per-realization how much the CATE predictions
# differ across them.

set -euo pipefail
mkdir -p logs_diag_adj_sens

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

[ -f "$CKPT" ] || { echo "FATAL: checkpoint not found at $CKPT" >&2; exit 1; }

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] running adj-sensitivity diagnostic"
python -u "$REPO/benchmarks/eval_graph2d/diag_adj_sensitivity.py"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] done"
