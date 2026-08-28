#!/bin/bash
#SBATCH --job-name=graph2d-ihdp-te
#SBATCH --output=logs_graph2d_ihdp_te/eval_%j.out
#SBATCH --error=logs_graph2d_ihdp_te/eval_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Graph2D IHDP eval with target-encoded T (matches UWYK's Table 1 pipeline
# in RealCauseEval/run_baselines/dofm_full_conditioning.py).
# Same architecture, same adjacency layout, same preprocessing, ONE change:
# T_obs is target-encoded via T -> mean(Y|T) before passing to the model.

set -euo pipefail
mkdir -p logs_graph2d_ihdp_te

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
export OUT="${OUT:-$DEPLOY_ROOT/results_graph2d_ihdp_te}"

[ -f "$CKPT" ] || { echo "FATAL: checkpoint not found at $CKPT" >&2; exit 1; }

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] launching graph2d IHDP eval with target-encoded T"
python -u "$REPO/benchmarks/eval_graph2d/eval_graph2d_ihdp_te.py"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] done"
