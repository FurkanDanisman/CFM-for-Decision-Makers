#!/bin/bash
#SBATCH --job-name=graph2d-ihdp
#SBATCH --output=logs_graph2d_ihdp/eval_%j.out
#SBATCH --error=logs_graph2d_ihdp/eval_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Evaluate a graph-conditioned 2D-head checkpoint on IHDP under two
# adjacency modes (anc / noanc), analogous to UWYK's two rows.
# Sequential loop over all 100 realizations; ~1 GPU-second per realization.

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
echo "[bootstrap] DEPLOY_ROOT=$DEPLOY_ROOT  PWD=$PWD"
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
export UWYK="${UWYK:-$DEPLOY_ROOT/external/uwyk}"
export CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
export CKPT="${CKPT:-$REPO/checkpoints_graph2d/step_10000.pt}"
export OUT="${OUT:-$DEPLOY_ROOT/results_graph2d_ihdp}"

[ -f "$CKPT" ] || { echo "FATAL: checkpoint not found at $CKPT" >&2; exit 1; }

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
# Shims dir must be BEFORE python starts so sitecustomize.py auto-loads
# and shims faiss/huggingface_hub — otherwise causalpfn/__init__.py fails
# at "ModuleNotFoundError: No module named 'faiss'" (job 5035871).
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] launching graph2d IHDP eval"
python -u "$REPO/benchmarks/eval_graph2d/eval_graph2d_ihdp.py"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] done"
