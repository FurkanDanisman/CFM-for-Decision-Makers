#!/bin/bash
#SBATCH --job-name=diag-deep
#SBATCH --output=logs_diag_deep/diag_%j.out
#SBATCH --error=logs_diag_deep/diag_%j.err
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
mkdir -p logs_diag_deep

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
export UWYK="${UWYK:-$DEPLOY_ROOT/external/uwyk}"
export CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
export CKPT="${CKPT:-$REPO/checkpoints_graph2d/step_50000.pt}"
export DATASET="${DATASET:-ACIC}"
export REAL="${REAL:-0}"
export OUT="${OUT:-$DEPLOY_ROOT/results_diag_deep_layers}"

[ -f "$CKPT" ] || { echo "FATAL: ckpt not at $CKPT" >&2; exit 1; }

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] diag deep layers  ckpt=$CKPT  dataset=$DATASET r=$REAL"
python -u "$REPO/benchmarks/eval_graph2d/diag_deep_layers_acic.py"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] done"
