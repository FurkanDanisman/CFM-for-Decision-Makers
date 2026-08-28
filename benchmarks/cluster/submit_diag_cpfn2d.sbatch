#!/bin/bash
#SBATCH --job-name=cpfn2d-diag
#SBATCH --output=logs_cpfn2d_ihdp_raw/diag_%j.out
#SBATCH --error=logs_cpfn2d_ihdp_raw/diag_%j.err
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
mkdir -p logs_cpfn2d_ihdp_raw

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
for _ in 1 2 3; do
    [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
    DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
done
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
export CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
export CKPT="${CKPT:?CKPT env var required}"
export REAL="${REAL:-0}"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%H:%M:%S')] diagnostic on CKPT=$CKPT  REAL=$REAL"
python -u "$REPO/benchmarks/eval_causalpfn2d/diag_cpfn2d_ihdp.py"
echo "[$(date '+%H:%M:%S')] done"
