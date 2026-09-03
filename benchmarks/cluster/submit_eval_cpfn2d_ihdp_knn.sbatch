#!/bin/bash
#SBATCH --job-name=cpfn2d-ihdp-knn
#SBATCH --output=logs_cpfn2d_ihdp_knn/eval_%j.out
#SBATCH --error=logs_cpfn2d_ihdp_knn/eval_%j.err
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Per-query k-NN retrieval eval for cpfn2d on IHDP.
# Env: CKPT, OUT, CAUSALPFN, [K_NN=200, MAX_REAL=]

set -euo pipefail
mkdir -p logs_cpfn2d_ihdp_knn

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
export OUT="${OUT:?OUT required}"
export K_NN="${K_NN:-200}"
export MAX_REAL="${MAX_REAL:-}"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] cpfn2d IHDP k-NN eval"
echo "  CKPT=$CKPT"
echo "  OUT=$OUT   K_NN=$K_NN   MAX_REAL=$MAX_REAL"
python -u "$REPO/benchmarks/eval_causalpfn2d/eval_cpfn2d_ihdp_knn.py"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] done"
