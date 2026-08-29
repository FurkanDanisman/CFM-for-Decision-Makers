#!/bin/bash
#SBATCH --job-name=cpfn2d-ihdp-em
#SBATCH --output=logs_cpfn2d_ihdp_em/eval_%j.out
#SBATCH --error=logs_cpfn2d_ihdp_em/eval_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Evaluate a cpfn2d checkpoint on IHDP with BOTH raw-mean and MALC EM-mean
# CATE. Compare the two methods on the same forward pass.

set -euo pipefail
mkdir -p logs_cpfn2d_ihdp_em

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
export CKPT="${CKPT:?CKPT env var required — pass via --export=ALL,CKPT=/path/to/step_XXXXX.pt}"
STEP_TAG="${STEP_TAG:-$(basename "$CKPT" .pt)}"
export OUT="${OUT:-$DEPLOY_ROOT/results_cpfn2d_ihdp_em_${STEP_TAG}}"

[ -f "$CKPT" ] || { echo "FATAL: checkpoint not found at $CKPT" >&2; exit 1; }

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] launching cpfn2d IHDP EM+raw eval"
echo "  CKPT=$CKPT"
echo "  OUT=$OUT"
python -u "$REPO/benchmarks/eval_causalpfn2d/eval_cpfn2d_ihdp_em.py"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] done"
