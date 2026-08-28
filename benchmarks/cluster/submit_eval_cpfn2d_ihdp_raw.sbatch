#!/bin/bash
#SBATCH --job-name=cpfn2d-ihdp-raw
#SBATCH --output=logs_cpfn2d_ihdp_raw/eval_%j.out
#SBATCH --error=logs_cpfn2d_ihdp_raw/eval_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Evaluate a cpfn2d checkpoint on IHDP using RAW MEAN CATE:
#   CATE = E[Y_do1] - E[Y_do0] computed from marginal means of the 2D
#   joint histogram (interior K*K softmax), un-standardised back to raw Y.
#
# Submit once per checkpoint by overriding CKPT via --export=ALL,CKPT=...
# STEP_TAG (optional): appended to OUT dir name so parallel runs of different
# checkpoints don't clobber each other's per-realization NPZ files.

set -euo pipefail
mkdir -p logs_cpfn2d_ihdp_raw

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
export OUT="${OUT:-$DEPLOY_ROOT/results_cpfn2d_ihdp_raw_${STEP_TAG}}"

[ -f "$CKPT" ] || { echo "FATAL: checkpoint not found at $CKPT" >&2; exit 1; }

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] launching cpfn2d IHDP raw-mean eval"
echo "  CKPT=$CKPT"
echo "  OUT=$OUT"
python -u "$REPO/benchmarks/eval_causalpfn2d/eval_cpfn2d_ihdp_raw.py"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] done"
