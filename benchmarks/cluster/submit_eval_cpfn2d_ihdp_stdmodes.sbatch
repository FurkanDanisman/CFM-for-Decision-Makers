#!/bin/bash
#SBATCH --job-name=cpfn2d-stdmode
#SBATCH --output=logs_cpfn2d_ihdp_stdmodes/eval_%j.out
#SBATCH --error=logs_cpfn2d_ihdp_stdmodes/eval_%j.err
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Eval cpfn2d on IHDP under a chosen Y-standardization mode.
# Env vars:
#   CKPT       (required) 2D checkpoint path
#   STD_MODE   pooled | per_arm | winsor | log | recursive
#   OUT        (optional) output dir (auto-generated from ckpt+mode)
#   CAUSALPFN
#   MAX_REAL   default '' (100)

set -euo pipefail
mkdir -p logs_cpfn2d_ihdp_stdmodes

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
export STD_MODE="${STD_MODE:?STD_MODE required (pooled|per_arm|winsor|log|recursive)}"
export MAX_REAL="${MAX_REAL:-}"

STEP_TAG="$(basename "$CKPT" .pt)"
EDGE_TAG="$(dirname "$CKPT" | awk -F/ '{print $(NF-1)"_"$NF}' | tr '/.' '__' | tail -c 40 || true)"
export OUT="${OUT:-$DEPLOY_ROOT/results_cpfn2d_ihdp_stdmodes/${EDGE_TAG}_${STEP_TAG}_${STD_MODE}}"

[ -f "$CKPT" ] || { echo "FATAL: checkpoint not found: $CKPT" >&2; exit 1; }

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] cpfn2d IHDP stdmode eval"
echo "  CKPT=$CKPT"
echo "  STD_MODE=$STD_MODE"
echo "  OUT=$OUT"
python -u "$REPO/benchmarks/eval_causalpfn2d/eval_cpfn2d_ihdp_stdmodes.py"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] done"
