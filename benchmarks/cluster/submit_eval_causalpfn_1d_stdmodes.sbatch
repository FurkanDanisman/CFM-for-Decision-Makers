#!/bin/bash
#SBATCH --job-name=cpfn1d-stdmode
#SBATCH --output=logs_cpfn1d_ihdp_stdmodes/eval_%j.out
#SBATCH --error=logs_cpfn1d_ihdp_stdmodes/eval_%j.err
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Env: CKPT (1D), STD_MODE, [K_NN], [OUT], [MAX_REAL]
set -euo pipefail
mkdir -p logs_cpfn1d_ihdp_stdmodes

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
export STD_MODE="${STD_MODE:?STD_MODE required}"
export K_NN="${K_NN:-0}"
export MAX_REAL="${MAX_REAL:-}"

STEP_TAG="$(basename "$CKPT" .pt)"
CKPT_TAG="$(dirname "$CKPT" | awk -F/ '{print $(NF-2)}' 2>/dev/null || echo unk)"
KNN_TAG=""
[ "$K_NN" != "0" ] && KNN_TAG="_knn${K_NN}"
export OUT="${OUT:-$DEPLOY_ROOT/results_cpfn1d_ihdp_stdmodes/${CKPT_TAG}_${STEP_TAG}_${STD_MODE}${KNN_TAG}}"

[ -f "$CKPT" ] || { echo "FATAL: checkpoint not found: $CKPT" >&2; exit 1; }

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] cpfn1d IHDP stdmode eval"
echo "  CKPT=$CKPT"
echo "  STD_MODE=$STD_MODE  K_NN=$K_NN"
echo "  OUT=$OUT"
python -u "$REPO/benchmarks/eval_causalpfn2d/eval_causalpfn_1d_stdmodes.py"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] done"
