#!/bin/bash
#SBATCH --job-name=g2d-ihdp-malc
#SBATCH --output=logs_graph2d_ihdp_malc/eval_%j.out
#SBATCH --error=logs_graph2d_ihdp_malc/eval_%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# 2D MALC mean CATE on IHDP for graph2d v6a vs noanc.
#
# Env overrides:
#   CKPT              (required) checkpoint path
#   OUT               (required) output dir
#   MALC_B            default 1000
#   MALC_MAX_K        default 3
#   MALC_N_EVAL       default 101
#   EVAL_MAX_CONTEXT  default '' (1000 recommended for parity with realcause eval)
#   EVAL_CONTEXT_SEED default '42'
#   MAX_REAL          default '' (100)

set -euo pipefail
mkdir -p logs_graph2d_ihdp_malc

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
export CKPT="${CKPT:?CKPT required}"
export OUT="${OUT:?OUT required}"

[ -f "$CKPT" ] || { echo "FATAL: ckpt not found at $CKPT" >&2; exit 1; }

export MALC_B="${MALC_B:-1000}"
export MALC_MAX_K="${MALC_MAX_K:-3}"
export MALC_N_EVAL="${MALC_N_EVAL:-101}"
export EVAL_MAX_CONTEXT="${EVAL_MAX_CONTEXT:-}"
export EVAL_CONTEXT_SEED="${EVAL_CONTEXT_SEED:-42}"
export MAX_REAL="${MAX_REAL:-}"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] graph2d IHDP MALC eval"
echo "  CKPT=$CKPT"
echo "  OUT=$OUT"
echo "  MALC_B=$MALC_B  MALC_MAX_K=$MALC_MAX_K  N_EVAL=$MALC_N_EVAL"
echo "  EVAL_MAX_CONTEXT=$EVAL_MAX_CONTEXT  seed=$EVAL_CONTEXT_SEED  MAX_REAL=$MAX_REAL"
python -u "$REPO/benchmarks/eval_graph2d/eval_graph2d_malc_mean_ihdp.py"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] done"
