#!/bin/bash
#SBATCH --job-name=cpfn2d-malc
#SBATCH --output=logs_cpfn2d_ihdp_malc/eval_%j.out
#SBATCH --error=logs_cpfn2d_ihdp_malc/eval_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# MALC-based IHDP eval — reports BOTH 1D MALC on marginals AND 2D MALC + marginals
# for CATE means. Multiprocessing across queries for the 2D fits.
#
# Env overrides:
#   CKPT        (required) checkpoint path
#   STEP_TAG    (optional) appended to OUT dir name
#   MALC_B      2D MALC B parameter (default 1000)
#   MALC_MAX_K  2D MALC max_K (default 5)
#   REAL_START, REAL_END  # IHDP realization range [start, end)  default [0, 100)

set -euo pipefail
mkdir -p logs_cpfn2d_ihdp_malc

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
export CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
export CKPT="${CKPT:?CKPT env var required}"
STEP_TAG="${STEP_TAG:-$(basename "$CKPT" .pt)}"
export OUT="${OUT:-$DEPLOY_ROOT/results_cpfn2d_ihdp_malc_${STEP_TAG}}"

[ -f "$CKPT" ] || { echo "FATAL: ckpt not found at $CKPT" >&2; exit 1; }

# MALC knobs (overridable via --export)
export MALC_B="${MALC_B:-1000}"
export MALC_MAX_K="${MALC_MAX_K:-5}"
export N_WORKERS="${SLURM_CPUS_PER_TASK:-8}"
export REAL_START="${REAL_START:-0}"
export REAL_END="${REAL_END:-100}"

source "$DEPLOY_ROOT/venv/bin/activate"
# BLAS threads capped at 1 — each MALC worker will spawn its own math. Otherwise
# workers × BLAS threads = thread thrash.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] cpfn2d MALC IHDP eval"
echo "  CKPT=$CKPT"
echo "  OUT=$OUT"
echo "  MALC_B=$MALC_B  MALC_MAX_K=$MALC_MAX_K  N_WORKERS=$N_WORKERS  realizations=[$REAL_START,$REAL_END)"
python -u "$REPO/benchmarks/eval_causalpfn2d/eval_cpfn2d_ihdp_malc.py"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] done"
