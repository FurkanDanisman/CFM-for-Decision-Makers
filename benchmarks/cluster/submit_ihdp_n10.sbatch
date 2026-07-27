#!/bin/bash
#SBATCH --job-name=rpfn-ihdp-n10
#SBATCH --output=logs_ihdp_n10/task_%j.out
#SBATCH --error=logs_ihdp_n10/task_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# ─────────────────────────────────────────────────────────────────────────────
# Regenerate the IHDP N=10 demonstration plots on a compute node.
#
# VARIANT env var selects which script(s) to run:
#   ours   plot_ihdp_n10.py            → UWYK-2DMALC + UWYK-2DMALC-NAIVE
#   noanc  plot_ihdp_n10_uwyk_noanc.py → UWYK-NOANC
#   both   both scripts (default)
#
# Usage on killarney:
#   sbatch --account=aip-rgrosse \
#          R-PFN/benchmarks/cluster/submit_ihdp_n10.sbatch
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
mkdir -p logs_ihdp_n10

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
DOPFN="${DOPFN:-$DEPLOY_ROOT/external/dopfn}"
UWYK_SRC="${UWYK_SRC:-$DEPLOY_ROOT/external/uwyk/src}"

VARIANT="${VARIANT:-both}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] VARIANT=$VARIANT"
echo "  DEPLOY_ROOT=$DEPLOY_ROOT"
echo "  REPO=$REPO   CAUSALPFN=$CAUSALPFN   DOPFN=$DOPFN"
echo "  UWYK_SRC=$UWYK_SRC"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

run_ours () {
    echo ">>> plot_ihdp_n10.py"
    REPO="$REPO" CAUSALPFN="$CAUSALPFN" DOPFN="$DOPFN" \
        python "$REPO/benchmarks/plots/plot_ihdp_n10.py"
}

run_noanc () {
    echo ">>> plot_ihdp_n10_uwyk_noanc.py"
    REPO="$REPO" CAUSALPFN="$CAUSALPFN" DOPFN="$DOPFN" UWYK_SRC="$UWYK_SRC" \
        python "$REPO/benchmarks/plots/plot_ihdp_n10_uwyk_noanc.py"
}

case "$VARIANT" in
    ours)  run_ours ;;
    noanc) run_noanc ;;
    both)  run_ours; run_noanc ;;
    *)     echo "[error] unknown VARIANT=$VARIANT (expected ours|noanc|both)" >&2; exit 2 ;;
esac

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE"
