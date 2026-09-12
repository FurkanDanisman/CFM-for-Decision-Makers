#!/bin/bash
#SBATCH --job-name=cmech-analysis
#SBATCH --output=logs_cmech_analysis/analysis_%j.out
#SBATCH --error=logs_cmech_analysis/analysis_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#
# Post-processing for the ComplexMech benchmark. CPU only — no GPU is needed,
# and none is requested. Scoring ~100k queries x 8 methods x 6 node counts is
# far too heavy for a login node.
#
# Runs, over $RESULT_ROOT:
#   1. progress audit (both subsets)
#   2. per-method PEHE / ATE tables, all contexts
#   3. pairwise 1D-vs-2D table with medians
#   4. CATE density calibration (coverage / length / WIS / CRPS), both couplings
#   5. bar figures for every (node count, context)
#
# Steps are independent: one failing does not stop the rest.
#
# Required:
#   RESULT_ROOT   the OUT_ROOT the eval array wrote to
# Optional:
#   CONTEXT       context size for the density tables (default 1000)
#   NODES         node counts (default "5 10 20 30 40 50")
#   SUBSET        nonzero | zero | total   (default total)
#   SKIP_DENSITY  1 to skip step 4
#   SKIP_PLOTS    1 to skip step 5
#
# Submit:
#   cd $DEPLOY_ROOT && RESULT_ROOT=$SCRATCH/cmech_dens \
#       sbatch $REPO/benchmarks/cluster/submit_cmech_analysis.sbatch

set -uo pipefail          # NB: no -e — a failing step must not kill the rest
mkdir -p logs_cmech_analysis

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
RESULT_ROOT="${RESULT_ROOT:?RESULT_ROOT required}"
CONTEXT="${CONTEXT:-1000}"
NODES="${NODES:-5 10 20 30 40 50}"
SUBSET="${SUBSET:-total}"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export UWYK_FIG34_DATA="${UWYK_FIG34_DATA:-$REPO/UWYK_Fig3_4/data}"

echo "[$(date '+%F %T')] analysis on $RESULT_ROOT   nodes='$NODES'  N=$CONTEXT  subset=$SUBSET"

step () { echo; echo "───── $* ─────"; }

step "1/5 progress audit"
for sub in nonzero zero; do
    python "$REPO/benchmarks/cmech_progress.py" --root "$RESULT_ROOT" --subset "$sub"
done

step "2/5 per-method PEHE / ATE tables"
python "$REPO/benchmarks/aggregate_cmech_methods.py" \
    --root "$RESULT_ROOT" --all-contexts --nodes $NODES

step "3/5 pairwise 1D-vs-2D (medians + sign test)"
python "$REPO/benchmarks/aggregate_cmech_1d_vs_2d.py" \
    --root "$RESULT_ROOT/N$CONTEXT"

if [ "${SKIP_DENSITY:-0}" != "1" ]; then
    step "4/5 CATE density calibration"
    for cpl in indep comonotonic; do
        echo "--- coupling = $cpl"
        python "$REPO/UWYK_Fig3_4/cate_density_metrics.py" \
            --root "$RESULT_ROOT" --context "$CONTEXT" --nodes $NODES \
            --subset "$SUBSET" --coupling "$cpl"
    done
fi

if [ "${SKIP_PLOTS:-0}" != "1" ]; then
    step "5/5 bar figures"
    python "$REPO/UWYK_Fig3_4/plot_1d_vs_2d.py" \
        --root "$RESULT_ROOT" --all --subset "$SUBSET"
fi

echo; echo "[$(date '+%F %T')] done. outputs under $RESULT_ROOT"
