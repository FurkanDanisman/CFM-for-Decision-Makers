#!/bin/bash
# Per-realization PEHE/L1 box plots (box = IQR, line = median, no whiskers or
# fliers) and compact median heatmaps coloured by rank.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/env.sh"
DEPLOY_ROOT="${DEPLOY_ROOT:-$SCRATCH/rpfn_bench_kit}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
SWEEP="${SWEEP:-${OUT_ROOT:?set SWEEP or OUT_ROOT}}"
source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONPATH="$REPO/benchmarks${PYTHONPATH:+:$PYTHONPATH}"
python -u "$REPO/realcause_eval/plot_scm_boxplots.py" --sweep "$SWEEP" \
    --out-dir "${FIG_DIR:-$SWEEP/figures}"
python -u "$REPO/realcause_eval/plot_scm_heatmap.py"  --sweep "$SWEEP" \
    --out-dir "${FIG_DIR:-$SWEEP/figures}"
