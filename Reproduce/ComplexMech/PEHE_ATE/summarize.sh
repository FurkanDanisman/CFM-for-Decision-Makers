#!/bin/bash
# PEHE (mean +- std and mean +- SEM) and L1-ATE per (model, d, N).
#   OUT_ROOT=$SCRATCH/cmech_1d2d bash summarize.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/env.sh"
DEPLOY_ROOT="${DEPLOY_ROOT:-$SCRATCH/rpfn_bench_kit}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT required}"
source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONPATH="$REPO/benchmarks${PYTHONPATH:+:$PYTHONPATH}"
python -u "$REPO/benchmarks/aggregate_cmech_methods.py" \
    --root "$OUT_ROOT" --subsets "${SUBSETS:-total}" \
    --contexts "${CONTEXTS:-50 100 250 500 1000}" \
    --data-root "${UWYK_FIG34_DATA:-$REPO/UWYK_Fig3_4/data}"
echo
echo "=== 1D vs 2D pairwise (medians + sign test) ==="
python -u "$REPO/benchmarks/aggregate_cmech_1d_vs_2d.py" \
    --root "$OUT_ROOT" --data-root "${UWYK_FIG34_DATA:-$REPO/UWYK_Fig3_4/data}"
