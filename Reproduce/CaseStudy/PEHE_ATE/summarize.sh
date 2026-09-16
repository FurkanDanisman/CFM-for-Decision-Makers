#!/bin/bash
# PEHE + L1-ATE per (model, d, case), pooled over shifts.
#   SWEEP=$DEPLOY_ROOT/results_case_study/dvar bash summarize.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/env.sh"
DEPLOY_ROOT="${DEPLOY_ROOT:-$SCRATCH/rpfn_bench_kit}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
SWEEP="${SWEEP:-${OUT_ROOT:?set SWEEP or OUT_ROOT}}"
source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONPATH="$REPO/benchmarks${PYTHONPATH:+:$PYTHONPATH}"
python -u "$REPO/realcause_eval/aggregate_scm_ctx_sweep.py" \
    --sweep "$SWEEP" --out "$SWEEP/summary"
