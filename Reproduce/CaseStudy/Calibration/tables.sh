#!/bin/bash
# One interval table per case study, pooled over d and shifts at N=1000.
#
# Aggregation is weighted by n_query: coverage/length/is05 are per-query
# means, so an unweighted mean over cells would give a sparse d=2 cell the
# same say as a dense d=50 one.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/env.sh"
DEPLOY_ROOT="${DEPLOY_ROOT:-$SCRATCH/rpfn_bench_kit}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
CPF="${CPF:-${OUT_ROOT:?set CPF or OUT_ROOT}}"
source "$DEPLOY_ROOT/venv/bin/activate"
for T in cate ate; do
    python -u "$REPO/case_study/density_eval/summarize_case_tables.py" "$CPF" \
        --target "$T" --out "$CPF/${T}_by_case_N1000.md"
done
