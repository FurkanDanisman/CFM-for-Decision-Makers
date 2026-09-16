#!/bin/bash
# Grouped 1D-vs-2D bar plots, one image per metric per (N, d) cell.
# Palette #D97706 (1D) / #B91C1C (2D) is validated by the dataviz six-checks:
# CVD deltaE 16.2 deutan, 18.8 normal, both well past the 8 threshold.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/env.sh"
DEPLOY_ROOT="${DEPLOY_ROOT:-$SCRATCH/rpfn_bench_kit}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT required}"
source "$DEPLOY_ROOT/venv/bin/activate"
python -u "$REPO/UWYK_Fig3_4/plot_1d_vs_2d.py" \
    --root "$OUT_ROOT" --all \
    --out-dir "${FIG_DIR:-$REPO/UWYK_Fig3_4/figures}"
