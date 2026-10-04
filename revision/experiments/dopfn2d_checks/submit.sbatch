#!/bin/bash
#SBATCH --job-name=dopfn2d-check
#SBATCH --output=logs_hist/dopfn2d_check.out
#SBATCH --error=logs_hist/dopfn2d_check.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# Two checks on Do-PFN 2D's ComplexMech under-coverage, hist intervals, existing dumps only:
#   1. coverage by arm-noise correlation rho (benchmarks/cmech_coverage_by_rho.py), all models
#   2. share of query outcomes outside Do-PFN 2D's grid, and coverage on vs off the grid
#   cd $KIT && sbatch R-PFN/revision/experiments/dopfn2d_checks/submit.sbatch
set -uo pipefail
mkdir -p logs_hist
DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"; REPO="$DEPLOY_ROOT/R-PFN"
OUT="${OUT:-$SCRATCH/hist/dopfn2d_checks}"; mkdir -p "$OUT"
source "$DEPLOY_ROOT/venv/bin/activate"
export CDM_INTERVAL=hist PYTHONUNBUFFERED=1 ATOM_GRID_MAX=4096
export PYTHONPATH="$REPO/benchmarks:$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
D="$SCRATCH/cmech_dumps"
python -u "$REPO/benchmarks/cmech_coverage_by_rho.py" \
    --root $D/dopfn_native $D/dopfn_repro_joint2d $D/uwyk1d $D/graph2d $D/cpfn1d_j1024 $D/cpfn2d_eta0 \
    --data-root "$SCRATCH/cmech_data_v2" --thresholds 0.8 0.9 0.99 --out "$OUT/rho_buckets_hist.md"
python -u "$REPO/revision/experiments/dopfn2d_checks/offgrid.py" \
    "$D/dopfn_repro_joint2d" "$SCRATCH/cmech_data_v2" | tee "$OUT/offgrid.md"
