#!/bin/bash
#SBATCH --job-name=rc-calib
#SBATCH --output=logs_reproduce/rc_calib_%j.out
#SBATCH --error=logs_reproduce/rc_calib_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#
# CPU ONLY. coverage95 / length / IS_0.05 for CATE and ATE, all six models
# through the single scorer, each on its own native bin resolution.
#
# Memory: the ATE path barycenters every query of a realization, and
# dopfn_native's atom grid reaches ATOM_GRID_MAX bins -> a tau support of
# 2J-1. This OOMs a login node.
#
#   OUT_ROOT=$SCRATCH/rc_dens_uni sbatch score.sbatch
set -uo pipefail
mkdir -p logs_reproduce
DEPLOY_ROOT="${DEPLOY_ROOT:-$SCRATCH/rpfn_bench_kit}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT required}"
source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks:$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
# Fixed across ALL benchmarks. Do not vary it per run: dopfn_native's numbers
# would then come off different tau resolutions and stop being comparable.
export ATOM_GRID_MAX="${ATOM_GRID_MAX:-4096}"
for T in ${TARGETS:-cate ate}; do
    echo "=== target=$T ==="
    python -u "$REPO/UWYK_Fig3_4/cate_density_metrics.py" \
        --root "$OUT_ROOT" --target "$T" \
        --dataset IHDP ACIC CPS PSID PSID_bal \
      || echo "WARN: target=$T exited nonzero"
done
