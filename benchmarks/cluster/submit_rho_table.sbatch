#!/bin/bash
#SBATCH --job-name=rho-table
#SBATCH --output=logs_rho/rho_%j.out
#SBATCH --error=logs_rho/rho_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#
# Cross-world correlation table: one row per benchmark, mean +- sd over realizations
# plus the inverse-variance combined CI and the p-value against rho = 0.
#
# A job rather than a login-node run: it opens every realization npz of every benchmark
# and fits an OLS per file, which is thousands of small reads on shared storage -- slow
# enough to look hung, and rude to run on a login node.
#
#   bash R-PFN/benchmarks/cluster/fq_submit.sh rho     # or sbatch directly
# NOT plain sbatch unless logs_rho exists: Slurm opens --output before line 1.
#
# Env: MAX_FILES (realizations per benchmark, 0 = all), RHO99_ROOT, OUT
set -uo pipefail
mkdir -p logs_rho

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
# shellcheck disable=SC1091
source "$DEPLOY_ROOT/venv/bin/activate"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
SC="${SCRATCH:?SCRATCH must be set}"
export CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"

MAX_FILES="${MAX_FILES:-0}"
RHO99_ROOT="${RHO99_ROOT:-$SC/cmech_data_rho99}"
OUT="${OUT:-$SC/rho_table.md}"

echo "[$(date '+%F %T')] rho table: max_files=${MAX_FILES:-all} -> $OUT"
echo "  rho99 root: $RHO99_ROOT"
cd "$DEPLOY_ROOT"
python "$REPO/benchmarks/arm_noise_table.py" \
    --no-download \
    --max-files "$MAX_FILES" \
    --rho99-root "$RHO99_ROOT" \
    --out "$OUT"
rc=$?
echo "[$(date '+%F %T')] done (exit $rc)"
exit $rc
