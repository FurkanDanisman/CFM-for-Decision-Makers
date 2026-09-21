#!/bin/bash
#SBATCH --job-name=cm-rho99
#SBATCH --output=logs_cmech_rho99/gen_%j.out
#SBATCH --error=logs_cmech_rho99/gen_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#
# Generate a ComplexMech cell of ONLY strongly-coupled realizations: keep those
# with arm-noise rho > RHO_MIN until TARGET_REAL are collected, N_TEST queries each.
#
# Purpose. ComplexMech scores coverage against true_cate = Y_do1 - Y_do0, a NOISY
# tau; every other benchmark uses the noiseless mu_1 - mu_0. The shared noise
# cancels between the paired passes exactly when the arms are coupled, so at
# rho -> 1 the target collapses onto the noiseless CATE the heads predict. Any
# undercoverage that SURVIVES on this cell cannot be blamed on the target.
#
# Resumable: re-running tops up toward TARGET_REAL and never re-consumes a source
# realization it already accepted (tracked in the manifest).
#
# Optional: NODES, TARGET_REAL, N_TEST, RHO_MIN, BATCH, MAX_BATCHES, OUT_ROOT, SEED_BASE

set -uo pipefail
mkdir -p logs_cmech_rho99

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
SCRATCH="${SCRATCH:-/scratch/$USER}"

NODES="${NODES:-5}"
TARGET_REAL="${TARGET_REAL:-100}"
N_TEST="${N_TEST:-100}"
RHO_MIN="${RHO_MIN:-0.99}"
BATCH="${BATCH:-100}"
MAX_BATCHES="${MAX_BATCHES:-60}"
SEED_BASE="${SEED_BASE:-100000}"
OUT_ROOT="${OUT_ROOT:-$SCRATCH/cmech_data_rho99}"

source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$REPO/benchmarks:$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%F %T')] n=$NODES  target=$TARGET_REAL  n_test=$N_TEST  rho>$RHO_MIN"
echo "  out=$OUT_ROOT"

python -u "$REPO/benchmarks/make_cmech_rho_filtered.py" \
    --nodes "$NODES" \
    --target-real "$TARGET_REAL" \
    --n-test "$N_TEST" \
    --rho-min "$RHO_MIN" \
    --batch "$BATCH" \
    --max-batches "$MAX_BATCHES" \
    --seed-base "$SEED_BASE" \
    --out-root "$OUT_ROOT"
rc=$?
echo "[$(date '+%F %T')] done rc=$rc"
if [ $rc -ne 0 ]; then
    echo "short of target -- resubmit to top up (it resumes), or lower RHO_MIN" >&2
fi
exit $rc
