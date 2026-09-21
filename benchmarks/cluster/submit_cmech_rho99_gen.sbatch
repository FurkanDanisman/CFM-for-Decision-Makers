#!/bin/bash
#SBATCH --job-name=cm-rho99
#SBATCH --output=logs_cmech_rho99/gen_%j.out
#SBATCH --error=logs_cmech_rho99/gen_%j.err
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
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
# 3h / 4 cpus / 32G matches submit_cmech_gen.sbatch, which generated ALL SIX
# node counts at 100 realizations x 100 queries inside that budget. This job
# does n=5 only, so the per-batch cost is a fraction of that; the open variable
# is how many batches the rho>0.99 acceptance rate demands. A short walltime is
# the right trade because the job resumes: if it times out, resubmit and it
# tops up from the manifest. Do not raise this to buy certainty -- longer
# walltime costs queue priority and buys nothing resume does not already give.
#
# Optional: NODES, TARGET_REAL, N_TEST, RHO_MIN, BATCH, MAX_BATCHES, OUT_ROOT, SEED_BASE
#
# nibi requires an explicit account (there is more than one _cpu
# allocation), and it is left off the #SBATCH lines so the same file
# works on both clusters:
#   sbatch --account=def-zhijing submit_cmech_rho99_gen.sbatch
# Killarney used --account=aip-rgrosse.

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
# Same acceptance band as submit_cmech_gen.sbatch. These are the ComplexMech
# repair: without them the generator emits constant-tau realizations and this
# cell would not be the same benchmark as every other ComplexMech number.
MIN_HET="${MIN_HET:-0.05}"; MAX_HET="${MAX_HET:-2.0}"; MAX_PIN="${MAX_PIN:-0.5}"
OUT_ROOT="${OUT_ROOT:-$SCRATCH/cmech_data_rho99}"

# generate_pehe_benchmark builds the SCMs through the UWYK checkout, so these
# must point at the deployed copy -- without them it looks in /tmp and dies.
export UWYK_SRC="${UWYK_SRC:-$DEPLOY_ROOT/external/uwyk_reproduce/src}"
export UWYK_ROOT="${UWYK_ROOT:-$DEPLOY_ROOT/external/uwyk_reproduce}"

source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1
[ -d "$UWYK_ROOT/experiments" ] || {
    echo "FATAL: no UWYK configs at $UWYK_ROOT/experiments" >&2; exit 1; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$REPO/benchmarks:$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%F %T')] n=$NODES  target=$TARGET_REAL  n_test=$N_TEST  rho>$RHO_MIN"
echo "  out=$OUT_ROOT"
echo "  band: sd(tau)/sd(y0) in [$MIN_HET, $MAX_HET], pinned <= $MAX_PIN"
echo "  uwyk: $UWYK_ROOT"

python -u "$REPO/benchmarks/make_cmech_rho_filtered.py" \
    --nodes "$NODES" \
    --target-real "$TARGET_REAL" \
    --n-test "$N_TEST" \
    --rho-min "$RHO_MIN" \
    --batch "$BATCH" \
    --max-batches "$MAX_BATCHES" \
    --seed-base "$SEED_BASE" \
    --out-root "$OUT_ROOT" \
    --extra --min-tau-het "$MIN_HET" --max-tau-het "$MAX_HET" --max-pinned "$MAX_PIN"
rc=$?
echo "[$(date '+%F %T')] done rc=$rc"
if [ $rc -ne 0 ]; then
    echo "short of target -- resubmit to top up (it resumes), or lower RHO_MIN" >&2
fi
exit $rc
