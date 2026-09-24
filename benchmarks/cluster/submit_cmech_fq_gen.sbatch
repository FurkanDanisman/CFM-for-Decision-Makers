#!/bin/bash
#SBATCH --job-name=cmfq-gen
#SBATCH --output=logs_fq/cmfq_gen_%j.out
#SBATCH --error=logs_fq/cmfq_gen_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#
# Generate the ComplexMech fixed-query cells for ONE node count: REALS accepted
# realizations at rho >= RHO_MIN, each with DRAWS replicate observational samples.
#
# One job per node count. Acceptance runs about one in three at rho >= 0.99, so
# REALS=10 takes ~30 attempts; the generator probes each with a single draw and only
# pays for the full DRAWS once rho passes.
#
# Output is one ordinary ComplexMech cell per realization:
#   <OUT>/n<nodes>/r<i>/complexmech/<nodes>node/<regime>/hide_<h>/r0..r{DRAWS-1}.npz
# Separate roots because query UNITS differ between source realizations -- query index
# q names the same unit only within one of them.
#
#   bash R-PFN/benchmarks/cluster/submit_cmech_fq_sweep.sh --submit
#
# Env: NODES (one value), REGIME, HIDE, REALS, DRAWS, NTEST, QUERIES, RHO_MIN,
#      MIN_TAU_HET, CMFQ, MAX_ATTEMPTS, QSEED
set -uo pipefail
mkdir -p logs_fq

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
export UWYK_SRC="${UWYK_SRC:-$DEPLOY_ROOT/external/uwyk_reproduce/src}"
export UWYK_ROOT="${UWYK_ROOT:-$DEPLOY_ROOT/external/uwyk_reproduce}"
[ -d "$UWYK_SRC" ] || { echo "FATAL: UWYK_SRC not found: $UWYK_SRC" >&2; exit 1; }

NODES="${NODES:-5}"
REGIME="${REGIME:-path_TY}"
HIDE="${HIDE:-0.0}"
REALS="${REALS:-10}"
DRAWS="${DRAWS:-100}"
NTEST="${NTEST:-100}"
QUERIES="${QUERIES:-10}"
RHO_MIN="${RHO_MIN:-0.99}"
MIN_TAU_HET="${MIN_TAU_HET:-0.05}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2000}"
QSEED="${QSEED:-12345}"
CMFQ="${CMFQ:-$SC/cmech_fq}"
OUT="$CMFQ/n$NODES"

echo "[$(date '+%F %T')] cmech-fq gen nodes=$NODES regime=$REGIME hide=$HIDE"
echo "  $REALS realizations x $DRAWS samples, $QUERIES of $NTEST queries, rho>=$RHO_MIN"
echo "  out: $OUT"

DONE=$(find "$OUT" -maxdepth 1 -name 'r*' -type d 2>/dev/null | wc -l)
if [ "$DONE" -ge "$REALS" ]; then
    echo "  SKIP: $DONE realization root(s) already present"
    exit 0
fi

# Node count varies the seed base so two node counts do not search the same SCM
# sequence; query seed likewise, or every cell would freeze the same row numbers.
python "$REPO/benchmarks/cmech_fixed_query.py" \
    --nodes "$NODES" --regime "$REGIME" --hide "$HIDE" \
    --target-real "$REALS" --draws "$DRAWS" --n-test "$NTEST" \
    --queries "$QUERIES" --rho-min "$RHO_MIN" \
    --min-tau-het "$MIN_TAU_HET" --max-attempts "$MAX_ATTEMPTS" \
    --seed-base $(( 7000000 + 100000 * NODES )) \
    --query-seed $(( QSEED + NODES )) \
    --out-root "$OUT"
rc=$?
echo "[$(date '+%F %T')] cmech-fq gen nodes=$NODES exit=$rc"
exit $rc
