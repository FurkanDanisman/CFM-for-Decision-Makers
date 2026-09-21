#!/bin/bash
#SBATCH --job-name=cm-rho
#SBATCH --output=logs_cmech_rho/rho_%j.out
#SBATCH --error=logs_cmech_rho/rho_%j.err
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#
# ComplexMech coverage bucketed by the arm-noise correlation, for EVERY model and
# every node count, on a compute node rather than the login node.
#
# The question: ComplexMech scores coverage against true_cate = Y_do1 - Y_do0, a
# NOISY tau, while every other benchmark uses the noiseless mu_1 - mu_0. Noise
# cancels between the paired passes exactly when the arms are coupled, so if the
# undercoverage is a target mismatch, coverage must rise with rho. The decisive
# check is coverage rising while LENGTH FALLS -- a model cannot cover more by
# predicting narrower, so that can only be the target shrinking.
#
# Raw scoring only (no MALC), so this is minutes per model, not hours.
#
# Optional: ROOTS, CM_NODES, CTX, THRESHOLDS, OUT, METHODS, MAX_REAL

set -uo pipefail
mkdir -p logs_cmech_rho

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
SCRATCH="${SCRATCH:-/scratch/$USER}"
CTX="${CTX:-1000}"
NODES="${CM_NODES:-5 10 20 30 40 50}"
THRESHOLDS="${THRESHOLDS:-0.8 0.9}"
OUT="${OUT:-$SCRATCH/cmech_rho_buckets.md}"

source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ATOM_GRID_MAX="${ATOM_GRID_MAX:-4096}"
export PYTHONPATH="$REPO/benchmarks:$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
export UWYK_FIG34_DATA="${UWYK_FIG34_DATA:-$SCRATCH/cmech_data_v2}"

# Default to every model dump root. Quoted expansion would pass the literal glob,
# so collect the matches first and fail loudly if there are none.
if [ -n "${ROOTS:-}" ]; then
    read -r -a ROOT_ARR <<<"$ROOTS"
else
    ROOT_ARR=("$SCRATCH"/cmech_dumps/*)
fi
if [ ! -d "${ROOT_ARR[0]:-}" ]; then
    echo "no model dump roots under $SCRATCH/cmech_dumps -- set ROOTS explicitly" >&2
    exit 1
fi

echo "[$(date '+%F %T')] ${#ROOT_ARR[@]} root(s), nodes=$NODES, ctx=$CTX"
echo "  data=$UWYK_FIG34_DATA"
echo "  out=$OUT"

EXTRA=()
[ -n "${METHODS:-}" ] && { read -r -a _m <<<"$METHODS"; EXTRA+=(--methods "${_m[@]}"); }
[ -n "${MAX_REAL:-}" ] && EXTRA+=(--max-real "$MAX_REAL")

python -u "$REPO/benchmarks/cmech_coverage_by_rho.py" \
    --root "${ROOT_ARR[@]}" \
    --nodes $NODES \
    --context "$CTX" \
    --thresholds $THRESHOLDS \
    --out "$OUT" \
    "${EXTRA[@]}"
rc=$?
echo "[$(date '+%F %T')] done rc=$rc"
exit $rc
