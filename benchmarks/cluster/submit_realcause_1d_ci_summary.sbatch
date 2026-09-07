#!/bin/bash
#SBATCH --job-name=rc-1d-ci-agg
#SBATCH --output=logs_rc_1d_ci/agg_%j.out
#SBATCH --error=logs_rc_1d_ci/agg_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Aggregator for the 1D density-CI table. NOT a login-node script.
# Computes PEHE / ε_ATE / Coverage / Length derived from each method's density
# via the exact N²-atom PMF of p(τ) under independence, then writes an
# uwyk_table3_ci.md alongside the results directory.
#
# Env vars:
#   OUT_ROOT       required — same OUT_ROOT the density-dump job wrote to
#   METHODS        default 'cpfn1d dopfn uwyk1d'
#   UWYK_TAG       default 'noanc' — anc tag row of uwyk1d to score
#   OUT_MD         default $OUT_ROOT/cate_ci.md
#
# Also supports --debug-one via DEBUG_METHOD / DEBUG_DATASET env pair, which
# prints the fingerprint of the first realization NPZ instead of aggregating.

set -euo pipefail
mkdir -p logs_rc_1d_ci

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT required — same one the density-dump job wrote to}"
METHODS="${METHODS:-cpfn1d dopfn uwyk1d}"
DATASETS_FILTER="${DATASETS:-}"    # e.g. DATASETS=IHDP for a fast validation pass
UWYK_TAG="${UWYK_TAG:-noanc}"
OUT_MD="${OUT_MD:-$OUT_ROOT/cate_ci.md}"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

if [ -n "${DEBUG_METHOD:-}" ] && [ -n "${DEBUG_DATASET:-}" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] --debug-one $DEBUG_METHOD $DEBUG_DATASET"
    python -u "$REPO/realcause_eval/cate_ci_from_density.py" \
        --out-root "$OUT_ROOT" \
        --uwyk-tag "$UWYK_TAG" \
        --debug-one "$DEBUG_METHOD" "$DEBUG_DATASET"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] aggregating  OUT_ROOT=$OUT_ROOT  METHODS=$METHODS  DATASETS=${DATASETS_FILTER:-all}"
    _DS_ARGS=()
    if [ -n "$DATASETS_FILTER" ]; then
        _DS_ARGS+=(--datasets $DATASETS_FILTER)
    fi
    python -u "$REPO/realcause_eval/cate_ci_from_density.py" \
        --out-root "$OUT_ROOT" \
        --methods $METHODS \
        "${_DS_ARGS[@]}" \
        --uwyk-tag "$UWYK_TAG" \
        --out-md "$OUT_MD"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] wrote $OUT_MD"
fi
