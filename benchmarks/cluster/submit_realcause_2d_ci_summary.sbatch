#!/bin/bash
#SBATCH --job-name=rc-2d-ci-agg
#SBATCH --output=logs_rc_2d_ci/agg_%j.out
#SBATCH --error=logs_rc_2d_ci/agg_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Aggregator for the 2D density-CI table. NOT a login-node script.
# Computes PEHE / ε_ATE / Coverage / Length derived from each 2D method's
# joint density via anti-diagonal projection of p(Y0, Y1) → p(τ) — no
# independence assumption needed — then writes cate_ci.md alongside the
# results directory.
#
# Env vars:
#   OUT_ROOT           required — same OUT_ROOT the 2D density-dump job wrote to
#   METHODS            default 'cpfn2d graph2d dopfnbb'
#   GRAPH2D_TAG        default 'noanc' — anc tag row of graph2d to score
#   DOPFNBB_PEHE_KEY   default 'pehe' — 'pehe' (inner-9-region) or 'pehe_em'
#   OUT_MD             default $OUT_ROOT/cate_ci_2d.md
#
# Also supports --debug-one via DEBUG_METHOD / DEBUG_DATASET env pair.

set -euo pipefail
mkdir -p logs_rc_2d_ci

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT required — same one the 2D density-dump job wrote to}"
METHODS="${METHODS:-cpfn2d graph2d dopfnbb}"
DATASETS_FILTER="${DATASETS:-}"    # e.g. DATASETS=IHDP for a fast validation pass
GRAPH2D_TAG="${GRAPH2D_TAG:-noanc}"
DOPFNBB_PEHE_KEY="${DOPFNBB_PEHE_KEY:-pehe}"
OUT_MD="${OUT_MD:-$OUT_ROOT/cate_ci_2d.md}"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

if [ -n "${DEBUG_METHOD:-}" ] && [ -n "${DEBUG_DATASET:-}" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] --debug-one $DEBUG_METHOD $DEBUG_DATASET"
    python -u "$REPO/realcause_eval/cate_ci_from_density.py" \
        --out-root "$OUT_ROOT" \
        --graph2d-tag "$GRAPH2D_TAG" \
        --dopfnbb-pehe-key "$DOPFNBB_PEHE_KEY" \
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
        --graph2d-tag "$GRAPH2D_TAG" \
        --dopfnbb-pehe-key "$DOPFNBB_PEHE_KEY" \
        --out-md "$OUT_MD"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] wrote $OUT_MD"
fi
