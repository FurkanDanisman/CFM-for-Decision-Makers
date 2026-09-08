#!/bin/bash
#SBATCH --job-name=rc-2d-malc-agg
#SBATCH --output=logs_rc_2d_malc_ci/agg_%j.out
#SBATCH --error=logs_rc_2d_malc_ci/agg_%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# CPU aggregator over malc_ci_r<###>.npz produced by
# submit_realcause_2d_malc_ci_dump.sbatch. Reports PEHE / ε_ATE (from the
# stored point CATE) alongside 2D-MALC CI Coverage / Length.
#
# Env vars:
#   OUT_ROOT          required — same one the dump job wrote to
#   METHODS           default 'cpfn2d graph2d dopfnbb'
#   DATASETS          optional filter (e.g. IHDP for smoke test)
#   GRAPH2D_TAG       default noanc
#   DOPFNBB_PEHE_KEY  default pehe (or pehe_em)
#   OUT_MD            default $OUT_ROOT/cate_ci_2d_malc.md

set -euo pipefail
mkdir -p logs_rc_2d_malc_ci

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT required — same one the MALC-CI dump job wrote to}"
METHODS="${METHODS:-cpfn2d graph2d dopfnbb}"
DATASETS_FILTER="${DATASETS:-}"
GRAPH2D_TAG="${GRAPH2D_TAG:-noanc}"
DOPFNBB_PEHE_KEY="${DOPFNBB_PEHE_KEY:-pehe}"
IN_TAG="${IN_TAG:-}"
OUT_MD="${OUT_MD:-$OUT_ROOT/cate_ci_2d_malc${IN_TAG:+_$IN_TAG}.md}"

source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1

_DS_ARGS=()
if [ -n "$DATASETS_FILTER" ]; then
    _DS_ARGS+=(--datasets $DATASETS_FILTER)
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] aggregating MALC CI  OUT_ROOT=$OUT_ROOT  METHODS=$METHODS  DATASETS=${DATASETS_FILTER:-all}"
_TAG_ARGS=()
[ -n "$IN_TAG" ] && _TAG_ARGS+=(--in-tag "$IN_TAG")

python -u "$REPO/realcause_eval/cate_ci_from_density_malc.py" \
    --out-root         "$OUT_ROOT" \
    --methods          $METHODS \
    "${_DS_ARGS[@]}" \
    "${_TAG_ARGS[@]}" \
    --graph2d-tag      "$GRAPH2D_TAG" \
    --dopfnbb-pehe-key "$DOPFNBB_PEHE_KEY" \
    --out-md           "$OUT_MD"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] wrote $OUT_MD"
