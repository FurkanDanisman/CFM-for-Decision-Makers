#!/bin/bash
#SBATCH --job-name=rc-ate-w2-agg
#SBATCH --output=logs_rc_ate_w2/agg_%j.out
#SBATCH --error=logs_rc_ate_w2/agg_%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

# Aggregator over ate_w2_{TAG}_r<###>.npz files. Emits one markdown table
# per (TAG, method-set). Runs one aggregate per specified TAG.
#
# Env vars:
#   OUT_ROOT     required
#   TAGS         default 'marginals joint malc_B100 malc_B500 malc_B1000'
#                (aggregator silently skips tags whose files don't exist)
#   DATASETS     optional filter, space-separated (e.g. 'IHDP ACIC')
#   OUT_MD_DIR   default $OUT_ROOT (each table lands as ate_w2_{TAG}.md)

set -euo pipefail
mkdir -p logs_rc_ate_w2

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT required}"
TAGS="${TAGS:-marginals joint malc_B100 malc_B500 malc_B1000}"
DATASETS_FILTER="${DATASETS:-}"
OUT_MD_DIR="${OUT_MD_DIR:-$OUT_ROOT}"

source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1

_DS_ARGS=()
[ -n "$DATASETS_FILTER" ] && _DS_ARGS+=(--datasets $DATASETS_FILTER)

# Method set inferred from tag: marginals → 1D methods; joint / malc_* → 2D.
for TAG in $TAGS; do
    case "$TAG" in
        marginals)  METHODS="cpfn1d dopfn uwyk1d" ;;
        joint|malc_*) METHODS="cpfn2d graph2d dopfnbb" ;;
        *) echo "  [skip] unknown tag: $TAG" >&2; continue ;;
    esac
    OUT_MD="$OUT_MD_DIR/ate_w2_${TAG}.md"
    echo ""
    echo "======================================================================"
    echo "[$(date '+%H:%M:%S')] TAG=$TAG   METHODS=$METHODS"
    echo "======================================================================"
    python -u "$REPO/realcause_eval/ate_density_w2_summary.py" \
        --out-root "$OUT_ROOT" \
        --tag      "$TAG" \
        --methods  $METHODS \
        "${_DS_ARGS[@]}" \
        --out-md   "$OUT_MD" || echo "  [warn] $TAG failed (files may not exist yet)"
done

echo ""
echo "[$(date '+%H:%M:%S')] wrote per-tag tables under $OUT_MD_DIR/ate_w2_*.md"
