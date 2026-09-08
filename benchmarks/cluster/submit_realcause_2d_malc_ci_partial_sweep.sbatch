#!/bin/bash
#SBATCH --job-name=rc-2d-malc-partial
#SBATCH --output=logs_rc_2d_malc_ci/partial_%j.out
#SBATCH --error=logs_rc_2d_malc_ci/partial_%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# Aggregator for the "affordable" MALC-CI slice — reports:
#   B=100  →  all 5 datasets       (IHDP, ACIC, CPS, PSID, PSID_bal)
#   B=500  →  IHDP, ACIC, PSID, PSID_bal   (skips CPS — expensive)
#   B=1000 →  IHDP, ACIC                    (skips CPS + PSID + PSID_bal)
#
# For each (B, dataset-subset) pair, walks
# <OUT_ROOT>/{cpfn2d,graph2d,dopfnbb}/<D>/malc_ci_B{100,500,1000}_r<###>.npz
# and reports PEHE / ε_ATE (stored point CATE from the density sibling)
# alongside MALC Coverage / Length, then concatenates the three tables into
# $OUT_ROOT/cate_ci_2d_malc_partial.md.
#
# Env vars:
#   OUT_ROOT           required — same one the MALC-CI dump job wrote to
#   METHODS            default 'cpfn2d graph2d dopfnbb'
#   GRAPH2D_TAG        default 'noanc'
#   DOPFNBB_PEHE_KEY   default 'pehe'
#   OUT_MD             default $OUT_ROOT/cate_ci_2d_malc_partial.md

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
GRAPH2D_TAG="${GRAPH2D_TAG:-noanc}"
DOPFNBB_PEHE_KEY="${DOPFNBB_PEHE_KEY:-pehe}"
OUT_MD="${OUT_MD:-$OUT_ROOT/cate_ci_2d_malc_partial.md}"

source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1

# (B, datasets to include)
declare -A DS_FOR
DS_FOR[100]=""                             # empty = default = all 5 datasets
DS_FOR[500]="IHDP ACIC PSID PSID_bal"
DS_FOR[1000]="IHDP ACIC"

# Fresh combined markdown; each aggregator call appends its section.
: > "$OUT_MD"
{
    echo "# 2D-MALC CI — partial B-sweep"
    echo ""
    echo "Skips expensive B×dataset combos:"
    echo "- B=100  → all 5 datasets"
    echo "- B=500  → skips CPS"
    echo "- B=1000 → skips CPS, PSID, PSID_bal"
    echo ""
} >> "$OUT_MD"

for B in 100 500 1000; do
    DS="${DS_FOR[$B]}"
    PARTIAL_MD="$OUT_ROOT/cate_ci_2d_malc_B${B}_partial.md"
    _DS_ARGS=()
    [ -n "$DS" ] && _DS_ARGS+=(--datasets $DS)

    echo ""
    echo "========================================"
    echo "[$(date '+%H:%M:%S')] B=$B   datasets=${DS:-all}"
    echo "========================================"
    python -u "$REPO/realcause_eval/cate_ci_from_density_malc.py" \
        --out-root         "$OUT_ROOT" \
        --methods          $METHODS \
        "${_DS_ARGS[@]}" \
        --in-tag           "B$B" \
        --graph2d-tag      "$GRAPH2D_TAG" \
        --dopfnbb-pehe-key "$DOPFNBB_PEHE_KEY" \
        --out-md           "$PARTIAL_MD"

    {
        echo ""
        echo "## B = $B  (datasets: ${DS:-all})"
        echo ""
        # Skip the shebang/first line of the per-B .md; keep the table.
        cat "$PARTIAL_MD"
    } >> "$OUT_MD"
done

echo ""
echo "[$(date '+%H:%M:%S')] wrote combined: $OUT_MD"
