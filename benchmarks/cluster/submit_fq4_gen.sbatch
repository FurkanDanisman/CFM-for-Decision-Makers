#!/bin/bash
#SBATCH --job-name=fq4-gen
#SBATCH --output=logs_fq/fq4gen_%j.out
#SBATCH --error=logs_fq/fq4gen_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#
# Generate every fq4 cell for ONE (case, shift): loops d x realization in one job.
#
# One job, not an array. Generation is ~1-2 min per cell, so 60 of them is about an
# hour of sequential work -- an array turns that into 60 scheduler entries and 60
# python startups for no gain, and array tasks count against the account's submit
# limit, which is what got the rest of the sweep rejected.
#
# Resumable: a cell that already has DRAWS npz is skipped, so hitting the wall and
# resubmitting continues instead of restarting.
#
# Env: CASE, SHIFT, DS, REALS, QUERIES, DRAWS, QSEED, FQ4, CTX
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

CASE="${CASE:-Observed_Confounder}"
SHIFT="${SHIFT:-0}"
CTX="${CTX:-1000}"
read -r -a DSA <<<"${DS:-5 10 20 30 40 50}"
REALS="${REALS:-10}"
QUERIES="${QUERIES:-10}"
DRAWS="${DRAWS:-100}"
QSEED="${QSEED:-12345}"
FQ4="${FQ4:-$SC/fq4}"

echo "[$(date '+%F %T')] gen $CASE shift$SHIFT: d=${DSA[*]} reals=$REALS draws=$DRAWS"
ok=0; skip=0; fail=0; i=-1
for D in "${DSA[@]}"; do
  for (( REAL=0; REAL<REALS; REAL++ )); do
    i=$((i+1))
    TAG="${CASE}_shift${SHIFT}_d${D}_r${REAL}"
    SRC="$DEPLOY_ROOT/case_study_data/d_variation/shift$SHIFT/d$D/$CASE/N$CTX/${CASE}_${REAL}.npz"
    GEN_OUT="$FQ4/$TAG"; CELL="$GEN_OUT/$CASE/N$CTX"
    if [ ! -f "$SRC" ]; then
        echo "  $TAG: no source npz -- SKIP"; fail=$((fail+1)); continue
    fi
    HAVE=$(find "$CELL" -name "${CASE}_*.npz" 2>/dev/null | wc -l)
    if [ "$HAVE" -ge "$DRAWS" ]; then
        skip=$((skip+1))
    else
        # --query-seed varies per cell: one seed for all of them would freeze the
        # same row NUMBERS in every realization, probing a fixed position rather
        # than a sample of queries.
        if python "$REPO/case_study/gen_fixed_query.py" \
              --from-npz "$SRC" --d "$D" \
              --random-queries "$QUERIES" --query-seed "$(( QSEED + 1000 * i ))" \
              --n-draws "$DRAWS" --out "$GEN_OUT" >/dev/null 2>&1; then
            ok=$((ok+1))
        else
            echo "  $TAG: generation FAILED"; fail=$((fail+1)); continue
        fi
    fi
    mkdir -p "$FQ4/${TAG}_root/shift0/d0"
    ln -sfn "$GEN_OUT/$CASE" "$FQ4/${TAG}_root/shift0/d0/$CASE"
  done
done
echo "[$(date '+%F %T')] gen $CASE shift$SHIFT done: $ok new, $skip already there, $fail failed"
[ "$fail" = 0 ] || exit 1
