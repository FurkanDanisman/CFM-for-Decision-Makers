#!/bin/bash
# One command for the cen3 density table: shift0 + shift+2 + shift-2, combined.
#
# Step 1 submits the density runs over all (shift, d) cells of that set.
# Step 2 (printed at the end, run once the jobs finish) aggregates them into a
# combined_cen3-shaped CSV with coverage / length / WIS / CRPS for CATE and ATE.
#
#   export DEPLOY_ROOT=$SCRATCH/rpfn_bench_kit
#   bash $DEPLOY_ROOT/R-PFN/case_study/density_eval/05_run_cen3_density.sh
#
# Env:
#   SHIFTS    default "0 +2 -2"          (the cen3 set)
#   DS        default "2 3 5 10 20 30 40 50"
#   CONTEXTS  default "50 100 250 500 1000"
#   MODELS    default all six
#   SMOKE=1   submit ONE cell only (shift+2/d10, one case, one context)
#   DRY_RUN=1 print submissions without sending them
set -euo pipefail

DEPLOY_ROOT="${DEPLOY_ROOT:?DEPLOY_ROOT required}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
HERE="$REPO/case_study/density_eval"
DATA="${DATA:-$DEPLOY_ROOT/case_study_data/d_variation}"
RES="${RES:-$DEPLOY_ROOT/results_density/dvar}"
OUT_CSV="${OUT_CSV:-$DEPLOY_ROOT/results_case_study/dvar/combined_cen3_density.csv}"

SHIFTS="${SHIFTS:-0 +2 -2}"
DS="${DS:-2 3 5 10 20 30 40 50}"

if [ "${SMOKE:-0}" = 1 ]; then
    SHIFTS="+2"; DS="10"
    export CONTEXTS="${CONTEXTS:-250}"
    export CASES="${CASES:-Observed_Confounder}"
    export MODELS="${MODELS:-cpfn2d}"
    echo "[cen3] SMOKE: one cell only"
fi

[ -d "$DATA" ] || { echo "FATAL: data root not found: $DATA" >&2; exit 1; }
n=0
for s in $SHIFTS; do
  for d in $DS; do
    root="$DATA/shift$s/d$d"
    if [ ! -d "$root" ]; then echo "  skip (no data): $root"; continue; fi
    echo "[cen3] shift$s d$d"
    DEPLOY_ROOT="$DEPLOY_ROOT" REPO="$REPO" \
      bash "$HERE/04_submit_density.sh" "$root" "$RES/shift$s/d$d"
    n=$((n+1))
  done
done
echo
echo "[cen3] submitted $n (shift,d) cells over shifts=[$SHIFTS]"
echo
echo "When the jobs finish, aggregate with:"
echo
echo "  python $HERE/dsweep_density_report.py \\"
echo "      --root      $RES \\"
echo "      --data-root $DATA \\"
echo "      --combine-shifts shift0 shift+2 shift-2 --combine-label cen3 \\"
echo "      --out $OUT_CSV"
