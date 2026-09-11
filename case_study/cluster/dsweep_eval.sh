#!/bin/bash
# Sweep the eval over the full d_variation grid: (shift x d x N) x models.
# For each (shift, d, N) it calls eval_table3.sh on that npz root — so every
# model runs with the standardization we settled on (dopfn_bb std/0.3, cpfn2d
# pooled+log, cpfn1d per_arm+pooled, graph2d case_family, uwyk variants).
#
# THIS IS LARGE. Default grid = 4 shifts x 8 d x 5 N = 160 cells. It prints the
# array-job count and only submits when CONFIRM=1 (otherwise it's a dry run).
#
# Usage:
#   DEPLOY_ROOT=$SCRATCH/rpfn_bench_kit CONFIRM=1 \
#     bash case_study/cluster/dsweep_eval.sh <DATA_ROOT> <RESULTS_ROOT>
#
# Env knobs (subset the grid to keep it manageable):
#   SHIFTS  default "+2 -2 +5 -5"
#   DS      default "2 3 5 10 20 30 40 50"
#   NS      default "50 100 250 500 1000"
#   MODELS  default "bb cpfn2d cpfn1d graph2d"  (native/uwyk excluded — still
#           failing from the earlier run; add them once their error is fixed)
#   CONFIRM must be 1 to actually submit.
set -euo pipefail

DATA_ROOT="${1:?DATA_ROOT required (d_variation root regenerated on cluster)}"
RES="${2:?RESULTS_ROOT required}"
DEPLOY_ROOT="${DEPLOY_ROOT:?DEPLOY_ROOT required}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
SHIFTS="${SHIFTS:-+2 -2 +5 -5}"
DS="${DS:-2 3 5 10 20 30 40 50}"
NS="${NS:-50 100 250 500 1000}"
export MODELS="${MODELS:-bb cpfn2d cpfn1d graph2d}"

# ~array jobs per (shift,d,N): bb(1)+cpfn2d(2:pooled,log)+cpfn1d(2)+graph2d(1)=6
per_cell=0
for m in $MODELS; do case $m in cpfn2d|cpfn1d) per_cell=$((per_cell+2));; *) per_cell=$((per_cell+1));; esac; done
n_cells=$(( $(echo $SHIFTS|wc -w) * $(echo $DS|wc -w) * $(echo $NS|wc -w) ))
echo "[dsweep] grid: shifts=[$SHIFTS] d=[$DS] N=[$NS] models=[$MODELS]"
echo "[dsweep] cells=$n_cells  ~array-jobs=$((n_cells*per_cell))  (x6 case tasks each)"

if [ "${CONFIRM:-0}" != "1" ]; then
    echo "[dsweep] DRY RUN — set CONFIRM=1 to submit. Example single-cell command:"
    s=$(echo $SHIFTS|awk '{print $1}'); d=$(echo $DS|awk '{print $1}'); n=$(echo $NS|awk '{print $1}')
    echo "  DATA=$DATA_ROOT/shift$s/d$d N=$n MODELS=\"$MODELS\" bash $REPO/case_study/cluster/eval_table3.sh $RES/shift$s/d$d"
    exit 0
fi

for s in $SHIFTS; do for d in $DS; do
  root="$DATA_ROOT/shift$s/d$d"
  [ -d "$root" ] || { echo "  WARN missing $root — skip"; continue; }
  for n in $NS; do
    DATA="$root" N="$n" MODELS="$MODELS" \
        bash "$REPO/case_study/cluster/eval_table3.sh" "$RES/shift$s/d$d" >/dev/null
  done
  echo "  submitted shift$s/d$d (N=[$NS])"
done; done
echo "[dsweep] done. Aggregate with:"
echo "  python $REPO/case_study/cluster/dsweep_report.py --root $RES --out $RES/dsweep.csv"
