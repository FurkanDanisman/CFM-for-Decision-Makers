#!/bin/bash
# Stage 7: RealCause and case-study point estimates with the RELATIVE eps_ATE the paper reports, for the
# 150k and the 262k Do-PFN 2D, same script for both. Writes only under revision/results/d262/points_rel.
#   cd $KIT && sbatch --account=aip-rgrosse --time=1:00:00 --cpus-per-task=4 --mem=16G \
#        -o logs_hist/d262_points_rel_%j.out --wrap "bash R-PFN/revision/experiments/D262k/7_points_rel.sh"
set -uo pipefail
KIT="${KIT:-$PWD}"; source "$KIT/R-PFN/revision/experiments/D262k/common.sh"
source "$KIT/venv/bin/activate"; export PYTHONPATH="$REPO/benchmarks"
PT="$REPO/realcause_eval/point_raw_em.py"
for m in dopfn_repro_joint2d $NAME; do
  S="$SCRATCH/dumps_all/$m"; D="$REPO/revision/results/d262/points_rel/$m"; mkdir -p "$D/rc"
  for ds in IHDP ACIC CPS PSID PSID_bal; do
    python "$PT" --root "$S/rc" --dataset "$ds" --modes raw --ate-metric rel --out-md "$D/rc/point_raw_em_$ds.md" >/dev/null 2>&1 \
      || echo "FAILED $m rc $ds"; done
  for cell in "$S"/cs/shift*/d{5,10,20,30,40,50}/ctx1000; do r="${cell#$S/}"; mkdir -p "$D/$r"
    for c in Observed_Confounder Backdoor_Criterion Observed_Mediator Observed_Mediator_and_Confounder Unobserved_Confounder Frontdoor_Criterion; do
      python "$PT" --root "$cell" --dataset "$c" --modes raw --ate-metric rel --out-md "$D/$r/point_raw_em_$c.md" >/dev/null 2>&1 \
        || echo "FAILED $m $r $c"; done; done
done
echo "done: $(find $REPO/revision/results/d262/points_rel -name '*.md' | wc -l) tables (expect 226)"
