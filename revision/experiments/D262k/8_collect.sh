#!/bin/bash
# Stage 8: rebuild every CSV (now with IS_0.05 and CRPS) and the E2 table, gather the point files, push.
#   cd $KIT && bash R-PFN/revision/experiments/D262k/8_collect.sh
set -uo pipefail
source "$(dirname "$0")/common.sh"; cd "$KIT"; source venv/bin/activate
P="$REPO/revision/experiments/E1_E8_rescoring/e8_tables.py"; D="$REPO/revision/results/d262"; A="--perreal $HIST/perreal --e1-perreal $HIST/perreal_e1"
python "$P" $A --out-dir $HIST/c_hist       --hist --rho99 2>/dev/null && cp $HIST/c_hist/e8_long.csv       $D/hist.csv
python "$P" $A --out-dir $HIST/c_hist_indep --hist --indep 2>/dev/null && cp $HIST/c_hist_indep/e8_long.csv $D/hist_indep.csv
python "$P" $A --out-dir $HIST/c_malc       --malc --rho99 2>/dev/null && cp $HIST/c_malc/e8_long.csv       $D/malc.csv
python "$P" $A --out-dir $HIST/c_malc_indep --malc --indep 2>/dev/null && cp $HIST/c_malc_indep/e8_long.csv $D/malc_indep.csv
cp "$SCRATCH/cm_point_allrho_d262.csv" "$D/cm_point_allrho.csv"
[ -s "$SCRATCH/cm_point_rho99_d262.csv" ] && cp "$SCRATCH/cm_point_rho99_d262.csv" "$D/cm_point_rho99.csv"
python "$REPO/revision/experiments/E2_in_prior/e2_tables.py" $SCRATCH/e2/new/data $SCRATCH/e2/new/results 1000,10000 > "$D/e2_new_world.txt"
grep -E "model|dopfn_joint2d" "$D/e2_new_world.txt"
ls -la $D
cd "$REPO" && git add revision/results/d262 && git commit -q -m "d262: all CSVs with IS and CRPS, rho99, MALC, E2, relative points" \
  && git pull -q --rebase --autostash && git push -q && git log --oneline -1
