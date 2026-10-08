#!/bin/bash
# Stage 17: CSVs and point estimates of the temperature-off model into revision/results/d262t1, then push.
# Run after stage 16 has finished (RealCause + ComplexMech now; again after the case studies).
#   cd $KIT && sbatch --account=aip-rgrosse --time=1:00:00 --cpus-per-task=4 --mem=32G -o logs_hist/t1_collect_%j.out \
#        --wrap "bash R-PFN/revision/experiments/D262k/17_collect_t1.sh"
set -uo pipefail
export D262_NAME=dopfn_repro_joint2d_262k_t1 DOPFN_SOFTMAX_TEMP=0
KIT="${KIT:-$PWD}"; source "$KIT/R-PFN/revision/experiments/D262k/common.sh"; cd "$KIT"; source venv/bin/activate
export PYTHONPATH="$REPO/benchmarks:$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
O="$REPO/revision/results/$TAG"; mkdir -p "$O"
P="$REPO/revision/experiments/E1_E8_rescoring/e8_tables.py"; A="--perreal $HIST/perreal --e1-perreal $HIST/perreal_e1"
python "$P" $A --out-dir $HIST/c_hist_$TAG       --hist --rho99 2>/dev/null && cp $HIST/c_hist_$TAG/e8_long.csv       "$O/hist.csv"
python "$P" $A --out-dir $HIST/c_hist_indep_$TAG --hist --indep 2>/dev/null && cp $HIST/c_hist_indep_$TAG/e8_long.csv "$O/hist_indep.csv"
python -u "$REPO/benchmarks/cmech_point_total.py" --dumps "$SCRATCH/cmech_dumps" --data "$SCRATCH/cmech_data_v2" \
  --nodes 5 10 20 30 40 50 --csv "$SCRATCH/cm_point_allrho_$TAG.csv" && cp "$SCRATCH/cm_point_allrho_$TAG.csv" "$O/cm_point_allrho.csv"
[ -s "$SCRATCH/cm_point_rho99_$TAG.csv" ] && cp "$SCRATCH/cm_point_rho99_$TAG.csv" "$O/cm_point_rho99.csv"
PT="$REPO/realcause_eval/point_raw_em.py"; S="$SCRATCH/dumps_all/$NAME"; R="$O/points_rel/$NAME"; mkdir -p "$R/rc"
for ds in IHDP ACIC CPS PSID PSID_bal; do
  python "$PT" --root "$S/rc" --dataset "$ds" --modes raw --ate-metric rel --out-md "$R/rc/point_raw_em_$ds.md" >/dev/null 2>&1 \
    || echo "FAILED rc $ds"; done
for cell in "$S"/cs/shift*/d{5,10,20,30,40,50}/ctx1000; do [ -d "$cell" ] || continue; r="${cell#$S/}"; mkdir -p "$R/$r"
  for c in Observed_Confounder Backdoor_Criterion Observed_Mediator Observed_Mediator_and_Confounder Unobserved_Confounder Frontdoor_Criterion; do
    python "$PT" --root "$cell" --dataset "$c" --modes raw --ate-metric rel --out-md "$R/$r/point_raw_em_$c.md" >/dev/null 2>&1 \
      || echo "FAILED $r $c"; done; done
grep -c "262k, T=1" "$O/hist.csv" | sed 's/^/hist rows for the T=1 model: /'
cd "$REPO" && git add "revision/results/$TAG" && git commit -q -m "$TAG: temperature-off Do-PFN 2D CSVs and point estimates" \
  && git pull -q --rebase && git push -q && echo pushed
