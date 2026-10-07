#!/bin/bash
# Progress of the 262k Do-PFN 2D rerun.   bash R-PFN/revision/experiments/D262k/progress.sh
KIT="${KIT:-/scratch/furkanbd/rpfn_bench_kit}"; source "$KIT/R-PFN/revision/experiments/D262k/common.sh"
echo "== jobs";  squeue --me -h -o "%.10i %.22j %.3t %.10M %.10l" | grep -E "d262" || echo "   (none queued or running)"
echo "== dumps (realizations written)"
for ds in IHDP ACIC CPS PSID PSID_bal; do
  printf '   rc %-9s %5s\n' $ds "$(ls $RC_OUT/dopfn_native/$ds/r*.npz 2>/dev/null | wc -l)"; done
for s in shift-2 shift0 shift+2; do
  printf '   cs %-8s ' $s
  for d in 5 10 20 30 40 50; do printf 'd%s=%-4s ' $d "$(find $CS_OUT/$s/d$d -name 'r*.npz' 2>/dev/null | wc -l)"; done; echo; done
printf '   cm all-rho  %s / 729 files\n' "$(find $CM_OUT -name '*.npz' 2>/dev/null | wc -l)"
echo "== hist scores"
printf '   RealCause %s/5   indep %s/5   case study %s/108   ComplexMech %s/6\n' \
  "$(ls $HIST/perreal_e1/$NAME/raw__*.npz $HIST/perreal/$NAME/raw__{CPS,PSID,PSID_bal}__-.npz 2>/dev/null | wc -l)" \
  "$(ls $HIST/perreal_e1/$NAME/indep__*.npz $HIST/perreal/$NAME/indep__*.npz 2>/dev/null | wc -l)" \
  "$(ls $HIST/perreal/$NAME/raw__d*_*.npz 2>/dev/null | wc -l)" \
  "$(ls $HIST/perreal/$NAME/raw__cmech_n*.npz 2>/dev/null | wc -l)"
echo "== point estimates"
printf '   RealCause %s/5   case study %s/108   ComplexMech csv %s\n' \
  "$(ls $RC_OUT/point_raw_em_*.md 2>/dev/null | wc -l)" "$(ls $CS_OUT/shift*/d*/ctx1000/point_raw_em_*.md 2>/dev/null | wc -l)" \
  "$([ -s $SCRATCH/cm_point_allrho_d262.csv ] && echo present || echo missing)"
echo "== errors in finished jobs"
for f in $(grep -l "model=$NAME" logs_dump1/d1_*.out logs_cmech_dump/cm_*.out 2>/dev/null) logs_hist/d262_*.out; do
  [ -f "$f" ] && grep -q "CELL FAILED\|ERROR\|Traceback\|failed_cells=[1-9]" "$f" && echo "   $f"
done; echo "   (end)"
