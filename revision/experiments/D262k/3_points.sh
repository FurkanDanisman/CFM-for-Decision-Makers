#!/bin/bash
# Stage 3: point estimates (sqrt PEHE, eps_ATE) of the 262k Do-PFN 2D, same scripts as the 150k model:
#   RealCause / case studies: realcause_eval/point_raw_em.py (writes point_raw_em_*.md inside the NEW dump folders)
#   ComplexMech: benchmarks/cmech_point_total.py over cmech_dumps (the new model is one more folder there)
# numpy over existing dumps, seconds per cell -- but run it as a job, not on the login node:
#   cd $KIT && sbatch --account=aip-rgrosse --time=3:00:00 --cpus-per-task=4 --mem=32G -o logs_hist/d262_points_%j.out \
#        --wrap "bash R-PFN/revision/experiments/D262k/3_points.sh"
set -uo pipefail
KIT="${KIT:-$PWD}"; source "$KIT/R-PFN/revision/experiments/D262k/common.sh"
source "$KIT/venv/bin/activate"; export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks:$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
PT="$REPO/realcause_eval/point_raw_em.py"
for ds in IHDP ACIC CPS PSID PSID_bal; do
  python -u "$PT" --root "$RC_OUT" --dataset "$ds" --modes raw --ate-metric rel --out-md "$RC_OUT/point_raw_em_${ds}.md" \
    && echo "ok rc $ds" || echo "FAILED rc $ds"
done
for cell in "$CS_OUT"/shift*/d*/ctx1000; do
  for c in Observed_Confounder Backdoor_Criterion Observed_Mediator Observed_Mediator_and_Confounder Unobserved_Confounder Frontdoor_Criterion; do
    python -u "$PT" --root "$cell" --dataset "$c" --modes raw --ate-metric l1 --out-md "$cell/point_raw_em_${c}.md" \
      >/dev/null 2>&1 || echo "FAILED cs $cell $c"
  done
done
python -u "$REPO/benchmarks/cmech_point_total.py" --dumps "$SCRATCH/cmech_dumps" --data "$SCRATCH/cmech_data_v2" \
  --nodes 5 10 20 30 40 50 --csv "$SCRATCH/cm_point_allrho_d262.csv"
echo "done"
