#!/bin/bash
# Score one model's dumps across all three benchmarks: point estimates
# (raw-mean and EM-mean) and RAW calibration. No MALC.
#
# Written as a script rather than a pasted loop because the case-study pass is
# 24 cells x 6 cases x 3 invocations and a mistyped continuation silently
# scores the wrong tree.
#
# Usage:
#   bash R-PFN/benchmarks/score_eta0_all.sh [MODEL] [RC_ROOT] [CM_ROOT] [CS_ROOT] [OUT_PREFIX]
# Defaults target the eta0 sweep.

set -uo pipefail
R="${REPO:-R-PFN}"
MODEL="${1:-cpfn2d}"
RC="${2:-$SCRATCH/rc_dens_eta0}"
CM="${3:-$SCRATCH/cmech_eta0}"
CS="${4:-$SCRATCH/cs_dvar_eta0}"
PRE="${5:-$SCRATCH/ETA0}"

M=(--methods "$MODEL")
CASES=(Observed_Confounder Observed_Mediator Observed_Mediator_and_Confounder
       Unobserved_Confounder Frontdoor_Criterion Backdoor_Criterion)
PT="$R/realcause_eval/point_raw_em.py"
CD="$R/UWYK_Fig3_4/cate_density_metrics.py"

echo "[1/4] RealCause"
for ds in IHDP ACIC CPS PSID PSID_bal; do
  echo "   $ds"
  python "$PT" --root "$RC" --dataset "$ds" --modes raw em "${M[@]}" \
      --out-md "$RC/point_raw_em_${ds}.md" >/dev/null 2>&1
  for t in cate ate; do
    python "$CD" --root "$RC" --dataset "$ds" --target "$t" --tau-smoother none \
        "${M[@]}" --out "$RC/calib_${ds}_raw_${t}.md" >/dev/null 2>&1
  done
done

echo "[2/4] ComplexMech (N=1000, total = nonzero + zero)"
for d in 5 10 20 30 40 50; do
  echo "   d=$d"
  python "$PT" --root "$CM/N1000" \
      --dataset "CMECH_n${d}_nonzero" "CMECH_n${d}_zero" --modes raw em "${M[@]}" \
      --out-md "$CM/point_raw_em_CMECH_d${d}.md" >/dev/null 2>&1
  for t in cate ate; do
    python "$CD" --root "$CM" --context 1000 --nodes "$d" --subset total \
        --data-root "$R/UWYK_Fig3_4/data" --target "$t" --tau-smoother none \
        "${M[@]}" --out "$CM/calib_CMECH_d${d}_raw_${t}.md" >/dev/null 2>&1
  done
done

echo "[3/4] Case studies (24 cells x 6 cases) -- the slow one"
n=0
for cell in "$CS"/shift*/d*/ctx1000; do
  [ -d "$cell" ] || continue
  n=$((n+1)); echo "   cell $n/24  ${cell#$CS/}"
  for c in "${CASES[@]}"; do
    python "$PT" --root "$cell" --dataset "$c" --modes raw em "${M[@]}" \
        --out-md "$cell/point_raw_em_${c}.md" >/dev/null 2>&1
    for t in cate ate; do
      python "$CD" --root "$cell" --dataset "$c" --target "$t" \
          --tau-smoother none "${M[@]}" --out "$cell/calib_raw_${t}_${c}.md" \
          >/dev/null 2>&1
    done
  done
done

echo "[4/4] reports"
for t in cate ate; do
  python "$R/benchmarks/build_malc_report.py" --rc-root "$RC" --cmech-root "$CM" \
      --cs-root "$CS" --tag none --target "$t" --out "${PRE}_${t}.md"
done
echo "done -> ${PRE}_cate.md  ${PRE}_ate.md"
