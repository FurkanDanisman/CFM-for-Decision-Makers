#!/bin/bash
#SBATCH --job-name=ate-collect-rho
#SBATCH --output=logs_ate/%x_%j.out
#SBATCH --error=logs_ate/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
# Collect step of submit_rho.sh (dependency chain approved by the user for this run). Checks every in-prior
# variant and every dump marker; resubmits what is missing and itself after it (at most ATE_MAX_ROUNDS rounds);
# when complete (or out of rounds) submits the three scoring jobs and, after them, the summary.
# RealCause ACIC is not required for CausalPFN-C models (its harness has failed there in both runs).
set -uo pipefail
mkdir -p logs_ate
KIT="${KIT:-$PWD}"; HERE="$KIT/R-PFN/revision/experiments/ate_subsample"
source "$HERE/models_rho.sh"
ROUND="${ATE_ROUND:-1}"; MAXR="${ATE_MAX_ROUNDS:-5}"
GPU=(--account=aip-rgrosse --gres=gpu:l40s:1 --parsable); CPU=(--account=aip-rgrosse --parsable)
IDS=()
for spec in "${MODEL_SPECS[@]}"; do
  name="${spec%%|*}"; ok=1
  for v in $(variants_of "$spec"); do [ -f "$SCRATCH/ate_bary/$v/coverage.txt" ] || ok=0; done
  [ "$ok" = 1 ] || { echo "in-prior missing: $name"; IDS+=($(ATE_MODEL=$name sbatch "${GPU[@]}" "$HERE/run_ate_bary.sbatch")); }
  rc_ok=1; for ds in IHDP ACIC CPS PSID PSID_bal; do
    [ "$ds" = ACIC ] && [[ "$name" == cpfn* ]] && continue
    [ -f "$SCRATCH/ate_rc_dumps/$name/$ds.done" ] || rc_ok=0; done
  [ "$rc_ok" = 1 ] || { echo "rc missing: $name"; IDS+=($(ATE_MODEL=$name sbatch "${GPU[@]}" "$HERE/rc_dumps.sbatch")); }
  [ "$(ls "$SCRATCH/ate_cs_dumps/$name"/shift*_d*.done 2>/dev/null | wc -l)" -ge 18 ] || { echo "cs missing: $name"
    IDS+=($(ATE_MODEL=$name ATE_BENCH=cs ATE_SHIFT=all sbatch "${GPU[@]}" "$HERE/cs_cm_dumps.sbatch")); }
  [ "$(ls "$SCRATCH/ate_cm_dumps/$name"/n*.done 2>/dev/null | wc -l)" -ge 12 ] || { echo "cm missing: $name"
    IDS+=($(ATE_MODEL=$name ATE_BENCH=cm sbatch "${GPU[@]}" "$HERE/cs_cm_dumps.sbatch")); }
done
if [ "${#IDS[@]}" -gt 0 ] && [ "$ROUND" -lt "$MAXR" ]; then
  NXT=$(ATE_ROUND=$((ROUND + 1)) sbatch "${CPU[@]}" --dependency=afterany:$(IFS=:; echo "${IDS[*]}") "$HERE/collect_rho.sbatch")
  echo "[$(date '+%F %T')] round $ROUND: resubmitted ${#IDS[@]} jobs; next collect $NXT"; exit 0
fi
[ "${#IDS[@]}" -gt 0 ] && echo "WARNING: incomplete after $MAXR rounds (just resubmitted ${IDS[*]}); scoring what exists"
SC=(); for B in rc cs cm; do SC+=($(ATE_BENCH=$B sbatch "${CPU[@]}" "$HERE/score_rho.sbatch")); done
FIN=$(sbatch "${CPU[@]}" --dependency=afterany:$(IFS=:; echo "${SC[*]}") "$HERE/summary_rho.sbatch")
echo "[$(date '+%F %T')] scoring ${SC[*]}; summary $FIN -> R-PFN/revision/results/ate_ot (pushed)"
