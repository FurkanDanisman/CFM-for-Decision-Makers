#!/bin/bash
#SBATCH --job-name=ate-collect
#SBATCH --output=logs_ate/%x_%j.out
#SBATCH --error=logs_ate/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
# Collect step of submit_all.sh (dependency chain approved by the user for this run). Checks that every
# dump job finished (in-prior results; .done markers of rc / cs / cm). Anything missing (a job that
# stopped at its time limit, or failed) is resubmitted and this step resubmits itself after it, at most
# ATE_MAX_ROUNDS times. When complete (or out of rounds) it submits the three scoring jobs and the final
# summary, which concatenates everything into $SCRATCH/ate_all_results.txt.
set -uo pipefail
mkdir -p logs_ate
KIT="${KIT:-$PWD}"; REPO="$KIT/R-PFN"; HERE="$REPO/revision/experiments/ate_subsample"
ROUND="${ATE_ROUND:-1}"; MAXR="${ATE_MAX_ROUNDS:-3}"
GPU=(--account=aip-rgrosse --gres=gpu:l40s:1 --parsable)
CPU=(--account=aip-rgrosse --parsable)
MODELS=(dopfn_native dopfn_joint2d_262k cpfn1d cpfn2d uwyk1d graph2d)
cnt () { ls $1 2>/dev/null | wc -l; }
IDS=()
for M in "${MODELS[@]}"; do
  case "$M" in uwyk1d|graph2d) T=(noanc v3a) ;; *) T=("") ;; esac
  ok=1; for t in "${T[@]}"; do [ -f "$SCRATCH/ate_bary/$M${t:+-$t}/coverage.txt" ] || ok=0; done
  [ "$ok" = 1 ] || { echo "in-prior missing: $M"; IDS+=($(ATE_MODEL=$M sbatch "${GPU[@]}" "$HERE/run_ate_bary.sbatch")); }
  [ "$(cnt "$SCRATCH/ate_rc_dumps/$M/*.done")" -ge 5 ] || { echo "rc missing: $M"; IDS+=($(ATE_MODEL=$M sbatch "${GPU[@]}" "$HERE/rc_dumps.sbatch")); }
  for S in 0 +2 -2; do
    [ "$(cnt "$SCRATCH/ate_cs_dumps/$M/shift${S}_*.done")" -ge 6 ] || { echo "cs missing: $M shift $S"
      IDS+=($(ATE_MODEL=$M ATE_BENCH=cs ATE_SHIFT=$S sbatch "${GPU[@]}" "$HERE/cs_cm_dumps.sbatch")); }
  done
  [ "$(cnt "$SCRATCH/ate_cm_dumps/$M/*.done")" -ge 12 ] || { echo "cm missing: $M"; IDS+=($(ATE_MODEL=$M ATE_BENCH=cm sbatch "${GPU[@]}" "$HERE/cs_cm_dumps.sbatch")); }
done
if [ "${#IDS[@]}" -gt 0 ] && [ "$ROUND" -lt "$MAXR" ]; then
  DEP=$(IFS=:; echo "${IDS[*]}")
  NXT=$(ATE_ROUND=$((ROUND + 1)) sbatch "${CPU[@]}" --dependency=afterany:$DEP "$HERE/collect.sbatch")
  echo "[$(date '+%F %T')] round $ROUND: resubmitted ${#IDS[@]} jobs ($DEP); next collect $NXT"
  exit 0
fi
[ "${#IDS[@]}" -gt 0 ] && echo "WARNING: still incomplete after $MAXR rounds (just resubmitted: ${IDS[*]}); scoring what exists"
SC=()
for B in rc cs cm; do SC+=($(ATE_BENCH=$B sbatch "${CPU[@]}" "$HERE/bench_score.sbatch")); done
FIN=$(sbatch "${CPU[@]}" --dependency=afterany:$(IFS=:; echo "${SC[*]}") "$HERE/summary.sbatch")
echo "[$(date '+%F %T')] scoring jobs ${SC[*]}; summary $FIN -> $SCRATCH/ate_all_results.txt"
