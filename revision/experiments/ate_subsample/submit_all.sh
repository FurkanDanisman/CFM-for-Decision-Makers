#!/bin/bash
# OT ATE coverage check for all models, hands-off (dependency chain approved by the user for this run).
# Submits the GPU jobs still needed (UWYK 1D / 2D in-prior and on RealCause, case studies, ComplexMech),
# then collect.sbatch after all of them; collect resubmits anything unfinished (at most 3 rounds), then
# scores rc / cs / cm and writes $SCRATCH/ate_all_results.txt. Run from the kit root:
#   cd /scratch/furkanbd/rpfn_bench_kit && bash R-PFN/revision/experiments/ate_subsample/submit_all.sh
set -euo pipefail
KIT="${KIT:-$PWD}"; HERE="$KIT/R-PFN/revision/experiments/ate_subsample"
mkdir -p logs_ate
GPU=(--account=aip-rgrosse --gres=gpu:l40s:1 --parsable)
IDS=()
for M in uwyk1d graph2d; do
  IDS+=($(ATE_MODEL=$M sbatch "${GPU[@]}" "$HERE/run_ate_bary.sbatch"))
  IDS+=($(ATE_MODEL=$M sbatch "${GPU[@]}" "$HERE/rc_dumps.sbatch"))
  for S in 0 +2 -2; do IDS+=($(ATE_MODEL=$M ATE_BENCH=cs ATE_SHIFT=$S sbatch "${GPU[@]}" "$HERE/cs_cm_dumps.sbatch")); done
  IDS+=($(ATE_MODEL=$M ATE_BENCH=cm sbatch "${GPU[@]}" "$HERE/cs_cm_dumps.sbatch"))
done
DEP=$(IFS=:; echo "${IDS[*]}")
C=$(sbatch --account=aip-rgrosse --parsable --dependency=afterany:$DEP "$HERE/collect.sbatch")
echo "GPU jobs: ${IDS[*]}"
echo "collect:  $C (waits for all of them; resubmits anything unfinished, then scores)"
echo "final file: $SCRATCH/ate_all_results.txt"
