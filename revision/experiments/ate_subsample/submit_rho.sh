#!/bin/bash
# OT ATE density under the paper's convention: single-outcome models under assumed rho in {-0.5, 0, 0.5, 1},
# 2D models with the learned joint, the matched-resolution and matched-label ablations included; in-prior
# calibration first, then RealCause / case studies / ComplexMech. Hands-off (dependency chain approved by the
# user for this run): GPU jobs now, then collect_rho.sbatch after all of them, which resubmits anything
# unfinished, scores, and pushes the results to R-PFN/revision/results/ate_ot/. From the kit root:
#   cd /scratch/furkanbd/rpfn_bench_kit && bash R-PFN/revision/experiments/ate_subsample/submit_rho.sh
set -euo pipefail
KIT="${KIT:-$PWD}"; HERE="$KIT/R-PFN/revision/experiments/ate_subsample"
source "$HERE/models_rho.sh"
mkdir -p logs_ate
GPU=(--account=aip-rgrosse --gres=gpu:l40s:1 --parsable)
IDS=()
for spec in "${MODEL_SPECS[@]}"; do                     # in-prior: every model with a rho grid (2D models are done)
  IFS='|' read -r name tags coup <<<"$spec"
  [ -n "$coup" ] && IDS+=($(ATE_MODEL=$name sbatch "${GPU[@]}" "$HERE/run_ate_bary.sbatch"))
done
for M in "${ABLATIONS[@]}"; do                           # benchmark dumps of the ablation models
  IDS+=($(ATE_MODEL=$M sbatch "${GPU[@]}" "$HERE/rc_dumps.sbatch"))
  IDS+=($(ATE_MODEL=$M ATE_BENCH=cs ATE_SHIFT=all sbatch "${GPU[@]}" "$HERE/cs_cm_dumps.sbatch"))
  IDS+=($(ATE_MODEL=$M ATE_BENCH=cm sbatch "${GPU[@]}" "$HERE/cs_cm_dumps.sbatch"))
done
C=$(sbatch --account=aip-rgrosse --parsable --dependency=afterany:$(IFS=:; echo "${IDS[*]}") "$HERE/collect_rho.sbatch")
echo "GPU jobs (${#IDS[@]}): ${IDS[*]}"
echo "collect: $C  ->  results pushed to R-PFN/revision/results/ate_ot/ when done"
