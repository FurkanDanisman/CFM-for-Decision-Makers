#!/bin/bash
# Stage 16: hist scoring of the temperature-off model (dopfn_repro_joint2d_262k_t1), same scripts as the 262k:
# RealCause (task 0), ComplexMech all rho (4), RealCause independent coupling (5); ComplexMech rho > 0.99.
# Case studies (tasks 1-3) once their dumps are complete:  ... bash 16_score_t1.sh cs
#   cd $KIT && git -C R-PFN pull --rebase && bash R-PFN/revision/experiments/D262k/16_score_t1.sh
set -euo pipefail
export D262_NAME=dopfn_repro_joint2d_262k_t1 DOPFN_SOFTMAX_TEMP=0
D="$(cd "$(dirname "$0")" && pwd)"; source "$D/common.sh"; cd "$KIT"
if [ "${1:-}" = cs ]; then
  sbatch --account=$ACCOUNT --array=1-3 --job-name=t1-score "$D/2_score.sbatch"
else
  sbatch --account=$ACCOUNT --array=0,4,5 --job-name=t1-score "$D/2_score.sbatch"
  sbatch --account=$ACCOUNT --job-name=t1-rho99 "$D/6_rho99_score.sbatch"
fi
