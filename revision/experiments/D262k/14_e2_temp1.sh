#!/bin/bash
# Stage 14: Do-PFN 2D (262k) on E2 new world, first 1000 worlds (shards 0-9), with Do-PFN's
# inference temperature switched OFF (softmax_temperature = 0, i.e. outputs not divided by 0.8).
# Separate results root results_t1; nothing existing is touched. Writes e2_new_world_R1000_t1.txt.
#   cd $KIT && sbatch --account=aip-rgrosse --gres=gpu:l40s:1 --time=1:00:00 --cpus-per-task=4 --mem=32G \
#        -o logs_e2/temp1_%j.out R-PFN/revision/experiments/D262k/14_e2_temp1.sh
set -uo pipefail
KIT="${KIT:-$PWD}"; REPO="$KIT/R-PFN"; cd "$KIT"
source "$KIT/venv/bin/activate"; export PYTHONUNBUFFERED=1
NEW="$SCRATCH/e2/new"
SLURM_ARRAY_TASK_ID=8 DOPFN_SOFTMAX_TEMP=0 E2_DATA="$NEW/data" E2_RESULTS="$NEW/results_t1" \
  E2_SHARD_LIST="0 1 2 3 4 5 6 7 8 9" \
  bash "$REPO/revision/experiments/E2_in_prior/eval_e2.sbatch" 2>&1 | grep -E "softmax_temperature|WARN|datasets scored|^\[" | sort -u
D="$REPO/revision/results/d262"
python "$REPO/revision/experiments/E2_in_prior/e2_tables.py" "$NEW/data" "$NEW/results_t1" 1000 > "$D/e2_new_world_R1000_t1.txt"
cat "$D/e2_new_world_R1000_t1.txt"
cd "$REPO" && git add revision/results/d262/e2_new_world_R1000_t1.txt \
  && git commit -q -m "E2 new world R=1000: Do-PFN 2D (262k) with the inference temperature off" \
  && git pull -q --rebase && git push -q && echo pushed
