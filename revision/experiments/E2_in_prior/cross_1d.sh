#!/bin/bash
# In-prior point estimation on the SAME datasets: the 1D models of UWYK and CausalPFN-C evaluated on their 2D
# model's in-prior datasets (uwyk_2d, cpfn_2d; R = 1000, one new SCM each, shared noise), with every outcome-level
# readout saved (E2_OUTCOMES_ALL). Do-PFN 1D and 2D already share the dopfn datasets (results_y).
#   bash R-PFN/revision/experiments/E2_in_prior/cross_1d.sh eval       2 GPU jobs (resubmit to resume: done shards are skipped)
#   bash R-PFN/revision/experiments/E2_in_prior/cross_1d.sh progress
#   bash R-PFN/revision/experiments/E2_in_prior/cross_1d.sh table      CPU job: inprior_pehe.py over all three families
set -euo pipefail
KIT="${KIT:-$SCRATCH/rpfn_bench_kit}"; REPO="$KIT/R-PFN"; E2="$REPO/revision/experiments/E2_in_prior"; cd "$KIT"
DATA="$SCRATCH/e2/new/data"; RES="$SCRATCH/e2/new/results_y"; CROSS="$SCRATCH/e2/new/results_y_cross"; NW=1000
mkdir -p logs_e2 "$CROSS"
JOBS=(2:uwyk1d:uwyk_2d 4:cpfn1d:cpfn_2d)
cnt() { (ls $1 2>/dev/null || true) | wc -l; }
case "$1" in
eval)
  for spec in "${JOBS[@]}"; do
    IFS=: read -r I M PR <<< "$spec"
    [ "$(cnt "$DATA/$PR/truths_y/*.npz")" -ge $NW ] || { echo "FATAL: $DATA/$PR/truths_y incomplete"; exit 1; }
    if squeue -u "$USER" -h -o %j | grep -qx "x1d-$M"; then echo "$M: already queued"; continue; fi
    E2_PRIOR=$PR E2_OUTCOMES_ALL=1 E2_DATA=$DATA E2_RESULTS="$CROSS/on_$PR" E2_SHARDS=$((NW / 100)) \
      sbatch --account=aip-rgrosse --gres=gpu:l40s:1 --time=3:00:00 --array=$I --job-name="x1d-$M" \
      -o "logs_e2/cross_${M}_%a.out" -e "logs_e2/cross_${M}_%a.err" "$E2/eval_e2.sbatch"
  done ;;
progress)
  for spec in "${JOBS[@]}"; do IFS=: read -r I M PR <<< "$spec"; echo "$M on $PR: $(cnt "$CROSS/on_$PR/$M/outall_shard*.npz")/$((NW / 100)) shards"; done ;;
table)
  sbatch --account=aip-rgrosse --time=1:00:00 --cpus-per-task=2 --mem=8G --job-name=inprior-pehe -o "$SCRATCH/inprior_pehe.log" \
    --wrap "cd $KIT && $KIT/venv/bin/python $E2/inprior_pehe.py $DATA $RES $REPO/revision/results/d262t1/rho_pred_inprior.csv $REPO/revision/results/d262t1/inprior_pehe.txt $CROSS" ;;
esac
