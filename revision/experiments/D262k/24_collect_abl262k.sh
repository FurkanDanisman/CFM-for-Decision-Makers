#!/bin/bash
# Stage 24: hist CSV of the 262k Do-PFN ablation model(s) from stage 23 -> revision/results/d262t1/hist_abl262k.csv, push.
#   cd $KIT && git -C R-PFN pull --rebase && sbatch --account=aip-rgrosse --time=1:00:00 --cpus-per-task=4 --mem=32G \
#        -o logs_hist/abl262k_collect_%j.out --wrap "bash R-PFN/revision/experiments/D262k/24_collect_abl262k.sh"
set -uo pipefail
KIT="${KIT:-$PWD}"; REPO="$KIT/R-PFN"; HIST="$SCRATCH/hist"; cd "$KIT"; source venv/bin/activate
export PYTHONPATH="$REPO/benchmarks:$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
O="$REPO/revision/results/d262t1"; mkdir -p "$O"
python "$REPO/revision/experiments/E1_E8_rescoring/e8_tables.py" --perreal $HIST/perreal --e1-perreal $HIST/perreal_e1 \
  --out-dir $HIST/c_hist_abl262k --hist 2>/dev/null || { echo "FAILED e8_tables"; exit 1; }
grep "262k, T=1)" $HIST/c_hist_abl262k/e8_long.csv | grep "matched" > /dev/null || { echo "no ablation rows"; exit 1; }
cp $HIST/c_hist_abl262k/e8_long.csv "$O/hist_abl262k.csv"
for m in "matched resolution, 262k" "matched label, 262k"; do echo "rows for $m: $(grep -c "$m" "$O/hist_abl262k.csv")"; done
cd "$REPO" && git add "$O/hist_abl262k.csv" && git commit -q -m "d262t1: hist CSV of the 262k Do-PFN ablation model" \
  && git pull -q --rebase && git push -q && echo pushed
