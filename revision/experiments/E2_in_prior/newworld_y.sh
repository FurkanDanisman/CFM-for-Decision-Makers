#!/bin/bash
# In-prior calibration, new SCM per dataset (the paper's in-prior table, R = 1000 worlds), recomputed for y0, y1 and the
# ITE (and the joint region for the 2D models). The paper's seven in-prior models, each on its own prior's existing data:
#   0 dopfn_native  8 dopfn_joint2d_262k (temperature off)  2 uwyk1d  6 uwyk1d_cont  3 graph2d  4 cpfn1d  5 cpfn2d
# Three steps, one after the other (no dependency jobs):
#   bash R-PFN/revision/experiments/E2_in_prior/newworld_y.sh truths   6 CPU jobs: true outcomes, checked against the saved ITEs
#   bash R-PFN/revision/experiments/E2_in_prior/newworld_y.sh eval     7 GPU jobs: the models again, outcome-level readouts saved
#   bash R-PFN/revision/experiments/E2_in_prior/newworld_y.sh table    per-world coverage table + CSV, committed and pushed
#   bash R-PFN/revision/experiments/E2_in_prior/newworld_y.sh progress
set -euo pipefail
KIT="${KIT:-$SCRATCH/rpfn_bench_kit}"; REPO="$KIT/R-PFN"; E2="$REPO/revision/experiments/E2_in_prior"; cd "$KIT"
DATA="$SCRATCH/e2/new/data"; RES="$SCRATCH/e2/new/results_y"; NW=1000; mkdir -p logs_e2 "$RES"
PRIORS=(dopfn uwyk_2d uwyk_1d cpfn_1d cpfn_2d uwyk_1dc)        # gen_e2.sbatch priors 0-5
MODELS=(0:dopfn_native:dopfn 8:dopfn_joint2d_262k:dopfn 2:uwyk1d:uwyk_1d 6:uwyk1d_cont:uwyk_1dc 3:graph2d:uwyk_2d 4:cpfn1d:cpfn_1d 5:cpfn2d:cpfn_2d)
cnt() { (ls $1 2>/dev/null || true) | wc -l; }
case "$1" in
truths)
  for P in 0 1 2 3 4 5; do                                     # task P*10 = worlds 0..999 of prior P
    E2_TRUTHY=1 E2_RUN=new E2_DATA=$DATA sbatch --account=aip-rgrosse --array=$((P * 10)) --job-name="ty-${PRIORS[$P]}" \
      "$E2/gen_e2.sbatch"
  done ;;
eval)
  for pr in "${PRIORS[@]}"; do
    [ "$(cnt "$DATA/$pr/truths_y/*.npz")" -ge $NW ] || { echo "FATAL: truths_y of $pr incomplete (run truths first)"; exit 1; }
  done
  for spec in "${MODELS[@]}"; do
    IFS=: read -r I M PR <<< "$spec"; T=""; [ "$I" = 8 ] && T="DOPFN_SOFTMAX_TEMP=0"
    env $T E2_OUTCOMES_ALL=1 E2_DATA=$DATA E2_RESULTS=$RES E2_SHARDS=$((NW / 100)) \
      sbatch --account=aip-rgrosse --gres=gpu:l40s:1 --array=$I --job-name="nwy-$M" "$E2/eval_e2.sbatch"
  done ;;
table)
  source venv/bin/activate; O="$REPO/revision/results/d262t1/e2_new_world_y"
  python "$E2/newworld_y_table.py" "$DATA" "$RES" "$NW" "$O.csv" | tee "$O.txt"
  cd "$REPO" && git add "$O.txt" "$O.csv" && git commit -qm "E2 new world: in-prior coverage of y0, y1, joint and ITE (R = $NW)" \
    && git pull -q --rebase --autostash origin main && git push -q origin main && echo pushed ;;
progress)
  for pr in "${PRIORS[@]}"; do echo "truths_y $pr: $(cnt "$DATA/$pr/truths_y/*.npz")/$NW"; done
  for spec in "${MODELS[@]}"; do IFS=: read -r I M PR <<< "$spec"; echo "eval $M: $(cnt "$RES/$M/outall_shard*.npz")/$((NW / 100)) shards"; done ;;
esac
