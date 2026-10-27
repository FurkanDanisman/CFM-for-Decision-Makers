#!/bin/bash
# In-prior calibration per SCM: 100 worlds per prior, 1000 observational sets (+500 query units) each.
# World w is the same-mode world of gen_*.py with E2_WORLD=w (w = 0 is the original same-world run).
# The appendix in-prior models, settings as in the appendix tables:
#   0 dopfn_native  2 uwyk1d  3 graph2d  4 cpfn1d  5 cpfn2d  6 uwyk1d_cont  8 dopfn_joint2d_262k (temperature off)
# Three steps, one after the other (no dependency jobs). Each job runs <= 3 h and requeues itself until done;
# finished datasets / shards are skipped, so a requeue continues where it stopped.
#   bash R-PFN/revision/experiments/E2_in_prior/multi.sh gen      6 CPU jobs, one per prior
#   bash R-PFN/revision/experiments/E2_in_prior/multi.sh eval     7 GPU jobs, one per model (after gen is done)
#   bash R-PFN/revision/experiments/E2_in_prior/multi.sh table    per-world tables, committed and pushed
#   bash R-PFN/revision/experiments/E2_in_prior/multi.sh progress
set -euo pipefail
KIT="${KIT:-$SCRATCH/rpfn_bench_kit}"; REPO="$KIT/R-PFN"; E2="$REPO/revision/experiments/E2_in_prior"; cd "$KIT"
ROOT="$SCRATCH/e2/multi"; NW=100; mkdir -p logs_e2 "$ROOT"
PRIORS=(dopfn uwyk_2d uwyk_1d cpfn_1d cpfn_2d uwyk_1dc)          # gen_e2.sbatch tasks 0-5
MODELS=(0:dopfn_native:dopfn 2:uwyk1d:uwyk_1d 3:graph2d:uwyk_2d 4:cpfn1d:cpfn_1d 5:cpfn2d:cpfn_2d 6:uwyk1d_cont:uwyk_1dc 8:dopfn_joint2d_262k:dopfn)
ndone() { ls "$1" 2>/dev/null | wc -l; }
case "$1" in
gen)
  for P in 0 1 2 3 4 5; do
    sbatch --account=aip-rgrosse --time=3:00:00 --cpus-per-task=4 --mem=64G --requeue --job-name="multi-gen-${PRIORS[$P]}" \
      -o "logs_e2/multi_gen_${PRIORS[$P]}_%j.out" "$E2/multi.sh" _gen "$P"
  done ;;
_gen)                                         # job body: worlds 0..NW-1 of prior $2
  P="$2"; pr="${PRIORS[$P]}"
  timeout 170m bash -c '
    for w in $(seq 0 '"$((NW - 1))"'); do
      [ "$(ls '"$ROOT"'/data/w$w/'"$pr"'/truths 2>/dev/null | wc -l)" = 1000 ] && continue
      echo "[$(date "+%F %T")] world $w"
      E2_RUN=same E2_WORLD=$w E2_DATA='"$ROOT"'/data/w$w SLURM_ARRAY_TASK_ID='"$P"' bash '"$E2"'/gen_e2.sbatch > /dev/null
    done' || { [ $? = 124 ] && { echo "time limit: requeue"; scontrol requeue "$SLURM_JOB_ID"; exit 0; }; exit 1; }
  echo "[$(date '+%F %T')] all $NW worlds of $pr done" ;;
eval)
  for spec in "${MODELS[@]}"; do
    IFS=: read -r I M PR <<< "$spec"
    for w in $(seq 0 $((NW - 1))); do
      [ "$(ndone "$ROOT/data/w$w/$PR/truths")" = 1000 ] || { echo "FATAL: data of $PR world $w incomplete (run gen first)"; exit 1; }
    done
  done
  for spec in "${MODELS[@]}"; do
    IFS=: read -r I M PR <<< "$spec"
    sbatch --account=aip-rgrosse --gres=gpu:l40s:1 --time=3:00:00 --cpus-per-task=4 --mem=32G --requeue \
      --job-name="multi-$M" -o "logs_e2/multi_eval_${M}_%j.out" "$E2/multi.sh" _eval "$I" "$M"
  done ;;
_eval)                                        # job body: model $3 (eval_e2.sbatch task $2) on every world
  I="$2"; M="$3"; T=""; [ "$I" = 8 ] && T="DOPFN_SOFTMAX_TEMP=0"
  timeout 170m bash -c '
    for w in $(seq 0 '"$((NW - 1))"'); do
      [ "$(ls '"$ROOT"'/results/w$w/'"$M"'/shard*.npz 2>/dev/null | wc -l)" = 10 ] && continue
      echo "[$(date "+%F %T")] world $w"
      env '"$T"' E2_DATA='"$ROOT"'/data/w$w E2_RESULTS='"$ROOT"'/results/w$w E2_SHARDS=10 SLURM_ARRAY_TASK_ID='"$I"' \
        bash '"$E2"'/eval_e2.sbatch > '"$ROOT"'/eval_'"$M"'_w$w.log 2>&1 || exit 1
    done' || { [ $? = 124 ] && { echo "time limit: requeue"; scontrol requeue "$SLURM_JOB_ID"; exit 0; }; exit 1; }
  echo "[$(date '+%F %T')] $M: all $NW worlds done" ;;
progress)
  for pr in "${PRIORS[@]}"; do n=0; for w in $(seq 0 $((NW - 1))); do [ "$(ndone $ROOT/data/w$w/$pr/truths)" = 1000 ] && n=$((n + 1)); done; echo "data $pr: $n/$NW worlds"; done
  for spec in "${MODELS[@]}"; do IFS=: read -r I M PR <<< "$spec"; n=$(ls $ROOT/results/w*/$M/shard*.npz 2>/dev/null | wc -l); echo "eval $M: $n/$((NW * 10)) shards"; done ;;
table)
  source venv/bin/activate; O="$REPO/revision/results/d262t1/e2_multi_world.txt"
  python "$E2/multi_table.py" "$ROOT" "$NW" | tee "$O"
  cd "$REPO" && git add "$O" && git commit -qm "E2: in-prior calibration per SCM (100 worlds x 1000 sets)" \
    && git pull -q --rebase --autostash origin main && git push -q origin main && echo pushed ;;
esac
