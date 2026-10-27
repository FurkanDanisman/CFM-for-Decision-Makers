#!/bin/bash
# In-prior same-world run for Do-PFN on an SCM WITHOUT a hidden confounder (find_seed_nohc.py), no retraining.
# Two models, settings as in the original same-world table:
#   dopfn_native (released Do-PFN)        -> results      (as d262/e2_same_world.txt)
#   dopfn_joint2d_262k, temperature off   -> results_t1   (as d262t1/e2_same_world.txt)
# Three steps, run one after the other (no dependency jobs):
#   bash R-PFN/revision/experiments/E2_in_prior/nohc.sh gen     CPU: find the seed, write the 1000 datasets
#   bash R-PFN/revision/experiments/E2_in_prior/nohc.sh eval    GPU: the 2 evaluations (after gen is done)
#   bash R-PFN/revision/experiments/E2_in_prior/nohc.sh table   the tables, committed and pushed
set -euo pipefail
KIT="${KIT:-$SCRATCH/rpfn_bench_kit}"; REPO="$KIT/R-PFN"; E2="$REPO/revision/experiments/E2_in_prior"; cd "$KIT"
ROOT="$SCRATCH/e2/same_nohc"; mkdir -p logs_e2 "$ROOT"
case "$1" in
gen)
  sbatch --account=aip-rgrosse --time=3:00:00 --cpus-per-task=4 --mem=64G --job-name=nohc-gen -o logs_e2/nohc_gen_%j.out \
    --wrap "cd $KIT && source venv/bin/activate && export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1 DOPFN_SRC=$KIT/external/dopfn && \
      S=\$(python $E2/find_seed_nohc.py | tee $ROOT/seed_search.txt | tail -1) && echo \$S > $ROOT/seed.txt && echo seed \$S && \
      cd $E2 && DOPFN_SAME_SEED=\$S python gen_dopfn.py same 0:1000 $ROOT/data/dopfn" ;;
eval)
  [ "$(ls $ROOT/data/dopfn/truths 2>/dev/null | wc -l)" = 1000 ] || { echo "FATAL: gen not finished"; exit 1; }
  E2_DATA=$ROOT/data E2_RESULTS=$ROOT/results E2_SHARDS=10 \
    sbatch --account=aip-rgrosse --gres=gpu:l40s:1 --array=0 "$E2/eval_e2.sbatch"
  DOPFN_SOFTMAX_TEMP=0 E2_DATA=$ROOT/data E2_RESULTS=$ROOT/results_t1 E2_SHARDS=10 \
    sbatch --account=aip-rgrosse --gres=gpu:l40s:1 --array=8 --job-name=e2-eval-t1 "$E2/eval_e2.sbatch" ;;
table)
  source venv/bin/activate; O="$REPO/revision/results/d262t1"
  { echo "seed $(cat $ROOT/seed.txt): $(tail -2 $ROOT/seed_search.txt | head -1)"; echo "== default temperature"
    python "$E2/e2_tables.py" $ROOT/data $ROOT/results 1000; echo "== temperature off"
    python "$E2/e2_tables.py" $ROOT/data $ROOT/results_t1 1000; } | tee "$O/e2_same_world_nohc.txt"
  cd "$REPO" && git add "$O/e2_same_world_nohc.txt" && git commit -qm "E2 same world without hidden confounder" \
    && git pull -q --rebase --autostash origin main && git push -q origin main && echo pushed ;;
esac
