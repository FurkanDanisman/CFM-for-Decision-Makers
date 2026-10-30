#!/bin/bash
# In-prior calibration, same SCM, FIXED query units: the seven models of the same-SCM table (tab:inprior-same) on
# the same-SCM data, but every one of the 1000 observational sets is scored on the same 500 query units, those of
# set 0 (covariates, noise and so the true ITE fixed). Only the observational context changes from set to set.
# No data is regenerated: the query part (X_test, truths) of set 0 is copied into every set of each prior.
# Models (eval_e2.sbatch tasks): 0 dopfn_native, 8 dopfn_joint2d_262k (temperature off), 2 uwyk1d, 6 uwyk1d_cont,
# 3 graph2d, 4 cpfn1d, 5 cpfn2d. Four steps, one after the other (no dependency jobs):
#   bash R-PFN/revision/experiments/E2_in_prior/fixq_all.sh make      1 CPU job: fixed-query copies of the six priors
#   bash R-PFN/revision/experiments/E2_in_prior/fixq_all.sh eval      7 GPU jobs, one per model
#   bash R-PFN/revision/experiments/E2_in_prior/fixq_all.sh table     per-set and per-query coverage, committed and pushed
#   bash R-PFN/revision/experiments/E2_in_prior/fixq_all.sh progress
set -euo pipefail
KIT="${KIT:-$SCRATCH/rpfn_bench_kit}"; REPO="$KIT/R-PFN"; E2="$REPO/revision/experiments/E2_in_prior"; cd "$KIT"
SRC="$SCRATCH/e2/same/data"; DATA="$SCRATCH/e2/same_fixq_all/data"; RES="$SCRATCH/e2/same_fixq_all/results"
mkdir -p logs_e2 "$RES"
PRIORS=(dopfn uwyk_2d uwyk_1d cpfn_1d cpfn_2d uwyk_1dc)
MODELS=(0:dopfn_native 8:dopfn_joint2d_262k 2:uwyk1d 6:uwyk1d_cont 3:graph2d 4:cpfn1d 5:cpfn2d)
cnt() { (ls $1 2>/dev/null || true) | wc -l; }
case "$1" in
make)
  for pr in "${PRIORS[@]}"; do
    [ "$(cnt "$SRC/$pr/truths/*.npz")" -ge 1000 ] || { echo "FATAL: $SRC/$pr incomplete"; exit 1; }
  done
  sbatch --account=aip-rgrosse --time=1:00:00 --cpus-per-task=2 --mem=16G --job-name=fixq-all-make \
    -o logs_e2/fixq_all_make_%j.out --wrap "cd $KIT && . venv/bin/activate && python - $SRC $DATA ${PRIORS[*]} <<'PY'
import os, shutil, sys
import numpy as np
src_root, dst_root, priors = sys.argv[1], sys.argv[2], sys.argv[3:]
inp = lambda root, r: os.path.join(root, f'shard{r // 100}', 'complexmech', '5node', 'path_TY', 'hide_0.0', f'r{r:03d}.npz')
for pr in priors:
    src, dst = os.path.join(src_root, pr), os.path.join(dst_root, pr)
    q = np.load(inp(src, 0)); X_q, cate_q = q['X_test'], q['true_cate']    # the query units of set 0
    os.makedirs(os.path.join(dst, 'truths'), exist_ok=True)
    for r in range(1000):
        z = dict(np.load(inp(src, r)))
        z['X_test'], z['true_cate'] = X_q, cate_q                           # same 500 query units in every set
        os.makedirs(os.path.dirname(inp(dst, r)), exist_ok=True)
        np.savez(inp(dst, r), **z)
        shutil.copy(os.path.join(src, 'truths', 'r000.npz'), os.path.join(dst, 'truths', f'r{r:03d}.npz'))
    print('fixed-query copy written:', dst, flush=True)
PY" ;;
eval)
  for pr in "${PRIORS[@]}"; do
    [ "$(cnt "$DATA/$pr/truths/*.npz")" = 1000 ] || { echo "FATAL: run make first ($pr)"; exit 1; }
  done
  for spec in "${MODELS[@]}"; do
    IFS=: read -r I M <<< "$spec"; T=""; [ "$I" = 8 ] && T="DOPFN_SOFTMAX_TEMP=0"
    env $T E2_DATA=$DATA E2_RESULTS=$RES E2_SHARDS=10 \
      sbatch --account=aip-rgrosse --gres=gpu:l40s:1 --array=$I --job-name="fxq-$M" "$E2/eval_e2.sbatch"
  done ;;
table)
  source venv/bin/activate; O="$REPO/revision/results/d262t1/e2_same_world_fixq"
  { python "$E2/e2_tables.py" "$DATA" "$RES" 1000; echo; python "$E2/fixq_table.py" "$DATA" "$RES"; } | tee "$O.txt"
  cd "$REPO" && git add "$O.txt" && git commit -qm "E2 same world, fixed query units: per-set and per-query coverage" \
    && git pull -q --rebase --autostash origin main && git push -q origin main && echo pushed ;;
progress)
  for pr in "${PRIORS[@]}"; do echo "data $pr: $(cnt "$DATA/$pr/truths/*.npz")/1000"; done
  for spec in "${MODELS[@]}"; do IFS=: read -r I M <<< "$spec"; echo "eval $M: $(cnt "$RES/$M/shard*.npz")/10 shards"; done ;;
esac
