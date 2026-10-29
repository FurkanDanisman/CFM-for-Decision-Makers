#!/bin/bash
# Fixed-SCM Do-PFN 1D models (natural, setup1-3) with FIXED query units: the same-SCM data, but every one of the
# 1000 observational sets is scored on the same 500 query units, those of set 0 (covariates, noise and so both true
# outcomes fixed). Only the observational context changes from set to set. No data is regenerated: the query part
# (X_test, truths, truths_y) of set 0 is copied into every set. Four steps, one after the other (no dependency jobs):
#   bash R-PFN/revision/experiments/E2_in_prior/fixq.sh make      1 CPU job: the fixed-query copy of the same-SCM data
#   bash R-PFN/revision/experiments/E2_in_prior/fixq.sh eval      4 GPU jobs: the four models, mean/sd/interval per query
#   bash R-PFN/revision/experiments/E2_in_prior/fixq.sh pack      small per-model files, committed and pushed
#   bash R-PFN/revision/experiments/E2_in_prior/fixq.sh progress
# Locally: python revision/experiments/E2_in_prior/fscm_formula_cov.py revision/results/d262t1/fscm_moments_fixq
set -euo pipefail
KIT="${KIT:-$SCRATCH/rpfn_bench_kit}"; REPO="$KIT/R-PFN"; E2="$REPO/revision/experiments/E2_in_prior"; cd "$KIT"
SRC="$SCRATCH/e2/same/data/dopfn"; DATA="$SCRATCH/e2/same_fixq/data"; RES="$SCRATCH/e2/same_fixq/results"
mkdir -p logs_e2 "$RES"
MODELS=(9:fscm_natural 10:fscm_setup1 11:fscm_setup2 12:fscm_setup3)
cnt() { (ls $1 2>/dev/null || true) | wc -l; }
case "$1" in
make)
  [ "$(cnt "$SRC/truths_y/*.npz")" = 1000 ] || { echo "FATAL: $SRC/truths_y incomplete (fscm_moments.sh truths)"; exit 1; }
  sbatch --account=aip-rgrosse --time=1:00:00 --cpus-per-task=2 --mem=16G --job-name=fixq-make -o logs_e2/fixq_make_%j.out \
    --wrap "cd $KIT && . venv/bin/activate && python - $SRC $DATA/dopfn <<'PY'
import glob, os, shutil, sys
import numpy as np
src, dst = sys.argv[1:]
inp = lambda root, r: os.path.join(root, f'shard{r // 100}', 'complexmech', '5node', 'path_TY', 'hide_0.0', f'r{r:03d}.npz')
q = np.load(inp(src, 0)); X_q, cate_q = q['X_test'], q['true_cate']        # the query units of set 0
for sub in ('truths', 'truths_y'):
    os.makedirs(os.path.join(dst, sub), exist_ok=True)
for r in range(1000):
    z = dict(np.load(inp(src, r)))
    z['X_test'], z['true_cate'] = X_q, cate_q                               # same 500 query units in every set
    os.makedirs(os.path.dirname(inp(dst, r)), exist_ok=True)
    np.savez(inp(dst, r), **z)
    for sub in ('truths', 'truths_y'):                                      # their truths: those of set 0
        shutil.copy(os.path.join(src, sub, 'r000.npz'), os.path.join(dst, sub, f'r{r:03d}.npz'))
print('fixed-query copy written:', dst)
PY" ;;
eval)
  [ "$(cnt "$DATA/dopfn/truths_y/*.npz")" = 1000 ] || { echo "FATAL: run make first"; exit 1; }
  for spec in "${MODELS[@]}"; do
    IFS=: read -r I M <<< "$spec"
    E2_MOMENTS=1 E2_DATA=$DATA E2_RESULTS=$RES E2_SHARDS=10 \
      sbatch --account=aip-rgrosse --gres=gpu:l40s:1 --array=$I --job-name="fixq-${M#fscm_}" "$E2/eval_e2.sbatch"
  done ;;
pack)
  FSCM_RES="$RES" bash "$E2/fscm_moments.sh" _pack_to "$DATA" "$REPO/revision/results/d262t1/fscm_moments_fixq" ;;
progress)
  echo "fixed-query data: $(cnt "$DATA/dopfn/truths_y/*.npz")/1000"
  for spec in "${MODELS[@]}"; do IFS=: read -r I M <<< "$spec"; echo "$M: $(cnt "$RES/$M/moments_shard*.npz")/10 shards"; done ;;
esac
