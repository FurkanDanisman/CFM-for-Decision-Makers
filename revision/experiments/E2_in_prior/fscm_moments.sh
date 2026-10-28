#!/bin/bash
# Fixed-SCM Do-PFN 1D models (natural, setup1-3) on the same-SCM data: Y0, Y1 and ITE coverage from the
# predicted means and variances (fscm_formula_cov.py, run locally on the packed files). Three steps, one after
# the other (no dependency jobs):
#   bash R-PFN/revision/experiments/E2_in_prior/fscm_moments.sh truths   1 CPU job: both true outcomes, both noise versions
#   bash R-PFN/revision/experiments/E2_in_prior/fscm_moments.sh eval     4 GPU jobs: the four models, saving mean/sd per query
#   bash R-PFN/revision/experiments/E2_in_prior/fscm_moments.sh pack     small per-model files, committed and pushed
#   bash R-PFN/revision/experiments/E2_in_prior/fscm_moments.sh progress
set -euo pipefail
KIT="${KIT:-$SCRATCH/rpfn_bench_kit}"; REPO="$KIT/R-PFN"; E2="$REPO/revision/experiments/E2_in_prior"; cd "$KIT"
DATA="$SCRATCH/e2/same/data"; RES="$SCRATCH/e2/same/results_fscm_m"; mkdir -p logs_e2 "$RES"
MODELS=(9:fscm_natural 10:fscm_setup1 11:fscm_setup2 12:fscm_setup3)
cnt() { (ls $1 2>/dev/null || true) | wc -l; }
case "$1" in
truths)
  sbatch --account=aip-rgrosse --time=3:00:00 --cpus-per-task=4 --mem=64G --job-name=fscm-truths -o logs_e2/fscm_truths_%j.out \
    --wrap "cd $KIT && . venv/bin/activate && cd $E2 && PYTHONHASHSEED=0 DOPFN_SRC=$KIT/external/dopfn python same_truth_y.py $DATA/dopfn 0" ;;
eval)
  [ "$(cnt "$DATA/dopfn/truths_y/*.npz")" = 1000 ] || { echo "FATAL: run truths first"; exit 1; }
  for spec in "${MODELS[@]}"; do
    IFS=: read -r I M <<< "$spec"
    E2_MOMENTS=1 E2_DATA=$DATA E2_RESULTS=$RES E2_SHARDS=10 \
      sbatch --account=aip-rgrosse --gres=gpu:l40s:1 --array=$I --job-name="fscm-m-${M#fscm_}" "$E2/eval_e2.sbatch"
  done ;;
pack)
  source venv/bin/activate; O="$REPO/revision/results/d262t1/fscm_moments"; mkdir -p "$O"
  python - "$DATA/dopfn/truths_y" "$RES" "$O" <<'PY'
import glob, os, sys
import numpy as np
ty, res, out = sys.argv[1:]
T = [np.load(os.path.join(ty, f"r{r:03d}.npz")) for r in range(1000)]
np.savez_compressed(os.path.join(out, "truths.npz"), **{k: np.stack([t[k] for t in T]) for k in ("y0", "y1", "y0_ind")})
for m in ("fscm_natural", "fscm_setup1", "fscm_setup2", "fscm_setup3"):
    fs = sorted(glob.glob(os.path.join(res, m, "moments_shard*.npz")))
    if len(fs) < 10:
        print("incomplete:", m, len(fs)); continue
    acc = {}
    for f in fs:
        k = int(f.rsplit("shard", 1)[1].split(".")[0]); z = np.load(f)
        for j, i in enumerate(z["idx"]):
            acc[k * 100 + int(i)] = [z[c][j] for c in ("mu0", "sd0", "mu1", "sd1")]
    assert sorted(acc) == list(range(1000)), (m, len(acc))
    a = np.array([acc[r] for r in range(1000)])                 # (1000, 4, 500)
    np.savez_compressed(os.path.join(out, f"{m}.npz"), mu0=a[:, 0], sd0=a[:, 1], mu1=a[:, 2], sd1=a[:, 3])
    print("packed", m)
PY
  cd "$REPO" && git add "$O" && git commit -qm "fscm: per-query predicted moments and true outcomes (same SCM)" \
    && git pull -q --rebase --autostash origin main && git push -q origin main && echo pushed ;;
progress)
  echo "truths_y: $(cnt "$DATA/dopfn/truths_y/*.npz")/1000"
  for spec in "${MODELS[@]}"; do IFS=: read -r I M <<< "$spec"; echo "$M: $(cnt "$RES/$M/moments_shard*.npz")/10 shards"; done ;;
esac
