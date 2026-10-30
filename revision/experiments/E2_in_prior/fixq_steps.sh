#!/bin/bash
# setup3 over training: the fixed-query evaluation (fixq.sh data) at several checkpoints, to see whether the
# predicted spread comes down to the actual error spread, and coverage to 0.95, as training goes on.
# 200 of the 1000 observational sets (shards 0-1). One GPU job, the checkpoints one after the other.
#   bash R-PFN/revision/experiments/E2_in_prior/fixq_steps.sh eval
#   bash R-PFN/revision/experiments/E2_in_prior/fixq_steps.sh table     printed, committed and pushed
#   bash R-PFN/revision/experiments/E2_in_prior/fixq_steps.sh progress
set -euo pipefail
KIT="${KIT:-$SCRATCH/rpfn_bench_kit}"; REPO="$KIT/R-PFN"; E2="$REPO/revision/experiments/E2_in_prior"; cd "$KIT"
DATA="$SCRATCH/e2/same_fixq/data"; RES="$SCRATCH/e2/same_fixq/results_steps"; mkdir -p logs_e2 "$RES"
STEPS="1000 2500 5000 7500 10000"
case "$1" in
eval)
  sbatch --account=aip-rgrosse --gres=gpu:l40s:1 --time=3:00:00 --cpus-per-task=4 --mem=32G --requeue \
    --job-name=fixq-steps -o logs_e2/fixq_steps_%j.out "$E2/fixq_steps.sh" _eval ;;
_eval)
  for s in $STEPS; do
    [ "$(ls "$RES/step_$s/fscm_setup3"/moments_shard*.npz 2>/dev/null | wc -l)" = 2 ] && continue
    echo "[$(date '+%F %T')] step $s"
    FSCM_CKPT="step_$s.pt" E2_MOMENTS=1 E2_DATA=$DATA E2_RESULTS="$RES/step_$s" E2_SHARDS=10 E2_SHARD_LIST="0 1" \
      SLURM_ARRAY_TASK_ID=12 bash "$E2/eval_e2.sbatch" > "$RES/eval_step_$s.log" 2>&1
  done
  echo "[$(date '+%F %T')] done" ;;
table)
  source venv/bin/activate; O="$REPO/revision/results/d262t1/fscm_fixq_steps.txt"
  python - "$DATA/dopfn/truths_y" "$RES" $STEPS <<'PY' | tee "$O"
import os, sys
import numpy as np
ty, res, steps = sys.argv[1], sys.argv[2], sys.argv[3:]
t = np.load(os.path.join(ty, "r000.npz")); y1, y0 = t["y1"], t["y0_ind"]   # fixed queries; independent noise (as fscm_formula_cov.py)
print("setup3 on the fixed queries (200 sets), posterior intervals; truths with independent noise: y1, y0_ind, ITE = y1 - y0_ind")
print(f"{'step':>6s} {'cov y1':>7s} {'cov y0':>7s} {'cov ITE':>8s} {'pred sd y1':>10s} {'err sd y1':>9s} {'ratio':>6s}")
for s in steps:
    d = os.path.join(res, f"step_{s}", "fscm_setup3")
    acc = {}
    for k in (0, 1):
        f = os.path.join(d, f"moments_shard{k}.npz"); g = os.path.join(d, f"shard{k}.npz")
        if not (os.path.exists(f) and os.path.exists(g)):
            continue
        z, zi = np.load(f), np.load(g)
        for j, i in enumerate(z["idx"]):
            acc.setdefault(k * 100 + int(i), {}).update({c: z[c][j] for c in ("mu1", "sd1", "lo0", "hi0", "lo1", "hi1")})
        for j, i in enumerate(zi["idx"]):
            acc.setdefault(k * 100 + int(i), {}).update(lo=zi["lo"][j], hi=zi["hi"][j])
    if not acc:
        print(f"{s:>6s} (no results)"); continue
    A = lambda c: np.array([acc[r][c] for r in sorted(acc)])
    cov = lambda lo, hi, y: float(((A(lo) <= y) & (y <= A(hi))).mean())
    sd, err = np.median(A("sd1")), (y1 - A("mu1")).std()
    print(f"{s:>6s} {cov('lo1', 'hi1', y1):7.3f} {cov('lo0', 'hi0', y0):7.3f} {cov('lo', 'hi', y1 - y0):8.3f} "
          f"{sd:10.4f} {err:9.4f} {sd / err:6.2f}   ({len(acc)} sets)")
PY
  cd "$REPO" && git add "$O" && git commit -qm "fixq: setup3 coverage and spread over training checkpoints" \
    && git pull -q --rebase --autostash origin main && git push -q origin main && echo pushed ;;
progress)
  for s in $STEPS; do echo "step $s: $(ls "$RES/step_$s/fscm_setup3"/moments_shard*.npz 2>/dev/null | wc -l)/2 shards"; done ;;
esac
