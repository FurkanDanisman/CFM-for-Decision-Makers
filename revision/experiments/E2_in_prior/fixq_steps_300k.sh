#!/bin/bash
# The four fixed-SCM Do-PFN 1D models of the 300k-step runs (checkpoints_fixed_scm_300k) over training: the
# fixed-query evaluation (fixq.sh data) at some of the saved checkpoints (one every 10k steps). Same evaluation as
# fixq_steps_all.sh: 200 of the 1000 observational sets (shards 0-1); truths with independent noise
# (y1, y0_ind, ITE = y1 - y0_ind); the table also scores against the shared-noise truths (y0, ITE = y1 - y0). Steps: natural, setup1, setup2 at 20k, 60k, 120k, 160k; setup3 at 10k.
# One GPU job per model, its checkpoints one after the other; finished checkpoints are skipped, and a checkpoint
# not written yet is reported and left for a later `eval` (setup3's 10k).
#   bash R-PFN/revision/experiments/E2_in_prior/fixq_steps_300k.sh eval
#   bash R-PFN/revision/experiments/E2_in_prior/fixq_steps_300k.sh progress
#   bash R-PFN/revision/experiments/E2_in_prior/fixq_steps_300k.sh table     printed, committed and pushed
set -euo pipefail
KIT="${KIT:-$SCRATCH/rpfn_bench_kit}"; REPO="$KIT/R-PFN"; E2="$REPO/revision/experiments/E2_in_prior"; cd "$KIT"
DATA="$SCRATCH/e2/same_fixq/data"; RES="$SCRATCH/e2/same_fixq/results_steps_300k"; mkdir -p logs_e2 "$RES"
CK="checkpoints_fixed_scm_300k"
MODELS=("9:fscm_natural:20000 60000 120000 160000" "10:fscm_setup1:20000 60000 120000 160000"
        "11:fscm_setup2:20000 60000 120000 160000" "12:fscm_setup3:10000")
n_done() { (ls "$RES/step_$1/$2"/moments_shard*.npz 2>/dev/null || true) | wc -l | tr -d ' '; }
ckpt() { echo "$KIT/$CK/${2#fscm_}/dopfn_1d/step_$1.pt"; }
case "$1" in
eval)
  for spec in "${MODELS[@]}"; do
    IFS=: read -r I M STEPS <<< "$spec"
    todo=""; for s in $STEPS; do
      [ "$(n_done "$s" "$M")" = 2 ] && continue
      if [ -f "$(ckpt "$s" "$M")" ]; then todo="$todo $s"; else echo "$M: step $s not saved yet"; fi
    done
    [ -z "$todo" ] && { echo "$M: nothing to run"; continue; }
    if squeue -u "$USER" -h -o %j | grep -qx "fx3-${M#fscm_}"; then echo "$M: already queued, skipped"; continue; fi
    echo "$M: steps$todo"
    sbatch --account=aip-rgrosse --gres=gpu:l40s:1 --time=3:00:00 --cpus-per-task=4 --mem=32G --requeue \
      --job-name="fx3-${M#fscm_}" -o "logs_e2/fixq_steps_300k_${M#fscm_}_%j.out" "$E2/fixq_steps_300k.sh" _eval "$I" "$M" "$todo"
  done ;;
_eval)
  I="$2"; M="$3"
  for s in $4; do
    [ "$(n_done "$s" "$M")" = 2 ] && continue
    echo "[$(date '+%F %T')] $M step $s"
    FSCM_DIR="$CK" FSCM_CKPT="step_$s.pt" E2_MOMENTS=1 E2_DATA=$DATA E2_RESULTS="$RES/step_$s" E2_SHARDS=10 \
      E2_SHARD_LIST="0 1" SLURM_ARRAY_TASK_ID=$I bash "$E2/eval_e2.sbatch" > "$RES/eval_${M}_step_$s.log" 2>&1
  done
  echo "[$(date '+%F %T')] $M done" ;;
progress)
  for spec in "${MODELS[@]}"; do
    IFS=: read -r I M STEPS <<< "$spec"; line="$M:"
    for s in $STEPS; do line="$line  $((s / 1000))k:$(n_done "$s" "$M")/2"; done; echo "$line"
  done ;;
table)
  source venv/bin/activate; O="$REPO/revision/results/d262t1/fscm_fixq_steps_300k.txt"
  python - "$DATA/dopfn/truths_y" "$RES" "${MODELS[@]}" <<'PY' | tee "$O"
import os, sys
import numpy as np
ty, res, specs = sys.argv[1], sys.argv[2], [m.split(":") for m in sys.argv[3:]]
t = np.load(os.path.join(ty, "r000.npz")); y1, y0, y0s = t["y1"], t["y0_ind"], t["y0"]   # fixed queries; y0 independent / shared noise
print("fixed-SCM Do-PFN 1D models, 300k-step runs, over training; fixed queries (200 sets), posterior intervals;")
print("truths: y1; independent noise (ind) y0_ind, ITE = y1 - y0_ind; shared noise (sh) y0, ITE = y1 - y0")
print(f"{'model':14s} {'step':>7s} {'cov y1':>7s} {'y0 ind':>7s} {'y0 sh':>7s} {'ITE ind':>8s} {'ITE sh':>8s} {'pred sd y1':>10s} {'err sd y1':>9s} {'ratio':>6s}")
for _, m, steps in specs:
    for s in steps.split():
        d = os.path.join(res, f"step_{s}", m)
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
            print(f"{m:14s} {s:>7s} (no results)"); continue
        A = lambda c: np.array([acc[r][c] for r in sorted(acc)])
        cov = lambda lo, hi, y: float(((A(lo) <= y) & (y <= A(hi))).mean())
        sd, err = np.median(A("sd1")), (y1 - A("mu1")).std()
        print(f"{m:14s} {s:>7s} {cov('lo1', 'hi1', y1):7.3f} {cov('lo0', 'hi0', y0):7.3f} {cov('lo0', 'hi0', y0s):7.3f} {cov('lo', 'hi', y1 - y0):8.3f} {cov('lo', 'hi', y1 - y0s):8.3f} "
              f"{sd:10.4f} {err:9.4f} {sd / err:6.2f}   ({len(acc)} sets)")
    print()
PY
  cd "$REPO" && git add "$O" && git commit -qm "fixq: the four fixed-SCM 300k-step runs over training checkpoints" \
    && git pull -q --rebase --autostash origin main && git push -q origin main && echo pushed ;;
esac
