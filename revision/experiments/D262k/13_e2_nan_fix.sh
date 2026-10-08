#!/bin/bash
# Stage 13: one job. With the stable ensemble averaging in eval_native_dopfn.py, re-evaluate the
# E2 new-world shards where Do-PFN 2D (262k) had NaN intervals (into results_rerun2; the original
# results stay as they are), rebuild e2_new_world.txt filling only those bounds, commit and push.
#   cd $KIT && sbatch --account=aip-rgrosse --gres=gpu:l40s:1 --time=1:00:00 --cpus-per-task=4 --mem=32G \
#        -o logs_e2/nan_fix_%j.out R-PFN/revision/experiments/D262k/13_e2_nan_fix.sh
set -uo pipefail
KIT="${KIT:-$PWD}"; REPO="$KIT/R-PFN"; cd "$KIT"
source "$KIT/venv/bin/activate"; export PYTHONUNBUFFERED=1
NEW="$SCRATCH/e2/new"
SHARDS=$(python - <<PY
import numpy as np, glob, re
s = set()
for f in glob.glob("$NEW/results/dopfn_joint2d_262k/shard*.npz"):
    z = np.load(f)
    if not (np.isfinite(z["lo"]).all() and np.isfinite(z["hi"]).all()):
        s.add(int(re.search(r"shard(\d+)", f).group(1)))
print(" ".join(map(str, sorted(s))))
PY
)
echo "shards: $SHARDS"
SLURM_ARRAY_TASK_ID=8 E2_DATA="$NEW/data" E2_RESULTS="$NEW/results_rerun2" E2_SHARD_LIST="$SHARDS" \
  bash "$REPO/revision/experiments/E2_in_prior/eval_e2.sbatch" 2>&1 | grep -E "WARN|average_bar|datasets scored|^\[" 
python - <<PY
import numpy as np, glob
n = 0
for f in glob.glob("$NEW/results_rerun2/dopfn_joint2d_262k/shard*.npz"):
    z = np.load(f); n += int((~np.isfinite(z["lo"])).sum() + (~np.isfinite(z["hi"])).sum())
print("non-finite bounds left in the rerun:", n)
PY
D="$REPO/revision/results/d262"
python "$REPO/revision/experiments/E2_in_prior/e2_tables.py" "$NEW/data" "$NEW/results" 1000,10000 "$NEW/results_rerun2" > "$D/e2_new_world.txt"
grep -E "^dopfn_joint2d_262k|non-finite" "$D/e2_new_world.txt"
cd "$REPO" && git add revision/results/d262/e2_new_world.txt && git commit -q -m "E2 new world: 262k NaN bounds filled from the stable-averaging rerun" \
  && git pull -q --rebase && git push -q && echo pushed
