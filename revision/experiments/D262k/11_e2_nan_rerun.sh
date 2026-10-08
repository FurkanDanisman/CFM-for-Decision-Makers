#!/bin/bash
# Stage 11: re-evaluate the E2 new-world shards in which Do-PFN 2D (262k) has non-finite interval
# bounds, into a SEPARATE results root; e2_tables.py then fills only those bounds from it.
#   cd $KIT && bash R-PFN/revision/experiments/D262k/11_e2_nan_rerun.sh      (submits one job per shard)
set -uo pipefail
KIT="${KIT:-$PWD}"; source "$KIT/venv/bin/activate"
NEW="$SCRATCH/e2/new"; RERUN="$NEW/results_rerun"
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
echo "shards with non-finite bounds: $SHARDS"
mkdir -p "$KIT/logs_e2"
for k in $SHARDS; do
  E2_DATA="$NEW/data" E2_RESULTS="$RERUN" E2_SHARD_LIST="$k" \
    sbatch --account=aip-rgrosse --gres=gpu:l40s:1 --time=3:00:00 --array=8 --job-name=e2-nan \
    -o "$KIT/logs_e2/nan_${k}_%a.out" -e "$KIT/logs_e2/nan_${k}_%a.err" \
    "$KIT/R-PFN/revision/experiments/E2_in_prior/eval_e2.sbatch"
done
