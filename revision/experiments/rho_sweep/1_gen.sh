#!/bin/bash
# rho sweep, step 1: the datasets (CPU, numpy only, well under a minute).
#   cd $KIT && git -C R-PFN pull --rebase && bash R-PFN/revision/experiments/rho_sweep/1_gen.sh
# Writes $SCRATCH/rho_sweep/data/rho<+r.rr>/{shard0/complexmech/5node/path_TY/hide_0.0,truths}/r###.npz.
# Finished datasets are skipped, so rerunning is safe.
# The same file is the job body: sbatch --wrap runs "bash <this file> run" (never `source`).
set -uo pipefail
KIT="${KIT:-/scratch/furkanbd/rpfn_bench_kit}"
SELF="$KIT/R-PFN/revision/experiments/rho_sweep/1_gen.sh"

if [ "${1:-}" != run ]; then
  cd "$KIT" && mkdir -p logs_rho_sweep
  sbatch --account=aip-rgrosse --time=0:20:00 --cpus-per-task=1 --mem=4G \
    --job-name=rho-gen -o logs_rho_sweep/gen_%j.out --wrap "KIT=$KIT bash $SELF run"
  exit $?
fi

# ---- job body ----
set -e
cd "$KIT"; source venv/bin/activate
python -u "$KIT/R-PFN/revision/experiments/rho_sweep/gen_rho.py" "${RS_DATA:-$SCRATCH/rho_sweep/data}"
echo "[$(date '+%F %T')] done"
