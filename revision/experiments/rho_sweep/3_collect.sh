#!/bin/bash
# rho sweep, step 3: score every (model, rho) -> revision/results/d262t1/rho_sweep.{csv,md}, then push.
# Run after all six step-2 jobs have finished (no dependency jobs).
#   cd $KIT && git -C R-PFN pull --rebase && bash R-PFN/revision/experiments/rho_sweep/3_collect.sh
# The same file is the job body: sbatch --wrap runs "bash <this file> run" (never `source`).
set -uo pipefail
KIT="${KIT:-/scratch/furkanbd/rpfn_bench_kit}"
SELF="$KIT/R-PFN/revision/experiments/rho_sweep/3_collect.sh"

if [ "${1:-}" != run ]; then
  cd "$KIT" && mkdir -p logs_rho_sweep
  sbatch --account=aip-rgrosse --time=1:00:00 --cpus-per-task=4 --mem=32G \
    --job-name=rho-collect -o logs_rho_sweep/collect_%j.out --wrap "KIT=$KIT bash $SELF run"
  exit $?
fi

# ---- job body ----
REPO="$KIT/R-PFN"; cd "$KIT"; source venv/bin/activate
export PYTHONPATH="$REPO/benchmarks:$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
O="$REPO/revision/results/d262t1"; mkdir -p "$O"
python -u "$REPO/revision/experiments/rho_sweep/rho_score.py" \
  "${RS_DATA:-$SCRATCH/rho_sweep/data}" "${RS_DUMPS:-$SCRATCH/rho_sweep/dumps}" "$O/rho_sweep" \
  || { echo "rho_score.py failed; nothing pushed"; exit 1; }
cat "$O/rho_sweep.md"
cd "$REPO" && git add revision/results/d262t1/rho_sweep.csv revision/results/d262t1/rho_sweep.md \
  && git commit -q -m "d262t1: rho sweep (same checkpoints, controlled cross-world correlation)" \
  && git pull -q --rebase && git push -q && echo pushed
