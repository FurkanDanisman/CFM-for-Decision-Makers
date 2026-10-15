#!/bin/bash
# Controlled-rho benchmarks, step 3: score every (benchmark, bin, model) from the step-2 dumps (CPU, ONE job),
# then push revision/results/d262t1/rho_ctrl.{csv,md}.
#   cd $KIT && git -C R-PFN pull --rebase && bash R-PFN/revision/experiments/rho_ctrl/3_score.sh
set -uo pipefail
KIT="${KIT:-/scratch/furkanbd/rpfn_bench_kit}"
SELF="$KIT/R-PFN/revision/experiments/rho_ctrl/3_score.sh"
if [ "${1:-}" != run ]; then
  cd "$KIT" && mkdir -p logs_rho_ctrl
  sbatch --account=aip-rgrosse --time="${RC_TIME:-2:00:00}" --cpus-per-task=4 --mem=64G \
    --job-name=rhoctrl-score -o logs_rho_ctrl/score_%j.out --wrap "KIT=$KIT bash $SELF run"
  exit $?
fi
REPO="$KIT/R-PFN"; cd "$KIT"; source venv/bin/activate
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks:$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
O="$REPO/revision/results/d262t1"; mkdir -p "$O"
python -u "$REPO/revision/experiments/rho_ctrl/score_rho_ctrl.py" "$SCRATCH/rho_ctrl" "$O/rho_ctrl" \
  || { echo "score_rho_ctrl.py failed; nothing pushed"; exit 1; }
cd "$REPO" && git add revision/results/d262t1/rho_ctrl.csv revision/results/d262t1/rho_ctrl.md \
  && git commit -q -m "d262t1: controlled-rho benchmark results" && git pull -q --rebase && git push -q && echo pushed
