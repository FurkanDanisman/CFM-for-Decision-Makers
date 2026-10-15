#!/bin/bash
# Controlled-rho benchmarks, coupling runs (revision v2, E1/E2): rescore the step-2 dumps with the 1D models'
# marginals combined under other couplings and the 2D models' marginals re-coupled. CPU, ONE job, couplings in
# parallel; pushes revision/results/d262t1/rho_ctrl_<coupling>.{csv,md}.
#   cd $KIT && git -C R-PFN pull --rebase && bash R-PFN/revision/experiments/rho_ctrl/4_score_coupling.sh
set -uo pipefail
KIT="${KIT:-/scratch/furkanbd/rpfn_bench_kit}"
SELF="$KIT/R-PFN/revision/experiments/rho_ctrl/4_score_coupling.sh"
if [ "${1:-}" != run ]; then
  cd "$KIT" && mkdir -p logs_rho_ctrl
  sbatch --account=aip-rgrosse --time="${RC_TIME:-3:00:00}" --cpus-per-task=8 --mem=64G \
    --job-name=rhoctrl-cpl -o logs_rho_ctrl/score_coupling_%j.out --wrap "KIT=$KIT bash $SELF run"
  exit $?
fi
REPO="$KIT/R-PFN"; cd "$KIT"; source venv/bin/activate
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks:$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
O="$REPO/revision/results/d262t1"; mkdir -p "$O"
S="$REPO/revision/experiments/rho_ctrl/score_rho_ctrl.py"
PIDS=()
for c in comonotonic gauss:-0.5 gauss:0.5 makarov; do
  n="rho_ctrl_$(echo "$c" | tr ':' '_')"
  COUPLING="$c" python -u "$S" "$SCRATCH/rho_ctrl" "$O/$n" > "logs_rho_ctrl/$n.log" 2>&1 & PIDS+=($!)
done
for c in indep makarov; do
  n="rho_ctrl_joint_$c"
  JOINT_COUPLING="$c" python -u "$S" "$SCRATCH/rho_ctrl" "$O/$n" > "logs_rho_ctrl/$n.log" 2>&1 & PIDS+=($!)
done
F=0; for p in "${PIDS[@]}"; do wait "$p" || F=1; done
tail -n 2 logs_rho_ctrl/rho_ctrl_*.log
[ $F = 0 ] || echo "some coupling runs failed (see logs_rho_ctrl/rho_ctrl_*.log); pushing what exists"
cd "$REPO" && git add revision/results/d262t1/rho_ctrl_*.csv revision/results/d262t1/rho_ctrl_*.md \
  && git commit -q -m "d262t1: controlled-rho coupling runs (E1/E2)" && git pull -q --rebase && git push -q && echo pushed
