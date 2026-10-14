#!/bin/bash
# True rho(x) on the pre-treatment ComplexMech realizations (cmech_true_rho.py): every cell n = 5..50 of
# $SCRATCH/cmech_data_v2 hide_0.0 is replayed with UWYK's generator (one sampler per cell, in order, as
# run_sweep), each needed realization is checked against its stored outcomes, then only Y's noise is redrawn.
# ONE CPU job; results pushed to revision/results/d262t1/rho_true_cmech.{csv,md}.
#   cd $KIT && git -C R-PFN pull --rebase && bash R-PFN/revision/experiments/rho_true/2_cmech_true_rho.sh
set -uo pipefail
KIT="${KIT:-/scratch/furkanbd/rpfn_bench_kit}"
SELF="$KIT/R-PFN/revision/experiments/rho_true/2_cmech_true_rho.sh"
if [ "${1:-}" != run ]; then
  cd "$KIT" && mkdir -p logs_rho_true
  sbatch --account=aip-rgrosse --time="${RT_TIME:-3:00:00}" --cpus-per-task=2 --mem=32G \
    --job-name=rho-true-cm -o logs_rho_true/cm_%j.out --wrap "KIT=$KIT bash $SELF run"
  exit $?
fi
REPO="$KIT/R-PFN"; cd "$KIT"; source venv/bin/activate
export UWYK_SRC="$KIT/external/uwyk_reproduce/src" UWYK_ROOT="$KIT/external/uwyk_reproduce"
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
O="$REPO/revision/results/d262t1"; mkdir -p "$O"
python -u "$REPO/revision/experiments/rho_true/cmech_true_rho.py" "${CM_DATA:-$SCRATCH/cmech_data_v2}" "$O/rho_true_cmech" \
  || { echo "cmech_true_rho.py failed; nothing pushed"; exit 1; }
cd "$REPO" && git add revision/results/d262t1/rho_true_cmech.csv revision/results/d262t1/rho_true_cmech.md \
  && git commit -q -m "d262t1: true rho(x) on the pre-treatment ComplexMech realizations" \
  && git pull -q --rebase && git push -q && echo pushed
