#!/bin/bash
# True rho(x) on the four pre-treatment case studies, exactly from the generator (cs_true_rho.py):
# the paper's cells (shifts 0/+2/-2, d = 5..50, context 1000, first 100 units as queries), 100 datasets each.
# ONE CPU job; results pushed to revision/results/d262t1/rho_true_cs.{csv,md}.
#   cd $KIT && git -C R-PFN pull --rebase && bash R-PFN/revision/experiments/rho_true/1_cs_true_rho.sh
set -uo pipefail
KIT="${KIT:-/scratch/furkanbd/rpfn_bench_kit}"
SELF="$KIT/R-PFN/revision/experiments/rho_true/1_cs_true_rho.sh"
if [ "${1:-}" != run ]; then
  cd "$KIT" && mkdir -p logs_rho_true
  sbatch --account=aip-rgrosse --time="${RT_TIME:-2:00:00}" --cpus-per-task=2 --mem=16G \
    --job-name=rho-true-cs -o logs_rho_true/cs_%j.out --wrap "KIT=$KIT bash $SELF run"
  exit $?
fi
REPO="$KIT/R-PFN"; cd "$KIT"; source venv/bin/activate
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
O="$REPO/revision/results/d262t1"; mkdir -p "$O"
python -u "$REPO/revision/experiments/rho_true/cs_true_rho.py" "${CS_DATA:-$KIT/case_study_data/d_variation}" "$O/rho_true_cs" \
  || { echo "cs_true_rho.py failed; nothing pushed"; exit 1; }
cd "$REPO" && git add revision/results/d262t1/rho_true_cs.csv revision/results/d262t1/rho_true_cs.md \
  && git commit -q -m "d262t1: true rho(x) on the pre-treatment case studies" \
  && git pull -q --rebase && git push -q && echo pushed
