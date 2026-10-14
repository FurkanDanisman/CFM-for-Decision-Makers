#!/bin/bash
# Controlled-rho benchmarks, step 1: generate every dataset (CPU). ONE job, three generators in turn:
#   case studies (gen_cs_rho.py), IHDP/ACIC (gen_rc_rho.py), ComplexMech n = 5 (gen_cm_rho.py).
# Output: $SCRATCH/rho_ctrl/{cs,rc,cm}. A generator whose output folder already exists is skipped.
#   cd $KIT && git -C R-PFN pull --rebase && bash R-PFN/revision/experiments/rho_ctrl/1_gen.sh
set -uo pipefail
KIT="${KIT:-/scratch/furkanbd/rpfn_bench_kit}"
SELF="$KIT/R-PFN/revision/experiments/rho_ctrl/1_gen.sh"
if [ "${1:-}" != run ]; then
  cd "$KIT" && mkdir -p logs_rho_ctrl
  sbatch --account=aip-rgrosse --time="${RC_TIME:-3:00:00}" --cpus-per-task=2 --mem=16G \
    --job-name=rhoctrl-gen -o logs_rho_ctrl/gen_%j.out --wrap "KIT=$KIT bash $SELF run"
  exit $?
fi
REPO="$KIT/R-PFN"; cd "$KIT"; source venv/bin/activate
export UWYK_SRC="$KIT/external/uwyk_reproduce/src" UWYK_ROOT="$KIT/external/uwyk_reproduce"
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
O="$SCRATCH/rho_ctrl"; mkdir -p "$O"
G="$REPO/revision/experiments/rho_ctrl"
[ -d "$O/cs" ] || python -u "$G/gen_cs_rho.py" "$O/cs" || echo "case-study generation FAILED"
[ -d "$O/rc" ] || python -u "$G/gen_rc_rho.py" "$O/rc" --causalpfn "$KIT/external/causalpfn" || echo "IHDP/ACIC generation FAILED"
[ -d "$O/cm" ] || python -u "$G/gen_cm_rho.py" "$O/cm" || echo "ComplexMech generation FAILED"
for b in 0 1 2; do
  echo "bin$b: cs $(ls $O/cs/bin$b/*/N1000/*.npz 2>/dev/null | wc -l)  IHDP $(ls $O/rc/bin$b/IHDP/truths/*.npz 2>/dev/null | wc -l)" \
       " ACIC $(ls $O/rc/bin$b/ACIC/truths/*.npz 2>/dev/null | wc -l)  cm $(ls $O/cm/bin$b/complexmech/5node/path_TY/hide_0.0/*.npz 2>/dev/null | wc -l)"
done
echo "[$(date '+%F %T')] done"
