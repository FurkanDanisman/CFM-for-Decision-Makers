#!/bin/bash
#SBATCH --job-name=d262-score
#SBATCH --output=logs_hist/d262_%a.out
#SBATCH --error=logs_hist/d262_%a.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --array=0-5
# Stage 2: hist scoring of the 262k Do-PFN 2D, identical settings to submit_hist.sbatch (CDM_INTERVAL=hist).
#   0 RealCause (IHDP/ACIC vs realized delta, CPS/PSID/PSID_bal vs ite)   1-3 case study shift -2/0/+2
#   4 ComplexMech all rho   5 RealCause with the coupling forced independent (indep__ files)
#   cd $KIT && sbatch --account=aip-rgrosse R-PFN/revision/experiments/D262k/2_score.sbatch
set -uo pipefail
mkdir -p logs_hist
KIT="${KIT:-$PWD}"; source "$KIT/R-PFN/revision/experiments/D262k/common.sh"
source "$KIT/venv/bin/activate"
export CAUSALPFN="$KIT/external/causalpfn"
export CDM_INTERVAL=hist PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ATOM_GRID_MAX="${ATOM_GRID_MAX:-4096}"
export PYTHONPATH="$REPO/benchmarks:$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
CBR="$REPO/benchmarks/coverage_by_realization.py"
FAILED=0
score() {
    local o="$1"; shift
    [ -s "$o" ] && { echo "skip (done) $o"; return; }
    mkdir -p "$(dirname "$o")"
    echo "[$(date '+%F %T')] $o"
    python -u "$CBR" "$@" --tau-smoother none --dump-per-real "$o" > "${o%.npz}.md" 2> "${o%.npz}.err" \
        || { echo "  ERROR: ${o%.npz}.err"; FAILED=1; }
}
[ -d "$HIST/truth" ] || { echo "FATAL: no $HIST/truth (built by submit_hist.sbatch task 0)"; exit 1; }
CASES="Observed_Confounder Observed_Mediator Observed_Mediator_and_Confounder Unobserved_Confounder Frontdoor_Criterion Backdoor_Criterion"
case "$SLURM_ARRAY_TASK_ID" in
  0) for ds in IHDP ACIC; do
       score "$HIST/perreal_e1/$NAME/raw__${ds}__-.npz" --root "$RC_OUT" --dataset "$ds" --methods dopfn_native --truth-dir "$HIST/truth"
     done
     for ds in CPS PSID PSID_bal; do
       score "$HIST/perreal/$NAME/raw__${ds}__-.npz" --root "$RC_OUT" --dataset "$ds" --methods dopfn_native
     done ;;
  1|2|3) SH=(shift-2 shift0 shift+2); sh=${SH[$(( SLURM_ARRAY_TASK_ID - 1 ))]}
     for cell in "$CS_OUT/$sh"/d*/ctx1000; do
       [ -d "$cell" ] || continue
       d="${cell%/ctx1000}"; d="${d##*/}"
       for c in $CASES; do
         score "$HIST/perreal/$NAME/raw__${d}_${c}__${sh}.npz" --root "$cell" --dataset "$c" --methods dopfn_native
       done
     done ;;
  4) [ -d "$CM_OUT/N1000" ] || { echo "MISSING $CM_OUT/N1000"; exit 1; }
     for n in 5 10 20 30 40 50; do
       score "$HIST/perreal/$NAME/raw__cmech_n${n}__-.npz" --root "$CM_OUT" --nodes "$n" --subset total --context 1000
     done ;;
  5) for ds in IHDP ACIC; do
       score "$HIST/perreal_e1/$NAME/indep__${ds}__-.npz" --root "$RC_OUT" --dataset "$ds" --methods dopfn_native \
             --truth-dir "$HIST/truth" --joint-coupling indep
     done
     for ds in CPS PSID PSID_bal; do
       score "$HIST/perreal/$NAME/indep__${ds}__-.npz" --root "$RC_OUT" --dataset "$ds" --methods dopfn_native --joint-coupling indep
     done ;;
esac
echo "[$(date '+%F %T')] done (failed=$FAILED)"
exit $FAILED
