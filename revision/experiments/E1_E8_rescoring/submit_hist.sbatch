#!/bin/bash
#SBATCH --job-name=hist-rescore
#SBATCH --output=logs_hist/hist_%a.out
#SBATCH --error=logs_hist/hist_%a.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --array=0-6
# Every benchmark rescored with the hist interval (CDM_INTERVAL=hist: the exact tau distribution
# of the predicted histogram), 1D and 2D models alike, no MALC. Reads the existing dumps only;
# no model is run. Writes ONLY under $OUT, never into the dump folders.
#   0  RealCause: IHDP, ACIC against realized delta (E1; truth rebuilt first); CPS, PSID, PSID_bal
#   1  case study, shift -2     2  shift 0     3  shift +2
#   4  ComplexMech, all rho     ($SCRATCH/cmech_dumps/<model>)
#   6  RealCause, 2D models with the coupling forced independent (--joint-coupling indep):
#      the same joint's two marginals combined as if independent; files indep__*
#   5  ComplexMech, rho > 0.99  ($SCRATCH/cmech_dumps_rho99/<model>, data $SCRATCH/cmech_data_rho99;
#      the paper's rho > 0.99 dumps, see benchmarks/cluster/submit_full_report.sbatch)
# Output: $OUT/perreal/<label>/raw__*.npz (IHDP/ACIC under $OUT/perreal_e1), the layout e8_tables.py reads.
#   cd $KIT && sbatch R-PFN/revision/experiments/E1_E8_rescoring/submit_hist.sbatch
set -uo pipefail
mkdir -p logs_hist
DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
export CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
OUT="${OUT:-$SCRATCH/hist}"
source "$DEPLOY_ROOT/venv/bin/activate"
export CDM_INTERVAL=hist PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ATOM_GRID_MAX="${ATOM_GRID_MAX:-4096}"
export PYTHONPATH="$REPO/benchmarks:$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
CBR="$REPO/benchmarks/coverage_by_realization.py"
FAILED=0
score() {   # score <out npz> <cbr args...>   (skips finished files; logs next to the output)
    local o="$1"; shift
    [ -s "$o" ] && { echo "skip (done) $o"; return; }
    mkdir -p "$(dirname "$o")"
    echo "[$(date '+%F %T')] $o"
    python -u "$CBR" "$@" --tau-smoother none --dump-per-real "$o" > "${o%.npz}.md" 2> "${o%.npz}.err" \
        || { echo "  ERROR: ${o%.npz}.err"; FAILED=1; }
}

# label | RealCause root | case-study root | published methods in it
# (Do-PFN 2D's dumps sit under the harness name dopfn_native; see benchmarks/cluster/inventory_models.sh)
ROOTS=(
  "orig|$SCRATCH/rc_dens_uni|$SCRATCH/cs_dvar_dens|dopfn_native uwyk1d-noanc uwyk1d-v3a graph2d-noanc graph2d-v3a cpfn1d"
  "joint2d|$SCRATCH/dumps_all/dopfn_repro_joint2d/rc|$SCRATCH/dumps_all/dopfn_repro_joint2d/cs|dopfn_native"
  "eta0|$SCRATCH/rc_dens_eta0|$SCRATCH/cs_dvar_eta0|cpfn2d"
)
CASES="Observed_Confounder Observed_Mediator Observed_Mediator_and_Confounder Unobserved_Confounder Frontdoor_Criterion Backdoor_Criterion"
CMECH_MODELS="dopfn_native dopfn_repro_joint2d uwyk1d graph2d cpfn1d_j1024 cpfn2d_eta0"

case "$SLURM_ARRAY_TASK_ID" in
  0) python -u "$REPO/revision/experiments/E1_E8_rescoring/build_delta_truth.py" \
         --causalpfn "$CAUSALPFN" --out "$OUT/truth" || { echo "FATAL: truth build failed"; exit 1; }
     for row in "${ROOTS[@]}"; do
       IFS='|' read -r lbl rc cs meths <<<"$row"
       for ds in IHDP ACIC; do
         score "$OUT/perreal_e1/$lbl/raw__${ds}__-.npz" --root "$rc" --dataset "$ds" --methods $meths --truth-dir "$OUT/truth"
       done
       for ds in CPS PSID PSID_bal; do
         score "$OUT/perreal/$lbl/raw__${ds}__-.npz" --root "$rc" --dataset "$ds" --methods $meths
       done
     done ;;
  1|2|3) SH=(shift-2 shift0 shift+2); sh=${SH[$(( SLURM_ARRAY_TASK_ID - 1 ))]}
     for row in "${ROOTS[@]}"; do
       IFS='|' read -r lbl rc cs meths <<<"$row"
       for cell in "$cs/$sh"/d*/ctx1000; do
         [ -d "$cell" ] || continue
         d="${cell%/ctx1000}"; d="${d##*/}"
         for c in $CASES; do
           score "$OUT/perreal/$lbl/raw__${d}_${c}__${sh}.npz" --root "$cell" --dataset "$c" --methods $meths
         done
       done
     done ;;
  6) for row in "${ROOTS[@]}"; do
       IFS='|' read -r lbl rc cs meths <<<"$row"
       m2=""; for m in $meths; do case "$m" in graph2d-*|cpfn2d) m2="$m2 $m" ;; esac; done
       [ "$lbl" = joint2d ] && m2="dopfn_native"                  # Do-PFN 2D (harness name)
       [ -n "$m2" ] || continue
       for ds in IHDP ACIC; do
         score "$OUT/perreal_e1/$lbl/indep__${ds}__-.npz" --root "$rc" --dataset "$ds" --methods $m2 \
               --truth-dir "$OUT/truth" --joint-coupling indep
       done
       for ds in CPS PSID PSID_bal; do
         score "$OUT/perreal/$lbl/indep__${ds}__-.npz" --root "$rc" --dataset "$ds" --methods $m2 --joint-coupling indep
       done
     done ;;
  4|5) if [ "$SLURM_ARRAY_TASK_ID" = 4 ]; then base="$SCRATCH/cmech_dumps"; pre=""
       else base="${CMECH_RHO99:-$SCRATCH/cmech_dumps_rho99}"; pre="rho99_"; fi
     for m in $CMECH_MODELS; do
       [ -d "$base/$m/N1000" ] || { echo "MISSING $base/$m/N1000"; FAILED=1; continue; }
       for n in 5 10 20 30 40 50; do
         compgen -G "$base/$m/N1000/*/CMECH_n${n}_*" >/dev/null || continue   # node counts that were run
         score "$OUT/perreal/$pre$m/raw__cmech_n${n}__-.npz" --root "$base/$m" --nodes "$n" --subset total --context 1000
       done
     done ;;
esac
echo "[$(date '+%F %T')] done (failed=$FAILED)"
exit $FAILED
