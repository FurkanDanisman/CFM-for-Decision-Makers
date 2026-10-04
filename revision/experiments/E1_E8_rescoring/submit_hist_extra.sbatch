#!/bin/bash
#SBATCH --job-name=hist-extra
#SBATCH --output=logs_hist/extra_%a.out
#SBATCH --error=logs_hist/extra_%a.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --array=0-26
# hist scoring (CDM_INTERVAL=hist, no MALC) for the 8 appendix models of v7: matched-label and
# matched-resolution 1D versions, UWYK-B and the published two-stage CausalPFN. Reads existing
# dumps only; writes only under $OUT/perreal{,_e1}/<model>/ (and rho99_<model>).
#    0      RealCause, all 8 (IHDP/ACIC against realized delta; needs $OUT/truth from submit_hist.sbatch)
#    1-24   case study: model k (0..7) x shift s (-2, 0, +2) = task 1 + 3k + s
#           (dopfn_1d_botharms is k = 7: tasks 22-24, run once its case-study dumps are complete)
#    25     ComplexMech all rho, all 8
#    26     ComplexMech rho > 0.99, all 8
#   cd $KIT && sbatch --array=0-21,25,26 R-PFN/revision/experiments/E1_E8_rescoring/submit_hist_extra.sbatch
set -uo pipefail
mkdir -p logs_hist
DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
OUT="${OUT:-$SCRATCH/hist}"
source "$DEPLOY_ROOT/venv/bin/activate"
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
S=$SCRATCH; A=$SCRATCH/aurora3_kill
# model | RealCause root | case-study root | ComplexMech all-rho root | rho>0.99 root | method keys
# (method key = the harness that wrote the dump: benchmarks/cluster/inventory_models.sh)
ROWS=(
  "dopfn_repro_1d_J10|$S/dumps_all/dopfn_repro_1d_J10/rc|$S/dumps_all/dopfn_repro_1d_J10/cs|$S/cmech_dumps/dopfn_repro_1d_J10|$S/cmech_dumps_rho99/dopfn_repro_1d_J10|dopfn_native"
  "uwyk_bin|$S/dumps_all/uwyk_bin/rc|$S/dumps_all/uwyk_bin/cs|$S/cmech_dumps/uwyk_bin|$S/cmech_dumps_rho99/uwyk_bin|uwyk1d-noanc uwyk1d-v3a"
  "cpfn1d_botharms|$S/dumps_all/cpfn1d_botharms/rc|$S/dumps_all/cpfn1d_botharms/cs|$S/cmech_dumps/cpfn1d_botharms|$S/cmech_dumps_rho99/cpfn1d_botharms|cpfn1d"
  "cpfn1d_j32|$S/dumps_all/cpfn1d_j32/rc|$S/dumps_all/cpfn1d_j32/cs|$S/cmech_dumps/cpfn1d_j32|$S/cmech_dumps_rho99/cpfn1d_j32|cpfn1d"
  "cpfn_v0|$S/dumps_all/cpfn_v0/rc|$S/dumps_all/cpfn_v0/cs|$S/cmech_dumps/cpfn_v0|$S/cmech_dumps_rho99/cpfn_v0|cpfn1d"
  "uwyk_C_botharms|$A/rc/uwyk_C_botharms|$A/cs/uwyk_C_botharms|$S/aurora3_allrho/cm_rho99/uwyk_C_botharms|$A/cm_rho99/uwyk_C_botharms|uwyk1d-noanc uwyk1d-v3a"
  "uwyk_D_j32_nobin|$A/rc/uwyk_D_j32_nobin|$A/cs/uwyk_D_j32_nobin|$S/aurora3_allrho/cm_rho99/uwyk_D_j32_nobin|$A/cm_rho99/uwyk_D_j32_nobin|uwyk1d-noanc uwyk1d-v3a"
  "dopfn_1d_botharms|$A/rc/dopfn_1d_botharms|$A/cs/dopfn_1d_botharms|$S/aurora3_allrho/cm_rho99/dopfn_1d_botharms|$A/cm_rho99/dopfn_1d_botharms|dopfn_native"
)
CASES="Observed_Confounder Observed_Mediator Observed_Mediator_and_Confounder Unobserved_Confounder Frontdoor_Criterion Backdoor_Criterion"
SHIFTS=(shift-2 shift0 shift+2)
i=$SLURM_ARRAY_TASK_ID

cm() {   # cm <prefix> <column of ROWS>
    local pre="$1" col="$2"
    for row in "${ROWS[@]}"; do
        IFS='|' read -r m rc cs cma cm99 meths <<<"$row"
        base=$([ "$col" = 4 ] && echo "$cma" || echo "$cm99")
        [ -d "$base/N1000" ] || { echo "MISSING $base/N1000"; FAILED=1; continue; }
        for n in 5 10 20 30 40 50; do
            compgen -G "$base/N1000/*/CMECH_n${n}_*" >/dev/null || continue
            score "$OUT/perreal/$pre$m/raw__cmech_n${n}__-.npz" --root "$base" --nodes "$n" --subset total --context 1000 --methods $meths
        done
    done
}

if [ $i -eq 0 ]; then
    [ -d "$OUT/truth" ] || { echo "FATAL: no $OUT/truth (run submit_hist.sbatch task 0 first)"; exit 1; }
    for row in "${ROWS[@]}"; do
        IFS='|' read -r m rc cs cma cm99 meths <<<"$row"
        for ds in IHDP ACIC; do
            score "$OUT/perreal_e1/$m/raw__${ds}__-.npz" --root "$rc" --dataset "$ds" --methods $meths --truth-dir "$OUT/truth"
        done
        for ds in CPS PSID PSID_bal; do
            score "$OUT/perreal/$m/raw__${ds}__-.npz" --root "$rc" --dataset "$ds" --methods $meths
        done
    done
elif [ $i -le 24 ]; then
    k=$(( (i - 1) / 3 )); sh=${SHIFTS[$(( (i - 1) % 3 ))]}
    IFS='|' read -r m rc cs cma cm99 meths <<<"${ROWS[$k]}"
    for d in 5 10 20 30 40 50; do
        cell="$cs/$sh/d$d/ctx1000"; [ -d "$cell" ] || { echo "MISSING $cell"; FAILED=1; continue; }
        for c in $CASES; do
            score "$OUT/perreal/$m/raw__d${d}_${c}__${sh}.npz" --root "$cell" --dataset "$c" --methods $meths
        done
    done
elif [ $i -eq 25 ]; then cm "" 4
else cm "rho99_" 5
fi
echo "[$(date '+%F %T')] done (failed=$FAILED)"
exit $FAILED
