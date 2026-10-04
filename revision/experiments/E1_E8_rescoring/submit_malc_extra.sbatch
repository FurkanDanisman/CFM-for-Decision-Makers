#!/bin/bash
#SBATCH --job-name=malc-extra
#SBATCH --output=logs_hist/malc_extra_%a.out
#SBATCH --error=logs_hist/malc_extra_%a.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --array=0-29
# MALC (hist input, as submit_malc.sbatch) for the three matched-resolution 1D models of v7
# (Do-PFN J=10, UWYK J=32, CausalPFN-C J=32), on every benchmark. Reads existing dumps only;
# writes malc__ files under $OUT/perreal{,_e1}/<model>/ and rho99_<model>/.
#    0-14   RealCause: model k (0..2) x dataset (IHDP ACIC CPS PSID PSID_bal) = 5k + dataset
#   15-23   case study: model k x shift (-2, 0, +2) = 15 + 3k + shift
#   24-26   ComplexMech all rho, model k
#   27-29   ComplexMech rho > 0.99, model k
# A finished file is skipped, so a task that hits the time limit is just resubmitted.
#   cd $KIT && sbatch R-PFN/revision/experiments/E1_E8_rescoring/submit_malc_extra.sbatch
set -uo pipefail
mkdir -p logs_hist
DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
OUT="${OUT:-$SCRATCH/hist}"
NW="${SLURM_CPUS_PER_TASK:-32}"
source "$DEPLOY_ROOT/venv/bin/activate"
export MALC_INPUT=hist CDM_INTERVAL=raw
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ATOM_GRID_MAX="${ATOM_GRID_MAX:-4096}"
export PYTHONPATH="$REPO/benchmarks:$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
CBR="$REPO/benchmarks/coverage_by_realization.py"
MALC=(--tau-smoother malc --malc-B 1000 --malc-K 1 --n-tau 4001 --malc-workers "$NW")
FAILED=0
score() {
    local o="$1"; shift
    [ -s "$o" ] && { echo "skip (done) $o"; return; }
    mkdir -p "$(dirname "$o")"
    echo "[$(date '+%F %T')] $o"
    python -u "$CBR" "$@" "${MALC[@]}" --dump-per-real "$o" > "${o%.npz}.md" 2> "${o%.npz}.err" \
        || { echo "  ERROR: ${o%.npz}.err"; FAILED=1; }
}
S=$SCRATCH; A=$SCRATCH/aurora3_kill
# model | RealCause root | case-study root | ComplexMech all-rho root | rho>0.99 root | method keys
ROWS=(
  "dopfn_repro_1d_J10|$S/dumps_all/dopfn_repro_1d_J10/rc|$S/dumps_all/dopfn_repro_1d_J10/cs|$S/cmech_dumps/dopfn_repro_1d_J10|$S/cmech_dumps_rho99/dopfn_repro_1d_J10|dopfn_native"
  "uwyk_D_j32_nobin|$A/rc/uwyk_D_j32_nobin|$A/cs/uwyk_D_j32_nobin|$S/aurora3_allrho/cm_rho99/uwyk_D_j32_nobin|$A/cm_rho99/uwyk_D_j32_nobin|uwyk1d-noanc uwyk1d-v3a"
  "cpfn1d_j32|$S/dumps_all/cpfn1d_j32/rc|$S/dumps_all/cpfn1d_j32/cs|$S/cmech_dumps/cpfn1d_j32|$S/cmech_dumps_rho99/cpfn1d_j32|cpfn1d"
)
RC_DS=(IHDP ACIC CPS PSID PSID_bal)
SHIFTS=(shift-2 shift0 shift+2)
CASES="Observed_Confounder Observed_Mediator Observed_Mediator_and_Confounder Unobserved_Confounder Frontdoor_Criterion Backdoor_Criterion"
i=$SLURM_ARRAY_TASK_ID

if [ $i -lt 15 ]; then
    IFS='|' read -r m rc cs cma cm99 meths <<<"${ROWS[$(( i / 5 ))]}"; ds=${RC_DS[$(( i % 5 ))]}
    if [ "$ds" = IHDP ] || [ "$ds" = ACIC ]; then
        score "$OUT/perreal_e1/$m/malc__${ds}__-.npz" --root "$rc" --dataset "$ds" --methods $meths --truth-dir "$OUT/truth"
    else
        score "$OUT/perreal/$m/malc__${ds}__-.npz" --root "$rc" --dataset "$ds" --methods $meths
    fi
elif [ $i -lt 24 ]; then
    j=$(( i - 15 )); IFS='|' read -r m rc cs cma cm99 meths <<<"${ROWS[$(( j / 3 ))]}"; sh=${SHIFTS[$(( j % 3 ))]}
    for d in 5 10 20 30 40 50; do
        cell="$cs/$sh/d$d/ctx1000"; [ -d "$cell" ] || { echo "MISSING $cell"; FAILED=1; continue; }
        for c in $CASES; do
            score "$OUT/perreal/$m/malc__d${d}_${c}__${sh}.npz" --root "$cell" --dataset "$c" --methods $meths
        done
    done
else
    j=$(( i - 24 )); IFS='|' read -r m rc cs cma cm99 meths <<<"${ROWS[$(( j % 3 ))]}"
    if [ $j -lt 3 ]; then base="$cma"; pre=""; else base="$cm99"; pre="rho99_"; fi
    [ -d "$base/N1000" ] || { echo "MISSING $base/N1000"; exit 1; }
    for n in 5 10 20 30 40 50; do
        compgen -G "$base/N1000/*/CMECH_n${n}_*" >/dev/null || continue
        score "$OUT/perreal/$pre$m/malc__cmech_n${n}__-.npz" --root "$base" --nodes "$n" --subset total --context 1000 --methods $meths
    done
fi
echo "[$(date '+%F %T')] done (failed=$FAILED)"
exit $FAILED
