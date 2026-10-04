#!/bin/bash
#SBATCH --job-name=malc-rescore
#SBATCH --output=logs_hist/malc_%a.out
#SBATCH --error=logs_hist/malc_%a.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --array=0-50%10
# MALC on every benchmark, with the hist distribution as its input (MALC_INPUT=hist: MALC is fitted
# to the exact tau distribution of the predicted histogram, not to atoms jittered within one bin).
# Same settings as the paper (B=1000, K=1, n_tau=4001). Reads the existing dumps; no model is run.
# Writes ONLY under $OUT (malc__* / indep_malc__* files, the layout e8_tables.py reads).
# One task per piece; a finished file is skipped, so a task that times out is just resubmitted.
#    0-14  RealCause      (3 dump roots x IHDP ACIC CPS PSID PSID_bal; IHDP/ACIC against realized delta)
#   15-29  RealCause, 2D models with the coupling forced independent
#   30-38  case study     (3 roots x 3 shifts; d = 5..50)
#   39-44  ComplexMech all rho   (one model each)
#   45-50  ComplexMech rho > 0.99 (one model each; needs $SCRATCH/cmech_dumps_rho99)
# Requires the hist job's truth files ($OUT/truth, task 0 of submit_hist.sbatch).
#   cd $KIT && sbatch R-PFN/revision/experiments/E1_E8_rescoring/submit_malc.sbatch
set -uo pipefail
mkdir -p logs_hist
DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
OUT="${OUT:-$SCRATCH/hist}"
NW="${SLURM_CPUS_PER_TASK:-32}"
source "$DEPLOY_ROOT/venv/bin/activate"
export MALC_INPUT=hist CDM_INTERVAL=raw      # MALC returns a 4001-point grid; read it as is
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ATOM_GRID_MAX="${ATOM_GRID_MAX:-4096}"
export PYTHONPATH="$REPO/benchmarks:$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
CBR="$REPO/benchmarks/coverage_by_realization.py"
MALC=(--tau-smoother malc --malc-B 1000 --malc-K 1 --n-tau 4001 --malc-workers "$NW")
FAILED=0
score() {   # score <out npz> <cbr args...>
    local o="$1"; shift
    [ -s "$o" ] && { echo "skip (done) $o"; return; }
    mkdir -p "$(dirname "$o")"
    echo "[$(date '+%F %T')] $o"
    python -u "$CBR" "$@" "${MALC[@]}" --dump-per-real "$o" > "${o%.npz}.md" 2> "${o%.npz}.err" \
        || { echo "  ERROR: ${o%.npz}.err"; FAILED=1; }
}

# label | RealCause root | case-study root | published methods (Do-PFN 2D: harness name dopfn_native)
ROOTS=(
  "orig|$SCRATCH/rc_dens_uni|$SCRATCH/cs_dvar_dens|dopfn_native uwyk1d-noanc uwyk1d-v3a graph2d-noanc graph2d-v3a cpfn1d"
  "joint2d|$SCRATCH/dumps_all/dopfn_repro_joint2d/rc|$SCRATCH/dumps_all/dopfn_repro_joint2d/cs|dopfn_native"
  "eta0|$SCRATCH/rc_dens_eta0|$SCRATCH/cs_dvar_eta0|cpfn2d"
)
RC_DS=(IHDP ACIC CPS PSID PSID_bal)
SHIFTS=(shift-2 shift0 shift+2)
CASES="Observed_Confounder Observed_Mediator Observed_Mediator_and_Confounder Unobserved_Confounder Frontdoor_Criterion Backdoor_Criterion"
CM=(dopfn_native dopfn_repro_joint2d uwyk1d graph2d cpfn1d_j1024 cpfn2d_eta0)
i=$SLURM_ARRAY_TASK_ID

rc() {   # rc <row> <dataset> <prefix> [extra cbr args]
    IFS='|' read -r lbl rc cs meths <<<"$1"; local ds="$2" pfx="$3"; shift 3
    if [ "$ds" = IHDP ] || [ "$ds" = ACIC ]; then
        score "$OUT/perreal_e1/$lbl/${pfx}__${ds}__-.npz" --root "$rc" --dataset "$ds" --methods $meths --truth-dir "$OUT/truth" "$@"
    else
        score "$OUT/perreal/$lbl/${pfx}__${ds}__-.npz" --root "$rc" --dataset "$ds" --methods $meths "$@"
    fi
}

if   [ $i -lt 15 ]; then rc "${ROOTS[$(( i / 5 ))]}" "${RC_DS[$(( i % 5 ))]}" malc
elif [ $i -lt 30 ]; then j=$(( i - 15 )); row="${ROOTS[$(( j / 5 ))]}"
     IFS='|' read -r lbl rc cs meths <<<"$row"
     m2=""; for m in $meths; do case "$m" in graph2d-*|cpfn2d) m2="$m2 $m" ;; esac; done
     [ "$lbl" = joint2d ] && m2="dopfn_native"
     [ -n "$m2" ] && rc "$lbl|$rc|$cs|$m2" "${RC_DS[$(( j % 5 ))]}" indep_malc --joint-coupling indep
elif [ $i -lt 39 ]; then j=$(( i - 30 )); IFS='|' read -r lbl rc cs meths <<<"${ROOTS[$(( j / 3 ))]}"
     sh=${SHIFTS[$(( j % 3 ))]}
     for d in 5 10 20 30 40 50; do
       cell="$cs/$sh/d$d/ctx1000"; [ -d "$cell" ] || continue
       for c in $CASES; do
         score "$OUT/perreal/$lbl/malc__d${d}_${c}__${sh}.npz" --root "$cell" --dataset "$c" --methods $meths
       done
     done
else j=$(( i - 39 )); m=${CM[$(( j % 6 ))]}
     if [ $j -lt 6 ]; then base="$SCRATCH/cmech_dumps"; pre=""; else base="$SCRATCH/cmech_dumps_rho99"; pre="rho99_"; fi
     [ -d "$base/$m/N1000" ] || { echo "MISSING $base/$m/N1000"; exit 1; }
     for n in 5 10 20 30 40 50; do
       compgen -G "$base/$m/N1000/*/CMECH_n${n}_*" >/dev/null || continue
       score "$OUT/perreal/$pre$m/malc__cmech_n${n}__-.npz" --root "$base/$m" --nodes "$n" --subset total --context 1000
     done
fi
echo "[$(date '+%F %T')] done (failed=$FAILED)"
exit $FAILED
