#!/bin/bash
#SBATCH --job-name=d262-malc
#SBATCH --output=logs_hist/d262_malc_%a.out
#SBATCH --error=logs_hist/d262_malc_%a.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --array=0-13
# Stage 5: MALC (hist input) for the 262k Do-PFN 2D, identical settings to submit_malc.sbatch
# (B=1000, K=1, n_tau=4001). A finished file is skipped, so a task that times out is just resubmitted.
#   0-4 RealCause   5-9 RealCause, coupling forced independent   10-12 case study shift -2/0/+2   13 ComplexMech all rho
#   cd $KIT && sbatch --account=aip-rgrosse R-PFN/revision/experiments/D262k/5_malc.sbatch
set -uo pipefail
mkdir -p logs_hist
KIT="${KIT:-$PWD}"; source "$KIT/R-PFN/revision/experiments/D262k/common.sh"
source "$KIT/venv/bin/activate"
NW="${SLURM_CPUS_PER_TASK:-32}"
export MALC_INPUT=hist CDM_INTERVAL=raw PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
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
RC_DS=(IHDP ACIC CPS PSID PSID_bal); SHIFTS=(shift-2 shift0 shift+2)
CASES="Observed_Confounder Observed_Mediator Observed_Mediator_and_Confounder Unobserved_Confounder Frontdoor_Criterion Backdoor_Criterion"
rc() {   # rc <dataset> <prefix> [extra]
    local ds="$1" pfx="$2"; shift 2
    if [ "$ds" = IHDP ] || [ "$ds" = ACIC ]; then
        score "$HIST/perreal_e1/$NAME/${pfx}__${ds}__-.npz" --root "$RC_OUT" --dataset "$ds" --methods dopfn_native --truth-dir "$HIST/truth" "$@"
    else
        score "$HIST/perreal/$NAME/${pfx}__${ds}__-.npz" --root "$RC_OUT" --dataset "$ds" --methods dopfn_native "$@"
    fi
}
i=$SLURM_ARRAY_TASK_ID
if   [ $i -lt 5 ];  then rc "${RC_DS[$i]}" malc
elif [ $i -lt 10 ]; then rc "${RC_DS[$(( i - 5 ))]}" indep_malc --joint-coupling indep
elif [ $i -lt 13 ]; then sh=${SHIFTS[$(( i - 10 ))]}
     for d in 5 10 20 30 40 50; do
       cell="$CS_OUT/$sh/d$d/ctx1000"; [ -d "$cell" ] || continue
       for c in $CASES; do score "$HIST/perreal/$NAME/malc__d${d}_${c}__${sh}.npz" --root "$cell" --dataset "$c" --methods dopfn_native; done
     done
else for n in 5 10 20 30 40 50; do
       score "$HIST/perreal/$NAME/malc__cmech_n${n}__-.npz" --root "$CM_OUT" --nodes "$n" --subset total --context 1000
     done
fi
echo "[$(date '+%F %T')] done (failed=$FAILED)"
exit $FAILED
