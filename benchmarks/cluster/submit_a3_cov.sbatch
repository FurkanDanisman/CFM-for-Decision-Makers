#!/bin/bash
#SBATCH --job-name=a3-cov
#SBATCH --output=logs_a3/cov_%x_%j.out
#SBATCH --error=logs_a3/cov_%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#
# Coverage / length / IS / CRPS per realization, for one model, one smoother.
# Reads the existing density dumps; no model is run and no GPU is needed.
#
#   raw, all three benchmarks:
#     MODEL=uwyk_C_botharms METHODS="uwyk1d-noanc uwyk1d-v3a" \
#     PARENT=$SCRATCH/aurora3_kill OUT=$SCRATCH/a3_cov \
#       sbatch --account=aip-rgrosse R-PFN/benchmarks/cluster/submit_a3_cov.sbatch
#
#   MALC, one case at a time (see the cost note below):
#     SMOOTHER=malc BENCHES=cs CASES=Observed_Confounder MODEL=uwyk_D_j32_nobin ...
#
# Env: MODEL, METHODS, PARENT, OUT, BENCHES, SMOOTHER, MALC_B, MALC_K,
#      CASES, NODES, DATASETS, CTX, MAXREAL, REPO, VENV
#
# Aggregation: one number per REALIZATION, then mean +- std across
# realizations. The pooled-over-queries mean in cate_density_metrics treats
# 10,000 clustered Bernoullis as independent and understates the SE about 10x.
#
# MALC COST. ~1.2 s per query at B=1000, divided by --malc-workers. RealCause
# and ComplexMech fit in one 3 h job. The CASE STUDY does not: 18 (shift, d)
# cells x 6 cases is ~18x a RealCause dataset, so run it one case per job
# (CASES=<one>) or it will hit the wall mid-case.
set -uo pipefail
REPO="${REPO:-$PWD/R-PFN}"
MODEL="${MODEL:?MODEL required}"
METHODS="${METHODS:?METHODS required}"
PARENT="${PARENT:?PARENT required}"
OUT="${OUT:-$SCRATCH/a3_cov}"
BENCHES="${BENCHES:-rc cm cs}"
SMOOTHER="${SMOOTHER:-none}"
MALC_B="${MALC_B:-1000}"; MALC_K="${MALC_K:-1}"
CTX="${CTX:-1000}"
DATASETS="${DATASETS:-IHDP ACIC CPS PSID PSID_bal}"
NODES="${NODES:-5 10 20 30 40 50}"
CASES="${CASES:-Observed_Confounder Backdoor_Criterion Observed_Mediator Observed_Mediator_and_Confounder Unobserved_Confounder Frontdoor_Criterion}"
W="${SLURM_CPUS_PER_TASK:-8}"
DEST="$OUT/$SMOOTHER"
mkdir -p "$DEST" logs_a3
[ -d "${VENV:-$PWD/venv}" ] && source "${VENV:-$PWD/venv}/bin/activate"

SM=(--tau-smoother "$SMOOTHER")
if [ "$SMOOTHER" = malc ]; then
    SM+=(--malc-B "$MALC_B" --malc-K "$MALC_K" --malc-workers "$W")
fi
[ -n "${MAXREAL:-}" ] && SM+=(--max-real "$MAXREAL")

echo "[$(date '+%F %T')] model=$MODEL smoother=$SMOOTHER workers=$W benches=$BENCHES"

run() {   # run <outfile stem> <args...>
    local stem="$1"; shift
    local npz="$DEST/${stem}.npz"
    # Resume: a finished selector is skipped, so a wall-kill costs one selector.
    if [ -s "$npz" ]; then echo "  [skip] $npz"; return 0; fi
    echo "---- $stem ----"
    python -u "$REPO/benchmarks/coverage_by_realization.py" \
        --methods $METHODS "${SM[@]}" --dump-per-real "$npz" "$@" \
        || { echo "[warn] $stem FAILED"; rm -f "$npz"; return 1; }
}

for B in $BENCHES; do
  case "$B" in
    rc) for D in $DATASETS; do
          run "rc_${MODEL}_${D}" --root "$PARENT/rc/$MODEL" --dataset "$D"
        done ;;
    cm) for N in $NODES; do
          # NOTE the root has NO /N$CTX: coverage_by_realization builds
          # <root>/N<context>/<subdir>/CMECH_... itself (line ~96), so passing
          # the N1000 level here makes it look under N1000/N1000 and score zero
          # realizations while exiting 0. point_raw_em.py does NOT append it and
          # therefore DOES want the suffix -- the two scripts differ.
          run "cm_${MODEL}_n${N}" --root "$PARENT/cm_rho99/$MODEL" \
              --context "$CTX" --nodes "$N" --subset total
        done ;;
    cs) for C in $CASES; do
          # Glob pools the shift x d cells: concatenating per-realization arrays
          # is exact, whereas averaging per-cell means is not unless every cell
          # has the same realization count.
          run "cs_${MODEL}_${C}" --root "$PARENT"/cs/"$MODEL"/shift*/d*/ctx"$CTX" \
              --dataset "$C"
        done ;;
  esac
done
echo "[$(date '+%F %T')] done -> $DEST"
