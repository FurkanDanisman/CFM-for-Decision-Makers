#!/bin/bash
#SBATCH --job-name=a3-point
#SBATCH --output=logs_a3/point_%x_%j.out
#SBATCH --error=logs_a3/point_%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#
# PEHE / eps_ATE for one model across all three benchmarks. Reads the existing
# density dumps -- no model is run, no GPU is needed.
#
#   MODEL=uwyk_C_botharms METHODS="uwyk1d-noanc uwyk1d-v3a" \
#   PARENT=$SCRATCH/aurora3_kill OUT=$SCRATCH/a3_tables \
#     sbatch --account=aip-rgrosse R-PFN/benchmarks/cluster/submit_a3_point.sbatch
#
# Env: MODEL, METHODS, PARENT, OUT, REPO, ATE_METRIC, BENCHES, CTX
set -uo pipefail
REPO="${REPO:-$PWD/R-PFN}"
MODEL="${MODEL:?MODEL required}"
METHODS="${METHODS:?METHODS required}"
PARENT="${PARENT:?PARENT required}"
OUT="${OUT:-$SCRATCH/a3_tables}"
ATE_METRIC="${ATE_METRIC:-rel}"
BENCHES="${BENCHES:-rc cm cs}"
CTX="${CTX:-1000}"
mkdir -p "$OUT" logs_a3

[ -d "${VENV:-$PWD/venv}" ] && source "${VENV:-$PWD/venv}/bin/activate"

echo "[$(date '+%F %T')] model=$MODEL methods=$METHODS metric=$ATE_METRIC"

for B in $BENCHES; do
  case "$B" in
    rc)
      # Per dataset, never pooled: the five differ by orders of magnitude in
      # outcome scale, so one pooled row hides which dataset a number came from.
      for D in IHDP ACIC CPS PSID PSID_bal; do
        echo "---- rc $D ----"
        python -u "$REPO/realcause_eval/point_raw_em.py" \
          --root "$PARENT/rc/$MODEL" --dataset "$D" \
          --methods $METHODS --modes raw --ate-metric "$ATE_METRIC" \
          --out-md "$OUT/point_rc_${MODEL}_${D}.md" || echo "[warn] rc $D failed"
      done ;;
    cm)
      # N=1000 only, one table per d, each pooling its nonzero+zero cells --
      # those split the QUERIES of the same realizations, so concatenating per
      # realization and scoring once is the exact 'total'.
      for N in 5 10 20 30 40 50; do
        echo "---- cm n$N ----"
        python -u "$REPO/realcause_eval/point_raw_em.py" \
          --root "$PARENT/cm_rho99/$MODEL/N$CTX" \
          --dataset "CMECH_n${N}_nonzero" "CMECH_n${N}_zero" \
          --methods $METHODS --modes raw --ate-metric "$ATE_METRIC" \
          --out-md "$OUT/point_cm_${MODEL}_n${N}.md" || echo "[warn] cm n$N failed"
      done ;;
    cs)
      echo "---- cs (shifts and d pooled) ----"
      python -u "$REPO/benchmarks/a3_case_study_point.py" \
        --root "$PARENT/cs/$MODEL" --methods $METHODS \
        --ate-metric "$ATE_METRIC" --ctx "$CTX" \
        --out-md "$OUT/point_cs_${MODEL}.md" || echo "[warn] cs failed" ;;
  esac
done
echo "[$(date '+%F %T')] done -> $OUT"
