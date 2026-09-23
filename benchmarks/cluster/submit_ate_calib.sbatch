#!/bin/bash
#SBATCH --job-name=ate-calib
#SBATCH --output=logs_ate_calib/ac_%j.out
#SBATCH --error=logs_ate_calib/ac_%j.err
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#
# ATE-interval calibration for ComplexMech, then the combined case-study +
# ComplexMech table.
#
# The case studies ALREADY have theirs -- submit_full_table runs --target ate in
# both the raw and malc stages and writes calib_raw_ate_<case>.md /
# calib_T_<tag>_ate_<case>.md per cell. ComplexMech does not: the scorer that was
# run (submit_cmech_score_one.sbatch) has only point/raw/malc CATE stages. So this
# job scores ComplexMech's ATE target and then aggregates both.
#
#   sbatch --account=def-zhijing R-PFN/benchmarks/cluster/submit_ate_calib.sbatch
#
# One job per model (recommended -- the all-models loop timed out at 6h):
#   for m in $SCRATCH/cmech_dumps/*/; do
#     ONLY=$(basename $m) SKIP_AGG=1 sbatch --account=def-zhijing \
#       R-PFN/benchmarks/cluster/submit_ate_calib.sbatch
#   done
#   # then once, to build the table:
#   SKIP_SCORE=1 sbatch --account=def-zhijing R-PFN/.../submit_ate_calib.sbatch
#
# Optional: CMECH_DUMPS, CMECH_DATA, CM_NODES, CTX, MALC_B, MALC_K, N_TAU,
#           ONLY=<model>, SKIP_MALC=1, SKIP_SCORE=1 (aggregate only),
#           SKIP_AGG=1 (score only), OUT

set -uo pipefail
mkdir -p logs_ate_calib
DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
SCRATCH="${SCRATCH:-/scratch/$USER}"
DUMPS="${CMECH_DUMPS:-$SCRATCH/cmech_dumps}"
DATA="${CMECH_DATA:-$SCRATCH/cmech_data_v2}"
NODES="${CM_NODES:-5 10 20 30 40 50}"
CTX="${CTX:-1000}"
MALC_B="${MALC_B:-1000}"; MALC_K="${MALC_K:-1}"; N_TAU="${N_TAU:-4001}"
NW="${MALC_WORKERS:-${SLURM_CPUS_PER_TASK:-32}}"
TAG="B${MALC_B}_K${MALC_K}"
OUT="${OUT:-$SCRATCH/ate_calib.md}"

source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ATOM_GRID_MAX="${ATOM_GRID_MAX:-4096}"
export PYTHONPATH="$REPO/benchmarks:$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
CD="$REPO/UWYK_Fig3_4/cate_density_metrics.py"

FAILED=0
if [ "${SKIP_SCORE:-0}" != 1 ]; then
  for md in "$DUMPS"/*; do
    [ -d "$md" ] || continue
    # ONLY=<model> restricts this job to one model so the 13 can be fanned out as
    # 13 jobs. The single-job loop timed out at 6h having finished ~2/3.
    [ -n "${ONLY:-}" ] && [ "$(basename "$md")" != "$ONLY" ] && continue
    for n in $NODES; do
      o_raw="$md/N$CTX/calib_cmech_n${n}_raw_ate.md"
      if [ -s "$o_raw" ]; then echo "skip (done): $(basename "$md") n=$n raw"
      else
        echo "==== [$(date '+%F %T')] $(basename "$md") n=$n raw"
        python -u "$CD" --root "$md" --context "$CTX" --nodes "$n" \
            --subset total --target ate --data-root "$DATA" \
            --tau-smoother none --out "$o_raw" >"$o_raw.log" 2>&1 \
            || { echo "    FAILED (see $o_raw.log)"; FAILED=$((FAILED+1)); }
      fi
      [ "${SKIP_MALC:-0}" = 1 ] && continue
      o_mal="$md/N$CTX/calib_cmech_n${n}_T_${TAG}_ate.md"
      if [ -s "$o_mal" ]; then echo "skip (done): $(basename "$md") n=$n MALC"
      else
        echo "==== [$(date '+%F %T')] $(basename "$md") n=$n MALC"
        python -u "$CD" --root "$md" --context "$CTX" --nodes "$n" \
            --subset total --target ate --data-root "$DATA" \
            --tau-smoother malc --malc-B "$MALC_B" --malc-K "$MALC_K" \
            --n-tau "$N_TAU" --malc-workers "$NW" --out "$o_mal" \
            >"$o_mal.log" 2>&1 \
            || { echo "    FAILED (see $o_mal.log)"; FAILED=$((FAILED+1)); }
      fi
    done
  done
fi

if [ "${SKIP_AGG:-0}" = 1 ]; then
    echo "[$(date '+%F %T')] scoring done (SKIP_AGG=1)  failures=$FAILED"
    exit $FAILED
fi

echo
echo "==== [$(date '+%F %T')] aggregating RealCause + case study + ComplexMech"
python -u "$REPO/benchmarks/ate_calib_tables.py" \
    --scratch "$SCRATCH" --cmech-dumps "$DUMPS" --nodes $NODES \
    --context "$CTX" --out "$OUT" || FAILED=$((FAILED+1))

echo "[$(date '+%F %T')] done  failures=$FAILED"
echo "  $OUT"
exit $FAILED
