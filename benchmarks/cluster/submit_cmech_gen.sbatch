#!/bin/bash
#SBATCH --job-name=cmech-gen
#SBATCH --output=logs_cmech_gen/gen_%j.out
#SBATCH --error=logs_cmech_gen/gen_%j.err
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#
# Regenerate the ComplexMech benchmark with the validity band. CPU only -- no
# --gres -- but killarney has no CPU partition, so it lands on a GPU one and simply
# does not use the GPU. 3h keeps it in the short tier.
#
# Measured locally: ~30-80s per cell depending on node count, 30 cells, so well
# inside the limit even with the ~45-60% resampling overhead the band costs.
#
#   sbatch benchmarks/cluster/submit_cmech_gen.sbatch
#
# Env: OUT (default $SCRATCH/cmech_data_v2), NODES, HIDES, NREAL, NTEST

set -uo pipefail
mkdir -p logs_cmech_gen
DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
export UWYK_SRC="${UWYK_SRC:-$DEPLOY_ROOT/external/uwyk_reproduce/src}"
export UWYK_ROOT="${UWYK_ROOT:-$DEPLOY_ROOT/external/uwyk_reproduce}"
OUT="${OUT:-$SCRATCH/cmech_data_v2}"
NODES="${NODES:-5 10 20 30 40 50}"
NREAL="${NREAL:-100}"; NTEST="${NTEST:-100}"
MIN_HET="${MIN_HET:-0.05}"; MAX_HET="${MAX_HET:-2.0}"; MAX_PIN="${MAX_PIN:-0.5}"

source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1
[ -d "$UWYK_ROOT/experiments" ] || {
    echo "FATAL: no UWYK configs at $UWYK_ROOT/experiments" >&2; exit 1; }

echo "[$(date '+%F %T')] out=$OUT nodes=$NODES n_real=$NREAL n_test=$NTEST"
echo "  band: sd(tau)/sd(y0) in [$MIN_HET, $MAX_HET], pinned <= $MAX_PIN"
python -u "$REPO/UWYK_Fig3_4/generate_pehe_benchmark.py" \
    --prior complexmech --nodes $NODES --regimes path_TY \
    --n-realizations "$NREAL" --n-test "$NTEST" \
    --min-tau-het "$MIN_HET" --max-tau-het "$MAX_HET" --max-pinned "$MAX_PIN" \
    --out-dir "$OUT"
st=$?
echo "[$(date '+%F %T')] generator exit=$st"
[ "$st" = 0 ] || exit "$st"

# Verify rather than assume: every cell must have no constant-tau realization.
echo
echo "==== validity check, all cells"
BAD=0
for N in $NODES; do
    for H in 0.0 0.25 0.5 0.75 1.0; do
        d="$OUT/complexmech/${N}node/path_TY/hide_$H"
        [ -d "$d" ] || continue
        out=$(python -u "$REPO/benchmarks/cmech_coverage_check.py" \
                --cells "$d" --labels "n${N}_h${H}" 2>&1)
        echo "$out" | grep -E "^### |exactly-constant|adaptive oracle"
        echo "$out" | grep -q "exactly-constant tau: 0/" || {
            echo "  FAIL: n=$N hide=$H still has constant-tau realizations" >&2
            BAD=1; }
    done
done
echo
echo "[$(date '+%F %T')] validity FAILURES=$BAD"
exit "$BAD"
