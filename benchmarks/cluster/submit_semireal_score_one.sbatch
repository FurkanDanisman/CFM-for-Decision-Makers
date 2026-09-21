#!/bin/bash
#SBATCH --job-name=sr-score
#SBATCH --output=logs_semireal_score/srs_%j.out
#SBATCH --error=logs_semireal_score/srs_%j.err
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#
# Score ONE model on both semi-real datasets: point estimates, raw coverage, MALC
# coverage. One job per model is enough -- 5 splits per dataset is tiny next to the
# 100 realizations RealCause and the case studies carry.
#
# Per-realization files use the slice suffix "semireal", NOT "-". RealCause files are
# raw__<DS>__-.npz, and ComplexMech reusing "-" made progress.sh count cmech cells as
# RealCause datasets and report 6/5. A distinct suffix keeps the three benchmarks
# countable apart.
#
# Required: MODEL_NAME, ROOT, PERREAL
# Optional: SR_DATASETS, STAGES, MALC_B, MALC_K, N_TAU, ATE_METRIC

set -uo pipefail
mkdir -p logs_semireal_score
: "${MODEL_NAME:?}"; : "${ROOT:?}"; : "${PERREAL:?}"

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
export DOPFN_ROOT="${DOPFN_ROOT:-$DEPLOY_ROOT/external/dopfn}"
read -r -a DSETS <<<"${SR_DATASETS:-SEMIREAL_sales SEMIREAL_law_race}"
STAGES="${STAGES:-point raw malc}"
MALC_B="${MALC_B:-1000}"; MALC_K="${MALC_K:-1}"; N_TAU="${N_TAU:-4001}"
NW="${MALC_WORKERS:-${SLURM_CPUS_PER_TASK:-32}}"
# Semi-real sits alongside RealCause in kind, so the relative eps_ATE applies --
# that is also what Do-PFN reports for these two.
ATE_METRIC="${ATE_METRIC:-rel}"
TAG="B${MALC_B}_K${MALC_K}"

source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ATOM_GRID_MAX="${ATOM_GRID_MAX:-4096}"
export PYTHONPATH="$REPO/benchmarks:$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
PT="$REPO/realcause_eval/point_raw_em.py"
CBR="$REPO/benchmarks/coverage_by_realization.py"
mkdir -p "$PERREAL/$MODEL_NAME"

done_already() {
    [ "${FORCE_RESCORE:-0}" = 1 ] && return 1
    [ -s "$1" ] || return 1
    # A markdown table with a header and NO data rows is not done. dopfn_bb's point
    # table came out exactly that way -- the directory it needed resolved to an empty
    # leftover -- and a plain -s test called it finished, so the rescore skipped it.
    case "$1" in
      *.md) grep -qE '^\| *[A-Za-z0-9_-]+ *\| *[0-9]' "$1" || return 1 ;;
    esac
    return 0
}

echo "[$(date '+%F %T')] $MODEL_NAME  root=$ROOT  datasets=${DSETS[*]}"
FAILED=0
for ds in "${DSETS[@]}"; do
    short="${ds#SEMIREAL_}"
    for S in $STAGES; do
        case "$S" in
        point)
            out="$ROOT/point_raw_em_${ds}.md"
            if done_already "$out"; then echo "  skip (done): point $short"; continue; fi
            echo "  point $short"
            python -u "$PT" --root "$ROOT" --dataset "$ds" --modes raw \
                --ate-metric "$ATE_METRIC" --out-md "$out" \
                > "$out.log" 2>&1 || { echo "    ERROR (see $out.log)"; FAILED=$((FAILED+1)); }
            ;;
        raw)
            npz="$PERREAL/$MODEL_NAME/raw__${short}__semireal.npz"
            if done_already "$npz"; then echo "  skip (done): raw $short"; continue; fi
            echo "  raw $short"
            python -u "$CBR" --root "$ROOT" --dataset "$ds" --tau-smoother none \
                --dump-per-real "$npz" \
                > "$ROOT/cbr_${ds}_raw.md" 2> "$ROOT/cbr_${ds}_raw.err" \
                || { echo "    ERROR (see cbr_${ds}_raw.err)"; FAILED=$((FAILED+1)); }
            ;;
        malc)
            npz="$PERREAL/$MODEL_NAME/malc__${short}__semireal.npz"
            if done_already "$npz"; then echo "  skip (done): MALC $short"; continue; fi
            echo "  MALC $short (B=$MALC_B K=$MALC_K, $NW workers)"
            python -u "$CBR" --root "$ROOT" --dataset "$ds" --tau-smoother malc \
                --malc-B "$MALC_B" --malc-K "$MALC_K" --n-tau "$N_TAU" \
                --malc-workers "$NW" --dump-per-real "$npz" \
                > "$ROOT/cbr_${ds}_T_${TAG}.md" 2> "$ROOT/cbr_${ds}_T_${TAG}.err" \
                || { echo "    ERROR (see cbr_${ds}_T_${TAG}.err)"; FAILED=$((FAILED+1)); }
            ;;
        esac
    done
done
echo "[$(date '+%F %T')] $MODEL_NAME done  failures=$FAILED"
exit 0
