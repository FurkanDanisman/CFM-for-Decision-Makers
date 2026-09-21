#!/bin/bash
#SBATCH --job-name=cm-score
#SBATCH --output=logs_cmech_score/cs_%j.out
#SBATCH --error=logs_cmech_score/cs_%j.err
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#
# Score ONE ComplexMech model: point estimates, raw coverage, MALC coverage.
#
# Matches how the other benchmarks are scored, which the existing
# submit_cmech_density_score.sbatch does NOT: that one calls the pooled scorer and
# writes only markdown, so nothing it produces can reach final_table or progress.sh.
# Here CATE coverage goes through coverage_by_realization with --dump-per-real, so
# ComplexMech lands in the same perreal tree as everything else and pools by the same
# rule.
#
# ComplexMech's reported figure is `total` = nonzero + zero, concatenated per
# realization -- point_raw_em takes both cell names and joins them, and the coverage
# scorer takes --subset total.
#
# ATE error is L1 here, not relative: that convention is RealCause-only.
#
# Required: MODEL_NAME, ROOT (the model's dump root), PERREAL
# Optional: CM_NODES, CTX, STAGES, MALC_B, MALC_K, N_TAU

set -uo pipefail
mkdir -p logs_cmech_score
: "${MODEL_NAME:?}"; : "${ROOT:?}"; : "${PERREAL:?}"

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
CTX="${CTX:-1000}"
read -r -a NODES <<<"${CM_NODES:-5 10 20 30 40 50}"
STAGES="${STAGES:-point raw malc}"
MALC_B="${MALC_B:-1000}"; MALC_K="${MALC_K:-1}"; N_TAU="${N_TAU:-4001}"
NW="${MALC_WORKERS:-${SLURM_CPUS_PER_TASK:-64}}"
TAG="B${MALC_B}_K${MALC_K}"

source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ATOM_GRID_MAX="${ATOM_GRID_MAX:-4096}"
export PYTHONPATH="$REPO/benchmarks:$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
PT="$REPO/realcause_eval/point_raw_em.py"
CBR="$REPO/benchmarks/coverage_by_realization.py"
mkdir -p "$PERREAL/$MODEL_NAME"

# Same resume rule as the other benchmarks: a cell whose output already exists is
# skipped, so a walltime kill costs only the cell in flight.
done_already() { [ "${FORCE_RESCORE:-0}" = 1 ] && return 1; [ -s "$1" ]; }

echo "[$(date '+%F %T')] $MODEL_NAME  root=$ROOT  nodes=${NODES[*]}  N=$CTX"
FAILED=0
for n in "${NODES[@]}"; do
    for S in $STAGES; do
        case "$S" in
        point)
            out="$ROOT/N$CTX/point_raw_em_CMECH_n${n}.md"
            if done_already "$out"; then echo "  skip (done): point n=$n"; continue; fi
            echo "  point n=$n"
            # total = nonzero + zero, concatenated per realization. L1 ATE.
            python -u "$PT" --root "$ROOT/N$CTX" \
                --dataset "CMECH_n${n}_nonzero" "CMECH_n${n}_zero" \
                --modes raw --ate-metric l1 --out-md "$out" \
                > "$out.log" 2>&1 || { echo "    ERROR (see $out.log)"; FAILED=$((FAILED+1)); }
            ;;
        raw)
            npz="$PERREAL/$MODEL_NAME/raw__cmech_n${n}__-.npz"
            if done_already "$npz"; then echo "  skip (done): raw n=$n"; continue; fi
            echo "  raw n=$n"
            python -u "$CBR" --root "$ROOT" --nodes "$n" --subset total \
                --context "$CTX" --tau-smoother none --dump-per-real "$npz" \
                > "$ROOT/N$CTX/cbr_cmech_n${n}_raw.md" \
                2> "$ROOT/N$CTX/cbr_cmech_n${n}_raw.err" \
                || { echo "    ERROR (see cbr_cmech_n${n}_raw.err)"; FAILED=$((FAILED+1)); }
            ;;
        malc)
            npz="$PERREAL/$MODEL_NAME/malc__cmech_n${n}__-.npz"
            if done_already "$npz"; then echo "  skip (done): MALC n=$n"; continue; fi
            echo "  MALC n=$n (B=$MALC_B K=$MALC_K, $NW workers)"
            python -u "$CBR" --root "$ROOT" --nodes "$n" --subset total \
                --context "$CTX" --tau-smoother malc --malc-B "$MALC_B" \
                --malc-K "$MALC_K" --n-tau "$N_TAU" --malc-workers "$NW" \
                --dump-per-real "$npz" \
                > "$ROOT/N$CTX/cbr_cmech_n${n}_T_${TAG}.md" \
                2> "$ROOT/N$CTX/cbr_cmech_n${n}_T_${TAG}.err" \
                || { echo "    ERROR (see cbr_cmech_n${n}_T_${TAG}.err)"; FAILED=$((FAILED+1)); }
            ;;
        esac
    done
done
echo "[$(date '+%F %T')] $MODEL_NAME done  failures=$FAILED"
exit 0
