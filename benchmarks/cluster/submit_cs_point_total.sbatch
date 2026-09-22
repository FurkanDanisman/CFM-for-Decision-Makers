#!/bin/bash
#SBATCH --job-name=cs-point
#SBATCH --output=logs_cs_point/cs_%j.out
#SBATCH --error=logs_cs_point/cs_%j.err
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#
# Case-study point estimates: PEHE, L1_ATE and RELATIVE eps_ATE, one table per case.
# Rescores existing dumps -- no inference, no GPU.
#
# nibi needs an explicit account:
#   sbatch --account=def-zhijing R-PFN/benchmarks/cluster/submit_cs_point_total.sbatch
#
# Optional: OUT, CS_D (space-separated d values), CASES

set -uo pipefail
mkdir -p logs_cs_point
DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
SCRATCH="${SCRATCH:-/scratch/$USER}"
OUT="${OUT:-$SCRATCH/cs_point_total.md}"

source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$REPO/benchmarks:$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%F %T')] out=$OUT"
EXTRA=()
[ -n "${CS_D:-}" ]   && { read -r -a _d <<<"$CS_D";   EXTRA+=(--cs-d "${_d[@]}"); }
[ -n "${CASES:-}" ]  && { read -r -a _c <<<"$CASES";  EXTRA+=(--cases "${_c[@]}"); }

python -u "$REPO/benchmarks/cs_point_total.py" --out "$OUT" "${EXTRA[@]}"
rc=$?
echo "[$(date '+%F %T')] done rc=$rc"
exit $rc
