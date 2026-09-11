#!/bin/bash
#SBATCH --job-name=dvar-combine
#SBATCH --output=logs_case_study/combine_%j.out
#SBATCH --error=logs_case_study/combine_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# Pool realizations across shifts into one CSV on a COMPUTE node (CPU only — this
# is file I/O over ~10^5-10^6 npz, no GPU involved). Submit from $DEPLOY_ROOT.
#
# Env:
#   RES              results root (holds shift<S>/d<K>/ctx<N>/...)  (required)
#   COMBINE_SHIFTS   default "shift0 shift+2 shift-2 shift+5 shift-5"
#   COMBINE_LABEL    default "all5"
#   D_VALUES         default "5 10 20 30 40 50"  (skips others -> fewer reads)
#   OUT              default $RES/combined_<LABEL>.csv
set -euo pipefail
mkdir -p logs_case_study

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
RES="${RES:?RES required (results root)}"
SHIFTS="${COMBINE_SHIFTS:-shift0 shift+2 shift-2 shift+5 shift-5}"
LABEL="${COMBINE_LABEL:-all5}"
D_VALUES="${D_VALUES:-5 10 20 30 40 50}"
OUT="${OUT:-$RES/combined_${LABEL}.csv}"

source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1
echo "[$(date '+%F %T')] combine [$SHIFTS] -> $LABEL  d=[$D_VALUES]  RES=$RES"

python -u "$REPO/case_study/cluster/dsweep_report.py" --root "$RES" \
    --combine-shifts $SHIFTS --combine-label "$LABEL" \
    --d-values $D_VALUES --out "$OUT"

echo "[$(date '+%F %T')] done -> $OUT"
