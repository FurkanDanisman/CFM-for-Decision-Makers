#!/bin/bash
#SBATCH --job-name=fq4-gen
#SBATCH --output=logs_fq/fq4gen_%A_%a.out
#SBATCH --error=logs_fq/fq4gen_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#
# Generate the fq4 cells for ONE (case, shift): an array over d x realization.
# Task id decodes as  d = DS[id / REALS],  real = id % REALS.
#
# Generation only -- submit_fq4_dump.sbatch does the models, gated on this with
# afterok so no dump can read a half-written cell.
#
# Env: CASE, SHIFT, DS, REALS, QUERIES, DRAWS, QSEED, FQ4, CTX
set -uo pipefail
mkdir -p logs_fq

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
# shellcheck disable=SC1091
source "$DEPLOY_ROOT/venv/bin/activate"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
SC="${SCRATCH:?SCRATCH must be set}"

CASE="${CASE:-Observed_Confounder}"
SHIFT="${SHIFT:-0}"
CTX="${CTX:-1000}"
read -r -a DSA <<<"${DS:-5 10 20 30 40 50}"
REALS="${REALS:-10}"
QUERIES="${QUERIES:-10}"
DRAWS="${DRAWS:-100}"
QSEED="${QSEED:-12345}"
FQ4="${FQ4:-$SC/fq4}"

ID="${SLURM_ARRAY_TASK_ID:-0}"
D="${DSA[$(( ID / REALS ))]}"
REAL=$(( ID % REALS ))
TAG="${CASE}_shift${SHIFT}_d${D}_r${REAL}"
SRC="$DEPLOY_ROOT/case_study_data/d_variation/shift$SHIFT/d$D/$CASE/N$CTX/${CASE}_${REAL}.npz"
GEN_OUT="$FQ4/$TAG"
CELL="$GEN_OUT/$CASE/N$CTX"

echo "[$(date '+%F %T')] task $ID -> $TAG"
[ -f "$SRC" ] || { echo "FATAL: no source npz $SRC" >&2; exit 1; }

HAVE=$(find "$CELL" -name "${CASE}_*.npz" 2>/dev/null | wc -l)
if [ "$HAVE" -ge "$DRAWS" ]; then
    echo "  SKIP ($HAVE >= $DRAWS already)"
else
    # --query-seed varies with the cell so different realizations do not all freeze
    # the same row numbers -- with one seed for all of them the sweep would probe a
    # single fixed position in every realization rather than a sample of queries.
    python "$REPO/case_study/gen_fixed_query.py" \
        --from-npz "$SRC" --d "$D" \
        --random-queries "$QUERIES" --query-seed "$(( QSEED + 1000 * ID ))" \
        --n-draws "$DRAWS" --out "$GEN_OUT" || exit 1
fi

mkdir -p "$FQ4/${TAG}_root/shift0/d0"
ln -sfn "$GEN_OUT/$CASE" "$FQ4/${TAG}_root/shift0/d0/$CASE"
[ -d "$FQ4/${TAG}_root/shift0/d0/$CASE/N$CTX" ] || {
    echo "FATAL: symlink did not resolve" >&2; exit 1; }
echo "[$(date '+%F %T')] $TAG ready"
