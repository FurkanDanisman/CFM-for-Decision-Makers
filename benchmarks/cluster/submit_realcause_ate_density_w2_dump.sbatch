#!/bin/bash
#SBATCH --job-name=rc-ate-w2
#SBATCH --output=logs_rc_ate_w2/dump_%A_%a.out
#SBATCH --error=logs_rc_ate_w2/dump_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-14

# CPU-only 15-task array — 3 methods × 5 RealCause datasets. Per task:
# for each realization in the cell, run 1D W2 barycenter over per-query
# p(τ) densities to obtain a single p_ATE(τ) per realization, then save
# ate_w2_{TAG}_r<###>.npz next to the density NPZ.
#
#   METHOD_SET     env:  '1d' (cpfn1d/dopfn/uwyk1d) or '2d' (cpfn2d/graph2d/dopfnbb).
#   SOURCE         env:  'malc' | 'joint' | 'marginals'.
#                        For METHOD_SET=1d,  SOURCE must be 'marginals'.
#                        For METHOD_SET=2d,  SOURCE ∈ {'joint', 'malc'}.
#   MALC_TAG       env:  required when SOURCE=malc  (e.g. B100, B500, B1000).
#   OUT_TAG        env:  overrides the default output filename tag.
#
#   method_idx = task / 5   → per METHOD_SET (see METHODS array below)
#   ds_idx     = task % 5   → 0=IHDP 1=ACIC 2=CPS 3=PSID 4=PSID_bal
#
# After the run:
#   sbatch benchmarks/cluster/submit_realcause_ate_density_w2_summary.sbatch

set -euo pipefail
mkdir -p logs_rc_ate_w2

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT required — where density (and MALC) NPZs live}"
METHOD_SET="${METHOD_SET:?METHOD_SET required (1d or 2d)}"
SOURCE="${SOURCE:?SOURCE required (malc / joint / marginals)}"
MALC_TAG="${MALC_TAG:-}"
OUT_TAG="${OUT_TAG:-}"
OVERWRITE="${OVERWRITE:-0}"

case "$METHOD_SET" in
    1d) METHODS=(cpfn1d dopfn uwyk1d) ;;
    2d) METHODS=(cpfn2d graph2d dopfnbb) ;;
    *)  echo "FATAL: METHOD_SET must be 1d or 2d, got $METHOD_SET" >&2; exit 2 ;;
esac
case "$SOURCE" in
    malc)
        [ "$METHOD_SET" = "2d" ] || { echo "FATAL: SOURCE=malc requires METHOD_SET=2d" >&2; exit 2; }
        [ -n "$MALC_TAG" ]      || { echo "FATAL: SOURCE=malc requires MALC_TAG"       >&2; exit 2; }
        ;;
    joint)
        [ "$METHOD_SET" = "2d" ] || { echo "FATAL: SOURCE=joint requires METHOD_SET=2d" >&2; exit 2; }
        ;;
    marginals)
        [ "$METHOD_SET" = "1d" ] || { echo "FATAL: SOURCE=marginals requires METHOD_SET=1d" >&2; exit 2; }
        ;;
    *) echo "FATAL: SOURCE must be malc / joint / marginals" >&2; exit 2 ;;
esac

METHOD_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
DS_IDX=$(( SLURM_ARRAY_TASK_ID % 5 ))
METHOD="${METHODS[$METHOD_IDX]}"
case "$DS_IDX" in
    0) DATASET=IHDP     ;;
    1) DATASET=ACIC     ;;
    2) DATASET=CPS      ;;
    3) DATASET=PSID     ;;
    4) DATASET=PSID_bal ;;
    *) echo "invalid DS_IDX=$DS_IDX" >&2; exit 2 ;;
esac

# dopfnbb legacy dir naming (PSIDbal instead of PSID_bal).
DS_ON_DISK="$DATASET"; SPLIT_FLAG=""
if [ "$METHOD" = "dopfnbb" ]; then
    SPLIT_FLAG="--is-split"
    [ "$DATASET" = "PSID_bal" ] && DS_ON_DISK="PSIDbal"
fi

METHOD_DIR="$OUT_ROOT/$METHOD"
CELL_DIR="$METHOD_DIR/$DS_ON_DISK"
[ -d "$CELL_DIR" ] || { echo "FATAL: cell dir not found: $CELL_DIR" >&2; exit 1; }

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export PYTHONUNBUFFERED=1

_EXTRA=()
[ "$OVERWRITE" = "1" ] && _EXTRA+=(--overwrite)
[ -n "$OUT_TAG" ]      && _EXTRA+=(--out-tag "$OUT_TAG")
[ -n "$MALC_TAG" ]     && _EXTRA+=(--malc-tag "$MALC_TAG")

echo "[$(date '+%Y-%m-%d %H:%M:%S')] $METHOD  $DATASET  SOURCE=$SOURCE"\
"${MALC_TAG:+  MALC_TAG=$MALC_TAG}"

python -u "$REPO/realcause_eval/compute_ate_density_w2_cell.py" \
    --method-dir "$METHOD_DIR" \
    --dataset    "$DS_ON_DISK" \
    --source     "$SOURCE" \
    $SPLIT_FLAG \
    --repo       "$REPO" \
    "${_EXTRA[@]}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] done  $METHOD  $DATASET"
