#!/bin/bash
# Combined fq4 status: case study and ComplexMech, dumps and scores, one screen.
#
# Counts with ONE find per model rather than one per cell: 45 cells x 13 models is 585
# stat storms on a filesystem already busy with the dumps, and the earlier per-cell
# version took long enough that it looked hung.
#
#   bash R-PFN/benchmarks/cluster/fq_status.sh
# Env: DS, NODES_LIST, REALS, DRAWS, SHIFTS, OUT_DIR, OUT4, CMFQ_DUMPS
set -uo pipefail
KIT="${KIT:-$PWD}"; REPO="${REPO:-$KIT/R-PFN}"
SC="${SCRATCH:?SCRATCH must be set}"
CK="${CK:-$REPO/Required_checkpoints}"; UWYKD="${UWYK_CKPT_DIR:-/nonexistent}"
# shellcheck disable=SC1091
source "$REPO/benchmarks/cluster/fq_models.sh"
DS="${DS:-5 20 50}"; NODES_LIST="${NODES_LIST:-5 20 50}"
SHIFTS="${SHIFTS:-0 +2 -2}"; REALS="${REALS:-5}"; DRAWS="${DRAWS:-30}"
OUT4="${OUT4:-$SC/fq4_dumps}"
CMFQ_DUMPS="${CMFQ_DUMPS:-$SC/cmech_fq_dumps}"
OUT_DIR="${OUT_DIR:-$HOME/projects/def-rgrosse/$USER/fq4_scores}"

# Build the IN-SCOPE cell paths explicitly. Globbing $OUT4/*/ counts leftover cells
# from earlier, wider runs -- d10/d30/d40 and r5-r9 -- which is how a model showed
# 173/45. Those are being deleted in the background and are not part of this run.
CS_PATHS=(); for sh in $SHIFTS; do for d in $DS; do
  for (( r=0; r<REALS; r++ )); do
    CS_PATHS+=("$OUT4/Observed_Confounder_shift${sh}_d${d}_r${r}")
  done
done; done
CM_PATHS=(); for n in $NODES_LIST; do
  for (( r=0; r<REALS; r++ )); do
    CM_PATHS+=("$CMFQ_DUMPS/CMECH_n${n}_${SUBSET:-nonzero}_r${r}")
  done
done
NCS=${#CS_PATHS[@]}
NCM=${#CM_PATHS[@]}

echo "=== queue ==="
squeue --me -o "%.10i %.22j %.2t %.10L %R" | head -34
echo
printf '%-22s %-14s %-14s\n' model "case ($NCS)" "cmech ($NCM)"
printf '%-22s %-14s %-14s\n' ---------------------- -------------- --------------
for r in "${ROWS[@]}"; do
    IFS='|' read -r m _rest <<<"$r"
    # files/DRAWS is the cell count. A partially written cell shows as a fraction,
    # which is honest -- it is not done and the scorer will not use it.
    a=$(find "${CS_PATHS[@]/%//$m}" -name '*.npz' ! -name 'summary.npz' 2>/dev/null | wc -l)
    b=$(find "${CM_PATHS[@]/%//$m}" -name '*.npz' ! -name 'summary.npz' 2>/dev/null | wc -l)
    printf '%-22s %-14s %-14s\n' "$m" "$((a / DRAWS))/$NCS" "$((b / DRAWS))/$NCM"
done
echo
CS=$(ls "$OUT_DIR"/*_four.json 2>/dev/null | grep -vc CMECH || true)
CM=$(ls "$OUT_DIR"/CMECH_*_four.json 2>/dev/null | wc -l)
echo "SCORED (all 13 models present, MALC done, safe from deletion):"
echo "  case study : ${CS:-0}/$NCS"
echo "  ComplexMech: ${CM:-0}/$NCM"
echo "  -> $OUT_DIR"
echo
echo "=== reapers ==="
for f in logs_fq/fq4reap_*.out; do
    [ -f "$f" ] || continue
    printf '%s: %s\n' "$(basename "$f")" "$(grep -E '^\[.*pass ' "$f" | tail -1)"
done
echo
echo "A cell is only scored once ALL 13 models have it, so the slowest model gates"
echo "every score. Run fq4_aggregate.py on \$OUT_DIR for the table at any point."
