#!/bin/bash
# fq4 progress: cells generated, and dumps done per model.
#
# Counts files rather than trusting job state: a job can be RUNNING and stuck, or
# COMPLETED having skipped everything. Cells and dumps on disk are the only thing
# the scorer actually reads.
#
#   bash R-PFN/benchmarks/cluster/fq4_progress.sh
# Env: CASES, SHIFTS, DS, REALS, DRAWS, FQ4, OUT4, CTX
set -uo pipefail
KIT="${KIT:-$PWD}"; REPO="${REPO:-$KIT/R-PFN}"
SC="${SCRATCH:?SCRATCH must be set}"
CK="${CK:-$REPO/Required_checkpoints}"; UWYKD="${UWYK_CKPT_DIR:-/nonexistent}"
# shellcheck disable=SC1091
source "$REPO/benchmarks/cluster/fq_models.sh"
CASES="${CASES:-Observed_Confounder}"
SHIFTS="${SHIFTS:-0 +2 -2}"
read -r -a DSA <<<"${DS:-5 10 20 30 40 50}"
REALS="${REALS:-10}"; DRAWS="${DRAWS:-100}"; CTX="${CTX:-1000}"
FQ4="${FQ4:-$SC/fq4}"; OUT4="${OUT4:-$SC/fq4_dumps}"

TAGS=()
for c in $CASES; do for s in $SHIFTS; do for d in "${DSA[@]}"; do
  for (( r=0; r<REALS; r++ )); do TAGS+=("${c}_shift${s}_d${d}_r${r}"); done
done; done; done
NT=${#TAGS[@]}

echo "=== queue ==="
squeue --me --format="%.12i %.18j %.2t %.12L %.18R" 2>/dev/null | head -20
echo
gen_full=0; gen_part=0; gen_none=0
for t in "${TAGS[@]}"; do
    c="${t%%_shift*}"
    n=$(find "$FQ4/$t/$c/N$CTX" -name "${c}_*.npz" 2>/dev/null | wc -l)
    if   [ "$n" -ge "$DRAWS" ]; then gen_full=$((gen_full+1))
    elif [ "$n" -gt 0 ];        then gen_part=$((gen_part+1))
    else                             gen_none=$((gen_none+1)); fi
done
echo "cells: $gen_full/$NT generated   ($gen_part partial, $gen_none absent)"
echo
printf '%-22s %7s %7s %7s  %s\n' model done partial none progress
for r in "${ROWS[@]}"; do
    IFS='|' read -r m _h _e <<<"$r"
    d=0; p=0; z=0
    for t in "${TAGS[@]}"; do
        n=$(find "$OUT4/$t/$m" -name '*.npz' ! -name 'summary.npz' 2>/dev/null | wc -l)
        if   [ "$n" -ge "$DRAWS" ]; then d=$((d+1))
        elif [ "$n" -gt 0 ];        then p=$((p+1))
        else                             z=$((z+1)); fi
    done
    f=$(( d * 20 / (NT > 0 ? NT : 1) )); bar=""; i=0
    while [ $i -lt 20 ]; do [ $i -lt $f ] && bar="$bar#" || bar="$bar."; i=$((i+1)); done
    printf '%-22s %7s %7s %7s  %s\n' "$m" "$d" "$p" "$z" "$bar"
done
echo
echo "cells=$NT per model. 'done' = all $DRAWS dumps present, so it is what the"
echo "scorer will find; a RUNNING job with done=0 is still on its first cell."
echo
echo "score one finished cell:"
echo "  DUMPS=$OUT4/${TAGS[0]} QUERIES=10 SKIP_MALC=1 \\"
echo "    bash $REPO/benchmarks/cluster/score_fq4.sh"
