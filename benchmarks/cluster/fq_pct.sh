#!/bin/bash
# One percentage per thing that can be waited on. Nothing else.
#
#   bash R-PFN/benchmarks/cluster/fq_pct.sh
# Env: DS, NODES_LIST, SHIFTS, REALS, DRAWS, OUT_DIR, OUT4, CMFQ_DUMPS
set -uo pipefail
KIT="${KIT:-$PWD}"; REPO="${REPO:-$KIT/R-PFN}"
SC="${SCRATCH:?SCRATCH must be set}"
CK="${CK:-$REPO/Required_checkpoints}"; UWYKD="${UWYK_CKPT_DIR:-/nonexistent}"
# shellcheck disable=SC1091
source "$REPO/benchmarks/cluster/fq_models.sh"
DS="${DS:-5 20 50}"; NODES_LIST="${NODES_LIST:-5 20 50}"
SHIFTS="${SHIFTS:-0 +2 -2}"; REALS="${REALS:-5}"; DRAWS="${DRAWS:-30}"
SUBSET="${SUBSET:-nonzero}"; CTX="${CTX:-1000}"
OUT4="${OUT4:-$SC/fq4_dumps}"; CMFQ_DUMPS="${CMFQ_DUMPS:-$SC/cmech_fq_dumps}"
OUT_DIR="${OUT_DIR:-$HOME/projects/def-rgrosse/$USER/fq4_scores}"
NCS=$(( $(echo "$SHIFTS" | wc -w) * $(echo "$DS" | wc -w) * REALS ))
NCM=$(( $(echo "$NODES_LIST" | wc -w) * REALS ))
NM=${#ROWS[@]}

bar() { # bar <have> <total>
    local h="${1:-0}" t="$2" f i s=""
    [ -n "$h" ] || h=0
    [ "$t" -gt 0 ] || t=1
    f=$(( h * 24 / t )); [ "$f" -gt 24 ] && f=24
    i=0; while [ $i -lt 24 ]; do [ $i -lt $f ] && s="$s#" || s="$s."; i=$((i+1)); done
    printf '%s %3d%% (%s/%s)' "$s" $(( h * 100 / t )) "$h" "$t"
}
# cntp <glob> <cs|cm>: ComplexMech parts are the ones whose tag starts with CMECH, so
# the two benchmarks are split on that rather than by directory. Written as two branches
# because passing "-v -e CMECH" as one argument made grep treat "-v" as the pattern and
# the case-study counts came out empty.
cntp() {
    if [ "$2" = cm ]; then ls "$OUT_DIR"/$1 2>/dev/null | grep -c CMECH || true
    else ls "$OUT_DIR"/$1 2>/dev/null | grep -vc CMECH || true; fi
}

echo "===== SCORING (what produces the table) ====="
# A part is one (cell, model). Counted rather than inferred: these files ARE the result.
printf '  %-16s %s\n' "vx case"    "$(bar "$(cntp '.*_vx_*.md' cs)" $((NCS*NM)))"
printf '  %-16s %s\n' "vx cmech"   "$(bar "$(cntp '.*_vx_*.md' cm)"       $((NCM*NM)))"
printf '  %-16s %s\n' "raw case"   "$(bar "$(cntp '.*_none_*.md.r' cs)" $((NCS*NM)))"
printf '  %-16s %s\n' "raw cmech"  "$(bar "$(cntp '.*_none_*.md.r' cm)"       $((NCM*NM)))"
printf '  %-16s %s\n' "SCORED case"  "$(bar "$(cntp '*_four.json' cs)" "$NCS")"
printf '  %-16s %s\n' "SCORED cmech" "$(bar "$(cntp '*_four.json' cm)"       "$NCM")"

echo
echo "===== DUMPS (the three slow models; the other ten are done) ====="
cells() { # cells <root> <model> <tag glob>
    find "$1"/$3/"$2" -name '*.npz' ! -name 'summary.npz' 2>/dev/null \
      | awk -F/ -v d="$DRAWS" -v b="$(basename "$1")" '
          { for (i=1;i<=NF;i++) if ($i==b) { n[$(i+1)]++; break } }
          END { c=0; for (k in n) if (n[k]>=d) c++; print c+0 }'
}
for m in uwyk1d uwyk_bin graph2d; do
    printf '  %-16s %s\n' "$m case"  "$(bar "$(cells "$OUT4" "$m" '*')" "$NCS")"
    printf '  %-16s %s\n' "$m cmech" "$(bar "$(cells "$CMFQ_DUMPS" "$m" '*')" "$NCM")"
done
echo
echo "SCORED is the only line that matters -- those json files are the table."
echo "Dumps for the slow three gate their own rows only; the other ten are already in."
