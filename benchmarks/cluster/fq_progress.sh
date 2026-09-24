#!/bin/bash
# Fixed-query dump progress: both settings, all 13 models, one table.
#
# Lists every model the driver KNOWS about, not just those with files on disk --
# a model that was never submitted (e.g. skipped for a git-lfs pointer
# checkpoint) then shows as 0 instead of vanishing from the report, which is how
# graph2d/dopfn_bb/cpfn1d_j1024 went unnoticed.
#
#   bash R-PFN/benchmarks/cluster/fq_progress.sh
# Env: A, B (dump parents), DRAWS (expected count)
set -uo pipefail
SC="${SCRATCH:?SCRATCH must be set}"
A="${A:-$SC/cs_fq1000_dumps}"
B="${B:-$SC/cs_fq1000_f100_dumps}"
DRAWS="${DRAWS:-1000}"
MODELS="dopfn_native dopfn_repro_1d_J10 dopfn_repro_1d_J100 dopfn_repro_joint2d
        dopfn_bb uwyk1d uwyk_bin graph2d cpfn1d_j1024 cpfn1d_j32
        cpfn1d_botharms cpfn_v0 cpfn2d_eta0"

count() {  # count <parent> <model>
    [ -d "$1/$2" ] || { echo -; return; }
    find "$1/$2" -name '*.npz' ! -name 'summary.npz' 2>/dev/null | wc -l
}
bar() {    # bar <n> <total>
    [ "$1" = "-" ] && { printf '%-20s' "not submitted"; return; }
    local f=$(( $1 * 20 / ($2 > 0 ? $2 : 1) )) i=0 s=""
    while [ $i -lt 20 ]; do [ $i -lt $f ] && s="$s#" || s="$s."; i=$((i+1)); done
    printf '%s' "$s"
}

echo "=== queue ==="
squeue --me --format="%.12i %.16j %.2t %.11L %.18R" 2>/dev/null | head -32
echo
printf "%-22s | %6s %-20s | %6s %-20s\n" "model" "A" "" "B" ""
printf "%-22s-+-%6s-%-20s-+-%6s-%-20s\n" "----------------------" "------" \
       "--------------------" "------" "--------------------"
ta=0; tb=0; na=0; nb=0
for m in $MODELS; do
    a=$(count "$A" "$m"); b=$(count "$B" "$m")
    [ "$a" != "-" ] && { ta=$((ta+a)); na=$((na+1)); }
    [ "$b" != "-" ] && { tb=$((tb+b)); nb=$((nb+1)); }
    printf "%-22s | %6s %s | %6s %s\n" "$m" "$a" "$(bar "$a" "$DRAWS")" \
           "$b" "$(bar "$b" "$DRAWS")"
done
echo
echo "A: $na/13 models present, $ta/$((13*DRAWS)) files   ($A)"
echo "B: $nb/13 models present, $tb/$((13*DRAWS)) files   ($B)"
echo
echo "A = row 0 frozen only;  B = rows 0-99 frozen.  Both dumped with NQ=1,"
echo "so each scores ONLY row 0 -- the two differ just in how much of the"
echo "surrounding context is held fixed."
