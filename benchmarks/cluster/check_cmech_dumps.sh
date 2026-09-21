#!/bin/bash
# Did the ComplexMech dumps actually work?
#
# Each model job walks 12 cells: 6 node counts x 2 subsets at N=1000, each cell
# holding ~100 realizations. Layout written by submit_cmech_pehe_1d_vs_2d.sbatch:
#
#   <root>/N1000/<harness>/CMECH_n<NODE>_<SUBSET>/*.npz
#
# Exit 0 is not the test. Several jobs left the queue in under two minutes, which is
# far too fast for 12 cells, and this pipeline exits 0 on a rejected dataset name, an
# adapter that globs nothing, and a harness whose wrapper swallows the error. So this
# counts cells and realizations, and optionally scores one cell to prove the dump is
# readable rather than merely present.
#
#   bash R-PFN/benchmarks/cluster/check_cmech_dumps.sh
#   SCORE=1 bash ...        also score one cell per model (slower, definitive)

set -uo pipefail
_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
KIT="${KIT:-$(cd "$_SELF_DIR/../../.." && pwd)}"
REPO="${REPO:-$KIT/R-PFN}"
SC="${SCRATCH:?SCRATCH must be set}"
ROOT="${CMECH_DUMPS:-$SC/cmech_dumps}"
NODES="${NODES:-5 10 20 30 40 50}"
SUBSETS="${SUBSETS:-nonzero zero}"
EXPECT_REAL="${EXPECT_REAL:-100}"
CTX="${CTX:-1000}"

[ -d "$ROOT" ] || { echo "FATAL: no dump root at $ROOT" >&2; exit 1; }

n_cells_expected=0
for n in $NODES; do for s in $SUBSETS; do n_cells_expected=$((n_cells_expected+1)); done; done

printf '%-22s %-8s %-9s %-11s %s\n' MODEL CELLS NPZ SHORT_CELLS VERDICT
printf '%.0s-' {1..74}; echo

TOTAL_BAD=0
for d in "$ROOT"/*; do
    [ -d "$d" ] || continue
    name="$(basename "$d")"
    cells=0; npz=0; short=0; missing=""
    for n in $NODES; do
        for s in $SUBSETS; do
            # the harness subdir name is not known here, so glob it
            cnt=$(find "$d/N$CTX" -type d -name "CMECH_n${n}_${s}" 2>/dev/null \
                  -exec find {} -name '*.npz' \; 2>/dev/null | wc -l | tr -d ' ')
            if [ "$cnt" -gt 0 ]; then
                cells=$((cells+1)); npz=$((npz+cnt))
                [ "$cnt" -lt "$EXPECT_REAL" ] && { short=$((short+1))
                    missing="$missing n${n}_${s}($cnt)"; }
            else
                missing="$missing n${n}_${s}(0)"
            fi
        done
    done
    verdict="OK"
    [ "$cells" -lt "$n_cells_expected" ] && { verdict="INCOMPLETE"; TOTAL_BAD=$((TOTAL_BAD+1)); }
    [ "$cells" = 0 ] && verdict="NO DUMPS"
    [ "$short" -gt 0 ] && [ "$verdict" = OK ] && { verdict="SHORT CELLS"; TOTAL_BAD=$((TOTAL_BAD+1)); }
    printf '%-22s %-8s %-9s %-11s %s\n' \
        "$name" "$cells/$n_cells_expected" "$npz" "$short" "$verdict"
    [ -n "$missing" ] && echo "    missing/short:$missing"
done

echo
echo "expected per model: $n_cells_expected cells x ~$EXPECT_REAL realizations"
echo "first error in each cmech job log:"
for f in $(ls -t logs_cmech_dump/*.err 2>/dev/null | head -14); do
    msg=$(grep -m1 -E "^[A-Za-z]*Error|Traceback|FATAL|invalid choice|no usable|CELL FAILED" "$f" 2>/dev/null)
    printf '  %-28s %s\n' "$(basename "$f")" "${msg:-clean}"
done

if [ "${SCORE:-0}" = 1 ]; then
    echo
    echo "--- scoring one cell per model (proves the dump is readable)"
    source "$KIT/venv/bin/activate" 2>/dev/null
    for d in "$ROOT"/*; do
        [ -d "$d" ] || continue
        printf '  %-22s ' "$(basename "$d")"
        out=$(python -u "$REPO/UWYK_Fig3_4/cate_density_metrics.py" \
                --root "$d/N$CTX" --nodes 5 --subset nonzero --context "$CTX" \
                --target cate --tau-smoother none --max-real 2 2>&1)
        rows=$(printf '%s\n' "$out" | awk -F'|' '
            NF>8 { f=$3; gsub(/ /,"",f); if (f ~ /^[0-9]+$/ && f+0>0) c++ } END {print c+0}')
        [ "$rows" -gt 0 ] && echo "scored $rows row(s)" || echo "0 scored rows -- NOT readable"
    done
fi

echo
echo "models with problems: $TOTAL_BAD"
exit 0
