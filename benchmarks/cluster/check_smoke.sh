#!/bin/bash
# Verify the smoke dumps actually produced scoreable output.
#
# "The job exited 0" is not the test. Several failure modes in this pipeline exit
# 0 and write nothing usable: a --dataset spelling the harness rejects, an npz
# backend that globs zero files and reports "summary (n=0)", a density whose
# implied point estimate does not reproduce the model's own (the scorer's
# MIN_DENSITY_R2 gate drops those realizations silently). So this scores each
# root and asserts a row came back with n > 0.
#
#   bash R-PFN/benchmarks/cluster/check_smoke.sh

set -uo pipefail
DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
SMOKE="${SMOKE:-$SCRATCH/smoke_new}"
source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1
CD="$REPO/UWYK_Fig3_4/cate_density_metrics.py"
PT="$REPO/realcause_eval/point_raw_em.py"
LOG="$SMOKE/check.log"; : > "$LOG"

printf '%-22s %-10s %-9s %-9s %s\n' MODEL BENCH NPZ SCORED NOTE
printf '%.0s-' {1..70}; echo

for d in "$SMOKE"/*; do
    [ -d "$d" ] || continue
    name="$(basename "$d")"
    for bench in rc cs; do
        root="$d/$bench"
        [ -d "$root" ] || { printf '%-22s %-10s %-9s %-9s %s\n' "$name" "$bench" - - "no root"; continue; }
        n_npz=$(find "$root" -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')

        if [ "$bench" = rc ]; then
            out=$(python -u "$CD" --root "$root" --dataset IHDP --target cate \
                    --tau-smoother none 2>&1)
        else
            cell=$(find "$root" -type d -name 'ctx*' 2>/dev/null | head -1)
            if [ -z "$cell" ]; then
                printf '%-22s %-10s %-9s %-9s %s\n' "$name" "$bench" "$n_npz" 0 "no ctx cell"
                continue
            fi
            out=$(python -u "$CD" --root "$cell" --dataset Observed_Confounder \
                    --target cate --tau-smoother none 2>&1)
        fi
        {   echo "######## $name / $bench"; echo "$out"; echo; } >> "$LOG"

        # Count the n_files column, NOT the last field. The table is
        #   | method | n_files | n_query | coverage95 | length | is05 | ... | bias |
        # and it ends with a trailing pipe, so splitting on "|" leaves $NF empty
        # and an $NF test counts zero rows for a table that scored perfectly.
        # Header and separator rows fail the numeric test on their own.
        scored=$(printf '%s\n' "$out" | awk -F'|' '
            NF>8 { f=$3; gsub(/ /,"",f); if (f ~ /^[0-9]+$/ && f+0 > 0) c++ }
            END { print c+0 }')
        note=""
        printf '%s\n' "$out" | grep -qi "no dumps" && note="no dumps seen"
        printf '%s\n' "$out" | grep -qiE "traceback|^[A-Za-z]*Error" && note="ERROR (see check.log)"
        [ "$scored" -gt 0 ] && [ -z "$note" ] && note="PASS"
        [ "$scored" -eq 0 ] && [ -z "$note" ] && note="FAIL: 0 scored rows"
        printf '%-22s %-10s %-9s %-9s %s\n' "$name" "$bench" "$n_npz" "$scored" "$note"
        # method | n_files | coverage | length | is05 -- the columns being verified
        printf '%s\n' "$out" | awk -F'|' '
            NF>8 { f=$3; gsub(/ /,"",f)
                   if (f ~ /^[0-9]+$/ && f+0 > 0) {
                     m=$2; c=$5; l=$6; i=$7
                     gsub(/^ +| +$/,"",m); gsub(/^ +| +$/,"",c)
                     gsub(/^ +| +$/,"",l); gsub(/^ +| +$/,"",i)
                     printf("      %-16s n=%-4s cov=%-7s len=%-18s IS=%s\n", m, f, c, l, i)
                   } }' 
    done
done

echo
echo "full scorer output: $LOG"
echo
echo "point estimates (RealCause/IHDP), one root per model:"
for d in "$SMOKE"/*; do
    [ -d "$d/rc" ] || continue
    printf '  %-22s ' "$(basename "$d")"
    echo
    # Drop the separator by its real shape: it is |---|---|---| with no colons,
    # so a ':---' filter let it through and it consumed one of the two lines
    # printed -- which is why only uwyk1d-noanc appeared and uwyk1d-v3a looked
    # like it had errored. Print every model row instead of the first two lines.
    python -u "$PT" --root "$d/rc" --dataset IHDP --modes raw 2>&1 \
        | grep -E '^\|' | grep -viE '^\| *method|^\|[- :|]*$' \
        | sed 's/^/        /' || echo "        (none)" 
done
