#!/bin/bash
# One TSV row per training job: which output dir it wrote, where it resumed
# from, how far it got, and its true wall-clock window.
#
#   CLUSTER=fir bash R-PFN/benchmarks/training_run_ledger.sh > ledger_fir.tsv
#
# Wall-clock comes from the log's own header timestamp to the file mtime, NOT
# from sacct: Slurm keeps only the final record of a requeued job, so several
# of these runs report Elapsed 00:00:00 while their logs show tens of thousands
# of steps and their .err files are tens of MB.
#
# Summing the rows that share an output dir gives the TOTAL cost of reaching
# that model's final step, which is what the cost table reports.
set -uo pipefail
CLUSTER="${CLUSTER:?CLUSTER required}"
printf 'cluster\tlogdir\tjob\tout_dir\ttag\tresume_step\tlast_step\tstart\tmtime\thours\n'

for D in logs_*; do
    [ -d "$D" ] || continue
    for F in "$D"/train_*.out; do
        [ -f "$F" ] || continue
        jid=$(basename "$F" | grep -oE '[0-9]+' | head -1)

        # Output dir identifies the MODEL; jobs sharing it are one training.
        out=$(grep -ohE "(OUT_DIR|CHECKPOINT_DIR|OUT)=[^ ]+" "$F" 2>/dev/null \
              | head -1 | cut -d= -f2-)
        tag=$(grep -ohE "TAG=[A-Za-z0-9_]+|variant=[A-Za-z0-9_]+|J=[0-9]+" "$F" 2>/dev/null \
              | head -2 | tr '\n' ',' | sed 's/,$//')
        res=$(grep -ohE "resuming from actual_step=[0-9]+|\[resume\] found" "$F" 2>/dev/null \
              | head -1 | grep -oE '[0-9]+$')
        [ -z "$res" ] && grep -q "no latest.pt" "$F" 2>/dev/null && res=0
        last=$(tr '\r' '\n' < "$F" 2>/dev/null \
               | grep -oE "done at step [0-9]+|reached actual_step=[0-9]+|^ *[0-9]{4,}" \
               | grep -oE '[0-9]+' | sort -n | tail -1)

        st=$(grep -ohE '^\[2026-[0-9-]+ [0-9:]+\]' "$F" 2>/dev/null | head -1 | tr -d '[]')
        mt=$(stat -c '%y' "$F" 2>/dev/null | cut -d. -f1)
        h=""
        if [ -n "$st" ] && [ -n "$mt" ]; then
            a=$(date -d "$st" +%s 2>/dev/null); b=$(date -d "$mt" +%s 2>/dev/null)
            [ -n "$a" ] && [ -n "$b" ] && [ "$b" -gt "$a" ] \
              && h=$(awk -v x=$((b-a)) 'BEGIN{printf "%.3f", x/3600}')
        fi
        # Skip logs with no training evidence at all.
        [ -z "$last" ] && [ -z "$out" ] && continue
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "$CLUSTER" "$D" "$jid" "${out:-?}" "${tag:-?}" "${res:-?}" "${last:-?}" \
          "${st:-?}" "${mt:-?}" "${h:-?}"
    done
done
