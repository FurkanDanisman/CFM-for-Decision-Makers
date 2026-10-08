#!/bin/bash
# Progress of the NLL runs (stages 18 / 22): latest [n/total] per log, and whether it pushed.
#   bash R-PFN/revision/experiments/D262k/progress_nll.sh [jobid ...]      (default: every t1_nll* log)
cd "${KIT:-/scratch/furkanbd/rpfn_bench_kit}"
logs=(); if [ $# -gt 0 ]; then for j in "$@"; do logs+=($(ls logs_hist/*_"$j".out 2>/dev/null)); done
else logs=($(ls -t logs_hist/t1_nll*.out 2>/dev/null)); fi
for f in "${logs[@]}"; do
  ds=$(grep -o "^\[nll\] [A-Z]*" "$f" | tail -1 | cut -d' ' -f2)
  last=$(grep -o "^\[[0-9]*/[0-9]*\]" "$f" | tail -1 | tr -d '[]')
  n=${last%/*}; t=${last#*/}
  pct=$([ -n "$t" ] && [ "$t" -gt 0 ] && echo $(( 100 * n / t )) || echo 0)
  state=$(grep -q "^pushed" "$f" && echo "DONE (pushed)" || (grep -q "FAILED\|Traceback" "$f" && echo "ERROR" || echo "running"))
  printf '%-42s %-5s %9s  %3s%%  %s\n' "$(basename "$f")" "${ds:--}" "${last:--}" "$pct" "$state"
done
