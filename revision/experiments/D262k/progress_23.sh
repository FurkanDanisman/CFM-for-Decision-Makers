#!/bin/bash
# Percent progress of the stage-23 job (one matched-resolution / matched-label evaluation).
#   bash R-PFN/revision/experiments/D262k/progress_23.sh <jobid> [model name]
# Dumps: 35 cells (RealCause 5, case studies 3 shifts x 6 d, ComplexMech 6 nodes x 2 subsets).
# Scoring: 119 files (RealCause 5, case studies 3 x 6 x 6, ComplexMech 6 nodes).
J="${1:?job id}"; NAME="${2:-dopfn_repro_1d_J10_262k_t1}"   # matched label: dopfn_1d_botharms_262k_t1
LOG="/scratch/furkanbd/rpfn_bench_kit/logs_hist/t262_1d_eval_$J.out"
[ -f "$LOG" ] || { echo "no log $LOG"; exit 1; }
started=$(grep -c "^===== " "$LOG")
if grep -q "dumps done" "$LOG"; then dumped=35; else dumped=$(( started > 0 ? started - 1 : 0 )); fi
scored=$(find "$SCRATCH/hist/perreal/$NAME" "$SCRATCH/hist/perreal_e1/$NAME" -name '*.npz' -size +0 2>/dev/null | wc -l)
printf "dumps   %3d/35  (%3d%%)   now: %s\n" "$dumped" $(( 100 * dumped / 35 )) "$(grep "^===== " "$LOG" | tail -1 | sed 's/^===== //')"
printf "scoring %3d/119 (%3d%%)\n" "$scored" $(( 100 * scored / 119 ))
grep -q "ALL DONE" "$LOG" && echo "ALL DONE"
grep -E "ERROR|FAILED|Traceback" "$LOG" | tail -3
