#!/bin/bash
# Percent progress of stage 15 (temperature-off Do-PFN 2D + NLL 2D dumps).
#   bash R-PFN/revision/experiments/D262k/progress_t1.sh
# Denominators are the file counts of the finished 262k (temperature-on) run, so they are the true totals.
KIT="${KIT:-/scratch/furkanbd/rpfn_bench_kit}"
OLD=dopfn_repro_joint2d_262k NEW=dopfn_repro_joint2d_262k_t1
cnt() { find "$1" -name "$2" 2>/dev/null | wc -l; }
row() { local n=$1 t=$2 lab=$3; [ "$t" -gt 0 ] && p=$(( 100 * n / t )) || p=0
        printf '  %-34s %6s / %-6s %3s%%\n' "$lab" "$n" "$t" "$p"; TOT_N=$((TOT_N + n)); TOT_T=$((TOT_T + t)); }
TOT_N=0; TOT_T=0
echo "== jobs"; squeue --me -h -o "  %.10i %.12j %.3t %.10M" | grep -E "d262|e2-t1|nll2d" || echo "  (none running)"
echo "== progress"
row "$(cnt $SCRATCH/dumps_all/$NEW/rc 'r*.npz')"     "$(cnt $SCRATCH/dumps_all/$OLD/rc 'r*.npz')"     "RealCause dumps"
for s in shift-2 shift0 shift+2; do
  row "$(cnt $SCRATCH/dumps_all/$NEW/cs/$s 'r*.npz')" "$(cnt $SCRATCH/dumps_all/$OLD/cs/$s 'r*.npz')" "case study $s"; done
row "$(cnt $SCRATCH/cmech_dumps/$NEW '*.npz')"       "$(cnt $SCRATCH/cmech_dumps/$OLD '*.npz')"       "ComplexMech all rho"
row "$(cnt $SCRATCH/cmech_dumps_rho99/$NEW '*.npz')" "$(cnt $SCRATCH/cmech_dumps_rho99/$OLD '*.npz')" "ComplexMech rho > 0.99"
row "$(cnt $SCRATCH/e2/new/results_t1/dopfn_joint2d_262k 'shard*.npz')"  100 "E2 new world (shards)"
row "$(cnt $SCRATCH/e2/same/results_t1/dopfn_joint2d_262k 'shard*.npz')" 10  "E2 same world (shards)"
for m in graph2d cpfn2d_pooled; do for ds in IHDP ACIC; do
  row "$(cnt $SCRATCH/rc_nll2d/$m/$ds '*r*.npz')" "$(cnt $SCRATCH/rc_dens_uni/$m/$ds '*r*.npz')" "NLL $m $ds"; done; done
printf '  %-34s %6s / %-6s %3s%%\n' "TOTAL" "$TOT_N" "$TOT_T" "$(( TOT_T > 0 ? 100 * TOT_N / TOT_T : 0 ))"
