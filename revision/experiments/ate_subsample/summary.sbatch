#!/bin/bash
#SBATCH --job-name=ate-summary
#SBATCH --output=logs_ate/%x_%j.out
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
# Final step of submit_all.sh: every result of the OT ATE coverage check in one file.
OUT="$SCRATCH/ate_all_results.txt"
{
  echo "OT ATE density coverage, $(date '+%F %T')"
  echo; echo "===== In-prior (500 datasets per model; s = split-conformal scale) ====="
  for f in "$SCRATCH"/ate_bary/*/coverage.txt; do cat "$f"; done
  for B in rc cs cm; do
    D="$SCRATCH/ate_${B}_dumps"
    echo; echo "===== $(case $B in rc) echo RealCause;; cs) echo 'Case studies (pooled by case; shifts 0, +2, -2; d = 5..50)';; cm) echo 'ComplexMech N=1000 (pooled by node count)';; esac) ====="
    cat "$D/coverage_$B.txt" 2>/dev/null || echo "missing $D/coverage_$B.txt"
  done
} > "$OUT"
echo "wrote $OUT"
