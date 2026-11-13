#!/bin/bash
#SBATCH --job-name=ate-summary-rho
#SBATCH --output=logs_ate/%x_%j.out
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
# Final step of submit_rho.sh: the in-prior summaries (recomputed, with the interval score) and the three benchmark
# CSVs, copied into R-PFN/revision/results/ate_ot/ and pushed, for the figures (ate_plots.py, run locally).
set -uo pipefail
KIT="${KIT:-$PWD}"; REPO="$KIT/R-PFN"; HERE="$REPO/revision/experiments/ate_subsample"
source "$KIT/venv/bin/activate"
R="$REPO/revision/results/ate_ot"; mkdir -p "$R"
for d in "$SCRATCH"/ate_bary/*/; do
  [ "$(basename "$d")" = _dumps ] && continue
  ls "$d"shard*.npz >/dev/null 2>&1 && python "$HERE/conformal_bary.py" "$d" > "$d/coverage.txt"
done
{ for f in "$SCRATCH"/ate_bary/*/coverage.txt; do cat "$f"; done; } > "$R/inprior.txt"
for B in rc cs cm; do cp "$SCRATCH/ate_rho/$B.csv" "$R/$B.csv" 2>/dev/null || echo "missing $B.csv"; done
cd "$REPO" && git add revision/results/ate_ot && git commit -q -m "ate_ot: OT ATE density coverage and interval score, in-prior and on RealCause / case studies / ComplexMech, every model and assumed rho

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AZV2d6TbWGuV4cMQVAXpNa" && git pull -q --rebase && git push -q && echo pushed
ls -l "$R"
