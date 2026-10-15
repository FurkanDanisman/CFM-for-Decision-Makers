#!/bin/bash
# Revision v2, E1/E2: tabulate the coupling runs with the paper's aggregator (e8_tables.py: same pooling,
# per-realization means, SE over realizations) and push the long CSV to revision/results/d262t1/coupling.csv.
#   cd $KIT && git -C R-PFN pull --rebase && sbatch --account=aip-rgrosse --time=0:30:00 --cpus-per-task=2 --mem=8G \
#       -o logs_coupling/collect_%j.out --wrap "bash R-PFN/revision/experiments/coupling/collect_coupling.sh"
set -uo pipefail
KIT="${KIT:-$PWD}"; REPO="$KIT/R-PFN"; source "$KIT/venv/bin/activate"
H="$SCRATCH/hist_coupling"; O="$REPO/revision/results/d262t1"
python "$REPO/revision/experiments/E1_E8_rescoring/e8_tables.py" --perreal "$H/perreal" --e1-perreal "$H/perreal_e1" \
  --out-dir "$H/tables" --stages comonotonic gauss_-0.5 gauss_0.5 makarov joint_indep joint_makarov \
  && cp "$H/tables/e8_long.csv" "$O/coupling.csv" || { echo "FAILED"; exit 1; }
echo "rows: $(($(wc -l < "$O/coupling.csv") - 1))"; cut -d, -f5 "$O/coupling.csv" | sort | uniq -c
cd "$REPO" && git add revision/results/d262t1/coupling.csv && git commit -q -m "d262t1: coupling sensitivity (E1/E2) long table" \
  && git pull -q --rebase && git push -q && echo pushed
