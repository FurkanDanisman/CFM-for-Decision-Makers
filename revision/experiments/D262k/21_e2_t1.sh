#!/bin/bash
# Stage 21: in-prior (E2) tables of the temperature-off Do-PFN 2D (results_t1 roots hold only that model).
#   cd $KIT && git -C R-PFN pull --rebase && sbatch --account=aip-rgrosse --time=1:00:00 --cpus-per-task=4 --mem=32G \
#        -o logs_e2/t1_tables_%j.out --wrap "bash R-PFN/revision/experiments/D262k/21_e2_t1.sh"
set -uo pipefail
KIT="${KIT:-$PWD}"; REPO="$KIT/R-PFN"; cd "$KIT"; source venv/bin/activate
O="$REPO/revision/results/d262t1"; mkdir -p "$O"; T="$REPO/revision/experiments/E2_in_prior/e2_tables.py"
python "$T" $SCRATCH/e2/new/data  $SCRATCH/e2/new/results_t1  1000,10000 > "$O/e2_new_world.txt"
python "$T" $SCRATCH/e2/same/data $SCRATCH/e2/same/results_t1 1000       > "$O/e2_same_world.txt"
cat "$O/e2_new_world.txt" "$O/e2_same_world.txt"
cd "$REPO" && git add revision/results/d262t1/e2_*.txt && git commit -q -m "d262t1: E2 in-prior tables, temperature off" \
  && git pull -q --rebase && git push -q && echo pushed
