#!/bin/bash
#SBATCH --job-name=a3
#SBATCH --output=logs_a3/%x_%j.out
#SBATCH --error=logs_a3/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#
# ONE job = one (model, benchmark). Loops every cell inside a single allocation
# instead of an array, so squeue shows one line per model+benchmark.
#
# The inner scripts are plain bash; their #SBATCH lines are comments when run
# this way, so the eval logic and the output paths are untouched.
#
# Env: BENCH (rc|cs|cm), IDX_LO, IDX_HI, INNER, plus whatever the inner needs.
set -uo pipefail
REPO="${REPO:?REPO required}"
BENCH="${BENCH:?BENCH required}"
INNER="${INNER:?INNER required}"
LO="${IDX_LO:?IDX_LO required}"; HI="${IDX_HI:?IDX_HI required}"

echo "[$(date '+%F %T')] $SLURM_JOB_NAME  bench=$BENCH  cells $LO..$HI"
ok=0; bad=0
for i in $(seq "$LO" "$HI"); do
    echo "---- cell $i / $HI ----"
    if SLURM_ARRAY_TASK_ID="$i" bash "$REPO/benchmarks/cluster/$INNER"; then
        ok=$((ok+1))
    else
        # Keep going: one bad cell must not cost the rest of the allocation.
        bad=$((bad+1)); echo "[warn] cell $i failed"
    fi
done
echo "[$(date '+%F %T')] done  ok=$ok  failed=$bad"
