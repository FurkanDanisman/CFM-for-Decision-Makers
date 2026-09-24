#!/bin/bash
#SBATCH --job-name=a3
#SBATCH --output=logs_a3/%x_%j.out
#SBATCH --error=logs_a3/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#
# ONE job = one (model, benchmark). Loops every cell inside a single allocation
# instead of an array, so squeue shows one line per model+benchmark.
#
# RESUMABLE: a cell that finishes writes a marker under $OUT_ROOT/.a3_done, and
# a rerun skips any cell whose marker exists. So when the 3 h wall hits, you
# resubmit the identical command and it picks up at the first unfinished cell.
# Markers are written ONLY on exit 0, so a failed or wall-killed cell is retried.
#
# The inner scripts are plain bash; their #SBATCH lines are comments when run
# this way, so the eval logic and the output paths are untouched.
#
# Env: BENCH, INNER, IDX_LO, IDX_HI, OUT_ROOT, plus whatever the inner needs.
set -uo pipefail
REPO="${REPO:?REPO required}"
BENCH="${BENCH:?BENCH required}"
INNER="${INNER:?INNER required}"
LO="${IDX_LO:?IDX_LO required}"; HI="${IDX_HI:?IDX_HI required}"
STATE="${OUT_ROOT:?OUT_ROOT required}/.a3_done"
mkdir -p "$STATE"

echo "[$(date '+%F %T')] $SLURM_JOB_NAME  bench=$BENCH  cells $LO..$HI"
echo "  resume state: $STATE"
ok=0; bad=0; skip=0
for i in $(seq "$LO" "$HI"); do
    if [ -f "$STATE/$i" ]; then
        skip=$((skip+1)); continue
    fi
    echo "---- cell $i / $HI  ($(date '+%H:%M:%S')) ----"
    if SLURM_ARRAY_TASK_ID="$i" bash "$REPO/benchmarks/cluster/$INNER"; then
        touch "$STATE/$i"; ok=$((ok+1))
    else
        # Keep going: one bad cell must not cost the rest of the allocation.
        # No marker, so a rerun retries it.
        bad=$((bad+1)); echo "[warn] cell $i FAILED"
    fi
done

left=0
for i in $(seq "$LO" "$HI"); do [ -f "$STATE/$i" ] || left=$((left+1)); done
echo "[$(date '+%F %T')] done  ok=$ok  failed=$bad  skipped=$skip  STILL TO DO=$left"
if [ "$left" -gt 0 ]; then
    echo "  resubmit the SAME command to continue from cell $(
        for i in $(seq "$LO" "$HI"); do [ -f "$STATE/$i" ] || { echo "$i"; break; }; done)"
fi
