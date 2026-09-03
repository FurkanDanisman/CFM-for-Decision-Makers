#!/bin/bash
# Fires the stdmode sweep — 2 checkpoints × 4 non-baseline modes = 8 jobs.
# Usage:
#   CKPT_WIDE=/path/to/edges_-10_10/step_XXXXX.pt \
#   CKPT_TIGHT=/path/to/edges_-4_4/step_XXXXX.pt \
#   bash submit_all_stdmodes.sh
#
# Optional: MAX_REAL=10 for a smoke run.
set -euo pipefail

CKPT_WIDE="${CKPT_WIDE:?set CKPT_WIDE to the -10,10 model checkpoint}"
CKPT_TIGHT="${CKPT_TIGHT:?set CKPT_TIGHT to the -4,4 model checkpoint}"
MAX_REAL="${MAX_REAL:-}"

SBATCH_FILE="$(cd "$(dirname "$0")" && pwd)/submit_eval_cpfn2d_ihdp_stdmodes.sbatch"
[ -f "$SBATCH_FILE" ] || { echo "FATAL: sbatch not found: $SBATCH_FILE" >&2; exit 1; }

# Include pooled as a sanity control (should reproduce prior baseline numbers).
MODES=(pooled per_arm winsor log recursive)

for CKPT in "$CKPT_WIDE" "$CKPT_TIGHT"; do
    for MODE in "${MODES[@]}"; do
        echo "→ submitting  CKPT=$(basename $(dirname "$CKPT"))/$(basename "$CKPT")  STD_MODE=$MODE"
        sbatch --export=ALL,CKPT="$CKPT",STD_MODE="$MODE",MAX_REAL="$MAX_REAL" "$SBATCH_FILE"
    done
done
echo "─── all ${#MODES[@]} × 2 = $((${#MODES[@]} * 2)) jobs submitted."
