#!/bin/bash
# Stage 4: ComplexMech rho > 0.99 dumps of the 262k Do-PFN 2D (for the appendix tables), beside
# cmech_dumps_rho99/dopfn_repro_joint2d.     cd $KIT && bash R-PFN/revision/experiments/D262k/4_rho99_dumps.sh
set -euo pipefail
source "$(dirname "$0")/common.sh"; cd "$KIT"
OUT="$SCRATCH/cmech_dumps_rho99/$NAME"
[ -e "$OUT" ] && { echo "FATAL: $OUT already exists -- refusing to write into it"; exit 1; }
[ -d "$SCRATCH/cmech_data_rho99/complexmech" ] || { echo "FATAL: no rho > 0.99 data at $SCRATCH/cmech_data_rho99"; exit 1; }
env MODEL_NAME=$NAME MODEL_IDX=0 CKPT_ENV=DOPFN_CKPT CKPT_PATH="$CKPT" EXTRA_ENV="" OUT_ROOT="$OUT" \
    UWYK_FIG34_DATA="$SCRATCH/cmech_data_rho99" CPU_ONLY=0 \
  sbatch --account=$ACCOUNT --gres=$GRES --time=$TIME --mem=32G --cpus-per-task=8 --job-name=d262-cm99 \
    "$REPO/benchmarks/cluster/submit_cmech_dump_one.sbatch"
