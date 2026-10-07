#!/bin/bash
# Stage 1: model dumps (DENSITY_DUMP=1) with the same harness as the 150k Do-PFN 2D.
#   RealCause: one job per dataset (5)   case studies: one job per (shift, d) (18)   ComplexMech all rho: 1
#   cd $KIT && bash R-PFN/revision/experiments/D262k/1_submit_dumps.sh
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "$KIT"
[ -f "$CKPT" ] || { echo "FATAL: no $CKPT"; exit 1; }
for d in "$RC_OUT" "$CS_OUT" "$CM_OUT"; do
  [ -e "$d" ] && { echo "FATAL: $d already exists -- refusing to write into it"; exit 1; }
done
SB="$REPO/benchmarks/cluster/submit_dump_one_model.sbatch"
RC=(IHDP ACIC CPS PSID PSID_bal)
for c in 0 1 2 3 4; do
  MODEL_NAME=$NAME MODEL_IDX=0 CKPT_ENV=DOPFN_CKPT CKPT="$CKPT" BENCH=rc OUT_ROOT="$RC_OUT" CELLS=$c \
    sbatch --account=$ACCOUNT --gres=$GRES --time=$TIME --job-name="d262-rc-${RC[$c]}" "$SB"
done
D=(2 3 5 10 20 30 40 50)                       # inner cell index -> d; the paper uses d = 5..50 (cells 2..7)
for s in 0 +2 -2; do
  for c in 2 3 4 5 6 7; do
    MODEL_NAME=$NAME MODEL_IDX=0 CKPT_ENV=DOPFN_CKPT CKPT="$CKPT" BENCH=cs OUT_ROOT="$CS_OUT" CELLS=$c SHIFT=$s \
      sbatch --account=$ACCOUNT --gres=$GRES --time=$TIME --job-name="d262-cs${s}-d${D[$c]}" "$SB"
  done
done
env MODEL_NAME=$NAME MODEL_IDX=0 CKPT_ENV=DOPFN_CKPT CKPT_PATH="$CKPT" EXTRA_ENV="" OUT_ROOT="$CM_OUT" \
    UWYK_FIG34_DATA="$SCRATCH/cmech_data_v2" CPU_ONLY=0 \
  sbatch --account=$ACCOUNT --gres=$GRES --time=$TIME --mem=32G --cpus-per-task=8 --job-name=d262-cm \
    "$REPO/benchmarks/cluster/submit_cmech_dump_one.sbatch"
echo "submitted 24 dump jobs; watch with: bash R-PFN/revision/experiments/D262k/progress.sh"
