#!/bin/bash
# Job body of stage 23 (submitted by 23_eval_1d_262k.sh, which sets NAME CKPT RC CS CM99 KIT).
set -uo pipefail
cd "$KIT"; REPO="$KIT/R-PFN"
export DOPFN_SOFTMAX_TEMP=0                       # temperature off: our checkpoint, trained without it
D1="$REPO/benchmarks/cluster/submit_dump_one_model.sbatch"
echo "[$(date '+%F %T')] [1/5] RealCause dumps"
MODEL_NAME=$NAME MODEL_IDX=0 CKPT_ENV=DOPFN_CKPT CKPT="$CKPT" BENCH=rc OUT_ROOT="$RC" bash "$D1"
i=2
for s in 0 +2 -2; do
  echo "[$(date '+%F %T')] [$i/5] case-study dumps, shift $s"; i=$((i + 1))
  MODEL_NAME=$NAME MODEL_IDX=0 CKPT_ENV=DOPFN_CKPT CKPT="$CKPT" BENCH=cs OUT_ROOT="$CS" CELLS="2 3 4 5 6 7" SHIFT=$s \
    bash "$D1"
done
echo "[$(date '+%F %T')] [5/5] ComplexMech rho > 0.99 dumps"
MODEL_NAME=$NAME MODEL_IDX=0 CKPT_ENV=DOPFN_CKPT CKPT_PATH="$CKPT" EXTRA_ENV="" OUT_ROOT="$CM99" \
  UWYK_FIG34_DATA="$SCRATCH/cmech_data_rho99" CPU_ONLY=0 bash "$REPO/benchmarks/cluster/submit_cmech_dump_one.sbatch"
echo "[$(date '+%F %T')] dumps done; hist scoring"
for t in 0 1 2 3 26; do                           # RealCause, case study x 3 shifts, ComplexMech rho > 0.99
  echo "[$(date '+%F %T')] scoring task $t"
  HIST_ROWS="$NAME|$RC|$CS|-|$CM99|dopfn_native" SLURM_ARRAY_TASK_ID=$t \
    bash "$REPO/revision/experiments/E1_E8_rescoring/submit_hist_extra.sbatch"
done
echo "[$(date '+%F %T')] ALL DONE"
