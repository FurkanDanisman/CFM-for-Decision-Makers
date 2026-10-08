#!/bin/bash
# Stage 12: why Do-PFN 2D (262k) returns NaN intervals in some E2 worlds.
# Worlds 604 (shard 6, index 4; T constant) and 1400 (shard 14, index 0; T varies), each run with the
# default fp16 inference and with DOPFN_FP32=1. The harness reports non-finite logits per query.
#   cd $KIT && sbatch --account=aip-rgrosse --gres=gpu:l40s:1 --time=0:30:00 --cpus-per-task=4 --mem=32G \
#        -o logs_e2/nan_diag_%j.out R-PFN/revision/experiments/D262k/12_nan_diag.sh
set -uo pipefail
KIT="${KIT:-$PWD}"; REPO="$KIT/R-PFN"
source "$KIT/venv/bin/activate"; export PYTHONUNBUFFERED=1
export CAUSALPFN="$KIT/external/causalpfn" DOPFN_ROOT="$KIT/external/dopfn"
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
export DATASET=CMECH_n5 DENSITY_DUMP=1
export DOPFN_CKPT="$REPO/Required_checkpoints/dopfn_repro_joint2d_step262144.pt"
for spec in "6 5" "14 1"; do
  set -- $spec; SH=$1; export MAX_REAL=$2
  export UWYK_FIG34_DATA="$SCRATCH/e2/new/data/dopfn/shard$SH"
  for fp32 in 0 1; do
    echo "=== shard $SH (first $MAX_REAL worlds)  DOPFN_FP32=$fp32"
    export DOPFN_FP32=$fp32 OUT="$SCRATCH/e2/nan_diag/s${SH}_fp32_$fp32"; rm -rf "$OUT"; mkdir -p "$OUT"
    python -u "$REPO/benchmarks/eval_scm_case_studies/eval_native_dopfn.py" --dataset "$DATASET" 2>&1 \
      | grep -E "WARN|^r=|fp16_inference"
  done
done
