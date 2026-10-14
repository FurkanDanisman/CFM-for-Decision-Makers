#!/bin/bash
# Stage 25: inference time of the six RealCause models (IHDP r0) -> revision/results/d262t1/inference_time.{csv,md}.
# ONE GPU job; the six models run sequentially (benchmarks/timing/inference_time.py), then the results are pushed.
#   cd $KIT && git -C R-PFN pull --rebase && bash R-PFN/revision/experiments/D262k/25_inference_time.sh
# The same file is the job body: sbatch --wrap runs "bash <this file> run" (never `source` under sh).
set -uo pipefail
KIT="${KIT:-/scratch/furkanbd/rpfn_bench_kit}"
SELF="$KIT/R-PFN/revision/experiments/D262k/25_inference_time.sh"

if [ "${1:-}" != run ]; then
  cd "$KIT" && mkdir -p logs_inftime
  sbatch --account=aip-rgrosse --gres=gpu:l40s:1 --time="${IT_TIME:-2:00:00}" --cpus-per-task=8 --mem=32G \
    --job-name=inftime -o logs_inftime/inftime_%j.out --wrap "KIT=$KIT bash $SELF run"
  exit $?
fi

# ---- job body ----
# Do-PFN 2D needs DOPFN_SOFTMAX_TEMP=0, so the name must be the _t1 one (common.sh enforces the pair).
# inference_time.py clears these per model and sets DOPFN_SOFTMAX_TEMP=0 only for Do-PFN 2D.
export D262_NAME=dopfn_repro_joint2d_262k_t1 DOPFN_SOFTMAX_TEMP=0
source "$KIT/R-PFN/revision/experiments/D262k/common.sh"; cd "$KIT"; source venv/bin/activate
export CAUSALPFN="$KIT/external/causalpfn" DOPFN_ROOT="$KIT/external/dopfn" UWYK="$KIT/external/uwyk_reproduce"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
O="$REPO/revision/results/d262t1"; mkdir -p "$O"
nvidia-smi --query-gpu=name --format=csv,noheader
python -u "$REPO/benchmarks/timing/inference_time.py" --kit "$KIT" --repo "$REPO" --out-dir "$O" \
  --work-dir "$SCRATCH/inference_time_work" --dopfn2d-ckpt "$CKPT" || echo "inference_time.py exited non-zero (see log)"
cd "$REPO" && git add revision/results/d262t1/inference_time.* \
  && git commit -q -m "d262t1: inference time of the six RealCause models (IHDP r0)" \
  && git pull -q --rebase && git push -q && echo pushed
