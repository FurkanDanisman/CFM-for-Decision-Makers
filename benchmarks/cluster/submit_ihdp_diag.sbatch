#!/bin/bash
#SBATCH --job-name=rpfn-ihdp-diag
#SBATCH --output=logs_ihdp_diag/task_%A.out
#SBATCH --error=logs_ihdp_diag/task_%A.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

# One-off IHDP diagnostic runner. Set DIAG_CMD to which script to run.
# Defaults to the correlation diagnostic; override on submit line:
#     DIAG_CMD="diagnose_shapes.py --shard ... --out-dir ..." sbatch ...

set -euo pipefail
mkdir -p logs_ihdp_diag

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
[ ! -d "$REPO/benchmarks" ] && REPO="$DEPLOY_ROOT"
DOPFN="${DOPFN:-$DEPLOY_ROOT/external/dopfn}"
CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
CHECKPOINT50="${CHECKPOINT50:-$REPO/checkpoints/step_50000_final.pt}"
CHECKPOINT10="${CHECKPOINT10:-$REPO/checkpoints_dopfn/step_50000_final.pt}"
REALIZATION="${REALIZATION:-0}"
MALC_B="${MALC_B:-60}"
N_EVAL="${N_EVAL:-200}"
MALC_MAX_K="${MALC_MAX_K:-1}"

DIAG_CMD="${DIAG_CMD:-diagnose_correlation.py \
    --repo $REPO \
    --causalpfn $CAUSALPFN \
    --dopfn $DOPFN \
    --checkpoint50 $CHECKPOINT50 \
    --checkpoint10 $CHECKPOINT10 \
    --realization $REALIZATION \
    --malc-B $MALC_B \
    --n-eval $N_EVAL \
    --malc-max-K $MALC_MAX_K}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DIAG_CMD=$DIAG_CMD"
echo "  REPO=$REPO   CAUSALPFN=$CAUSALPFN   DOPFN=$DOPFN"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

python "$REPO/benchmarks/l2_ihdp/"$DIAG_CMD

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE"
