#!/bin/bash
#SBATCH --job-name=rpfn-syn-l2
#SBATCH --output=logs_syn_l2/task_%A_%a.out
#SBATCH --error=logs_syn_l2/task_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-99%20

# Per-seed density L2 evaluation on linear-Gaussian synthetic SCM.
# Default: 100 seeds, N=500, d=5. Truth is closed-form Gaussian per query.

set -euo pipefail
mkdir -p logs_syn_l2

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
[ ! -d "$REPO/benchmarks" ] && REPO="$DEPLOY_ROOT"
DOPFN="${DOPFN:-$DEPLOY_ROOT/external/dopfn}"
CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
CHECKPOINT50="${CHECKPOINT50:-$REPO/checkpoints/step_50000_final.pt}"
CHECKPOINT_DOPFN_BB="${CHECKPOINT_DOPFN_BB:-$DEPLOY_ROOT/checkpoints_dopfn_backbone_j10/step_200000.pt}"
N_TRAIN="${N_TRAIN:-500}"
N_TEST="${N_TEST:-200}"
SYN_D="${SYN_D:-5}"
MALC_B="${MALC_B:-500}"
MALC_MAX_K="${MALC_MAX_K:-3}"
METHODS="${METHODS:-dopfn,ours_dopfn_bb}"
OUT="${OUT:-$DEPLOY_ROOT/l2_syn/out}"

SEED="${SLURM_ARRAY_TASK_ID:-0}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] seed=$SEED d=$SYN_D N=$N_TRAIN methods=$METHODS"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

mkdir -p "$(dirname "$OUT")"

python "$REPO/benchmarks/l2_syn/eval_realization.py" \
    --seed                "$SEED" \
    --out                 "$OUT" \
    --repo                "$REPO" \
    --dopfn               "$DOPFN" \
    --causalpfn           "$CAUSALPFN" \
    --checkpoint50        "$CHECKPOINT50" \
    --checkpoint-dopfn-bb "$CHECKPOINT_DOPFN_BB" \
    --n-train             "$N_TRAIN" \
    --n-test              "$N_TEST" \
    --syn-d               "$SYN_D" \
    --malc-B              "$MALC_B" \
    --malc-max-K          "$MALC_MAX_K" \
    --methods             "$METHODS"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE seed=$SEED"
