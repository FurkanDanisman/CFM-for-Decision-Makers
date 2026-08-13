#!/bin/bash
#SBATCH --job-name=fig2-pehe-l2
#SBATCH --output=logs_fig2/task_%A_%a.out
#SBATCH --error=logs_fig2/task_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-5

# 6-way array: one ρ per task (ρ ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0}).
# Each task writes an .npz shard and a diagnostic .npz. After all shards
# land, run with --plot to produce the summary PNG, per-ρ diagnostic PNGs,
# and printed tables.
#
# Env overrides:
#   DEPLOY_ROOT, REPO, DOPFN, CAUSALPFN, UWYK_SRC, UWYK_CKPT_DIR,
#   CHECKPOINT50, CHECKPOINT_DOPFN_BB, K, N_CONTEXT, N_TEST, UWYK_N_SAMPLES,
#   OUT

set -euo pipefail
mkdir -p logs_fig2

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
DOPFN="${DOPFN:-$DEPLOY_ROOT/external/dopfn}"
CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
UWYK_SRC="${UWYK_SRC:-$DEPLOY_ROOT/external/uwyk/src}"
UWYK_CKPT_DIR="${UWYK_CKPT_DIR:-$DEPLOY_ROOT/external/uwyk/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0}"
CHECKPOINT50="${CHECKPOINT50:-$REPO/checkpoints/step_50000_final.pt}"
CHECKPOINT_DOPFN_BB="${CHECKPOINT_DOPFN_BB:-$DEPLOY_ROOT/checkpoints_dopfn_backbone_j10/step_200000.pt}"
K="${K:-20}"
N_CONTEXT="${N_CONTEXT:-200}"
N_TEST="${N_TEST:-50}"
UWYK_N_SAMPLES="${UWYK_N_SAMPLES:-500}"
OUT="${OUT:-$DEPLOY_ROOT/fig2_pehe_l2/out}"

mkdir -p "$(dirname "$OUT")"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

RHO_IDX="${SLURM_ARRAY_TASK_ID:-0}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] rho_index=$RHO_IDX"

python "$REPO/benchmarks/empirical_tests/fig2_pehe_l2.py" \
    --repo               "$REPO" \
    --dopfn              "$DOPFN" \
    --causalpfn          "$CAUSALPFN" \
    --uwyk-src           "$UWYK_SRC" \
    --uwyk-ckpt-dir      "$UWYK_CKPT_DIR" \
    --checkpoint50       "$CHECKPOINT50" \
    --checkpoint-dopfn-bb "$CHECKPOINT_DOPFN_BB" \
    --K                  "$K" \
    --N-context          "$N_CONTEXT" \
    --N-test             "$N_TEST" \
    --uwyk-n-samples     "$UWYK_N_SAMPLES" \
    --out                "$OUT" \
    --rho-index          "$RHO_IDX"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE rho_index=$RHO_IDX"
