#!/bin/bash
#SBATCH --job-name=cpfn1d-rc-50k
#SBATCH --output=logs_cpfn1d_realcause_step50k/eval_%A_%a.out
#SBATCH --error=logs_cpfn1d_realcause_step50k/eval_%A_%a.err
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-4

# 5-task array — one per RealCause dataset. Evaluates the plain CausalPFN
# (1D bar-dist head) checkpoint @ step_50000 — a.k.a. causalpfn-orig-headrand
# / cpfn1d j1024_headrand — using realcause_eval/cpfn1d.py (which wraps
# benchmarks/eval_causalpfn2d/eval_causalpfn_v0_realcause.py).
#
# Env overrides:
#   CKPT       (default step_0050000.pt in causalpfn_j1024_headrand_A1_h100)
#   OUT_ROOT   per-dataset outputs go to OUT_ROOT/<DATASET>/
#   STD_MODE   default per_arm

set -euo pipefail
mkdir -p logs_cpfn1d_realcause_step50k

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found under $DEPLOY_ROOT" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
CKPT="${CKPT:-/scratch/furkanbd/rpfn_bench_kit/causalpfn_j1024_headrand_A1_h100/step_checkpoints/run/step_0050000.pt}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT required — where per-dataset outputs land}"
STD_MODE="${STD_MODE:-per_arm}"
EVAL_MAX_CONTEXT="${EVAL_MAX_CONTEXT:-0}"   # 0 = no cap; CausalPFN uses full context
EVAL_CONTEXT_SEED="${EVAL_CONTEXT_SEED:-1}"

[ -f "$CKPT" ] || { echo "FATAL: ckpt not found: $CKPT" >&2; exit 1; }

case "$SLURM_ARRAY_TASK_ID" in
    0) DATASET=IHDP     ;;
    1) DATASET=ACIC     ;;
    2) DATASET=CPS      ;;
    3) DATASET=PSID     ;;
    4) DATASET=PSID_bal ;;
    *) echo "invalid task id: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac
OUT="$OUT_ROOT/$DATASET"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] cpfn1d @ step_50000  DATASET=$DATASET  OUT=$OUT"
echo "  CKPT=$CKPT"
echo "  STD_MODE=$STD_MODE  EVAL_MAX_CONTEXT=$EVAL_MAX_CONTEXT  EVAL_CONTEXT_SEED=$EVAL_CONTEXT_SEED"

python -u "$REPO/realcause_eval/cpfn1d.py" \
    --dataset            "$DATASET" \
    --outdir             "$OUT" \
    --ckpt               "$CKPT" \
    --repo               "$REPO" \
    --causalpfn          "$CAUSALPFN" \
    --std-mode           "$STD_MODE" \
    --eval-max-context   "$EVAL_MAX_CONTEXT" \
    --eval-context-seed  "$EVAL_CONTEXT_SEED"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] done"
