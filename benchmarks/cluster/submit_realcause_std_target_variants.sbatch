#!/bin/bash
#SBATCH --job-name=rc-std-target
#SBATCH --output=logs_rc_std_target/eval_%A_%a.out
#SBATCH --error=logs_rc_std_target/eval_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --array=0-9

# 10-task array — 2 methods (graph2d, cpfn2d) × 5 RealCause datasets, both
# running with the pooled std_target Y-scaling from DoPFN-bb:
#     y_scaled = (y - mean(y)) * STD_TARGET / std(y)   →   σ_scaled = STD_TARGET
#
# STD_TARGET is scaled per-model to preserve the σ/edge_half ratio (0.3):
#   graph2d (edges [-1,+1])   → STD_TARGET = 0.3
#   cpfn2d  (edges [-10,+10]) → STD_TARGET = 3.0
#
# Writes results to $OUT_ROOT/{graph2d_std,cpfn2d_std}/<DATASET>/ so they can
# be compared side-by-side with the baseline mega-sbatch output.

set -euo pipefail
mkdir -p logs_rc_std_target

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
UWYK_REPRO="${UWYK_REPRO:-$DEPLOY_ROOT/external/uwyk_reproduce}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT required — per-method dirs land under it}"

CKPT_GRAPH2D="${CKPT_GRAPH2D:-$REPO/Required_checkpoints/graph2d_step_50000.pt}"
CKPT_CPFN2D="${CKPT_CPFN2D:-/scratch/furkanbd/rpfn_bench_kit/cpfn2d_j32_random_A1_h100/step_checkpoints/run/step_0050000.pt}"

METHOD_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
DATASET_IDX=$(( SLURM_ARRAY_TASK_ID % 5 ))

case "$METHOD_IDX" in
    0) METHOD=graph2d_std ;;
    1) METHOD=cpfn2d_std  ;;
    *) echo "invalid METHOD_IDX=$METHOD_IDX" >&2; exit 2 ;;
esac
case "$DATASET_IDX" in
    0) DATASET=IHDP     ;;
    1) DATASET=ACIC     ;;
    2) DATASET=CPS      ;;
    3) DATASET=PSID     ;;
    4) DATASET=PSID_bal ;;
    *) echo "invalid DATASET_IDX=$DATASET_IDX" >&2; exit 2 ;;
esac
OUT="$OUT_ROOT/$METHOD/$DATASET"
mkdir -p "$OUT"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] $METHOD  $DATASET  OUT=$OUT"

case "$METHOD" in
    graph2d_std)
        [ -f "$CKPT_GRAPH2D" ] || { echo "FATAL: CKPT_GRAPH2D not found: $CKPT_GRAPH2D" >&2; exit 1; }
        python -u "$REPO/realcause_eval/graph2d.py" \
            --dataset "$DATASET" --outdir "$OUT" \
            --ckpt "$CKPT_GRAPH2D" --repo "$REPO" \
            --causalpfn "$CAUSALPFN" --uwyk "$UWYK_REPRO" \
            --anc-mode v3b_only --density-primary-mode v3b \
            --y-scaling std --std-target 0.3
        ;;
    cpfn2d_std)
        [ -f "$CKPT_CPFN2D" ] || { echo "FATAL: CKPT_CPFN2D not found: $CKPT_CPFN2D" >&2; exit 1; }
        python -u "$REPO/realcause_eval/cpfn2d.py" \
            --dataset "$DATASET" --outdir "$OUT" \
            --ckpt "$CKPT_CPFN2D" --repo "$REPO" \
            --causalpfn "$CAUSALPFN" \
            --std-mode std_target --std-target 3.0
        ;;
esac

echo "[$(date '+%Y-%m-%d %H:%M:%S')] done  $METHOD  $DATASET"
