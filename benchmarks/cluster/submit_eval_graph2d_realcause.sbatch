#!/bin/bash
#SBATCH --job-name=graph2d-rc
#SBATCH --output=logs_graph2d_realcause/eval_%A_%a.out
#SBATCH --error=logs_graph2d_realcause/eval_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-4

# Evaluate the graph2d 50k checkpoint on the RealCause benchmark suite.
# 5-task array — one per dataset:
#   0: IHDP        (100 realizations)
#   1: ACIC        (10  realizations)
#   2: CPS         (100 realizations)
#   3: PSID        (100 realizations)
#   4: PSID_bal    (100 realizations, all T=1 + ≤500 T=0)
#
# Each realization: 2 adjacency modes × 2 mean estimators = 4 CATE variants.
# Output per realization: <OUT>/<DATASET>_r<idx>.npz with all 4 (pehe, err, ate).

set -euo pipefail
mkdir -p logs_graph2d_realcause

REPO="/home/lukez/projects/aip-rgrosse/lukez/CFM-for-Decision-Makers"

export UWYK="${UWYK:-$REPO/g4cfm}"
export CAUSALPFN="${CAUSALPFN:-/home/lukez/projects/aip-rgrosse/lukez/CausalPFN}"
export CKPT="${CKPT:-$REPO/Required_checkpoints/graph2d_step_50000.pt}"
export OUT_ROOT="${OUT_ROOT:-$REPO/results_graph2d_realcause/$SLURM_ARRAY_JOB_ID}"
export DENSITY_DUMP="${DENSITY_DUMP:-0}"
# export FIX_DIAG="1"

[ -f "$CKPT" ] || { echo "FATAL: checkpoint not found at $CKPT" >&2; exit 1; }

case "$SLURM_ARRAY_TASK_ID" in
    0) DATASET=IHDP     ;;
    1) DATASET=ACIC     ;;
    2) DATASET=CPS      ;;
    3) DATASET=PSID     ;;
    4) DATASET=PSID_bal ;;
    *) echo "invalid task id: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac
export DATASET
export OUT="$OUT_ROOT/$DATASET"

source "/home/lukez/projects/aip-rgrosse/lukez/CFM-for-Decision-Makers/.venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] graph2d RealCause  dataset=$DATASET  OUT=$OUT"
python -u "$REPO/benchmarks/eval_graph2d/eval_graph2d_realcause.py" --dataset "$DATASET"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] done"
