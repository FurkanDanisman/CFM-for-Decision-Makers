#!/bin/bash
#SBATCH --job-name=dopfn-bb-rc
#SBATCH --output=logs_dopfn_bb_rc/eval_%A_%a.out
#SBATCH --error=logs_dopfn_bb_rc/eval_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-4

# 5-task array — one per RealCause dataset.
#   0: IHDP  1: ACIC  2: CPS  3: PSID  4: PSIDbal
#
# NOTE: DoPFN-bb saves per-realization density files as density_r<r>.npz next
# to args.out. --out here is set to $OUT/summary.npz so the sidecars land in $OUT/.
#
# Env vars:
#   CKPT       required — DoPFN-backbone checkpoint (.pt with model_state_dict + config + edges)
#   OUT_ROOT   required — per-dataset outputs go to OUT_ROOT/<DATASET>/
#   DOPFN_ROOT (optional) path to dopfn_upstream repo root
#   CAUSALPFN  (optional) path to causalpfn repo root

set -euo pipefail
mkdir -p logs_dopfn_bb_rc

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
export DOPFN_ROOT="${DOPFN_ROOT:-$DEPLOY_ROOT/external/dopfn}"
export CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
export CKPT="${CKPT:?CKPT required — DoPFN-bb checkpoint path}"
export OUT_ROOT="${OUT_ROOT:?OUT_ROOT required}"
export DENSITY_DUMP="${DENSITY_DUMP:-1}"     # dopfn_bb reads DENSITY_DUMP env, default on here
# EVAL_MAX_CONTEXT convention — mirrors cpfn1d/graph2d. dopfn_bb's script takes
# --n-context instead of the env var, so we translate here.
EVAL_MAX_CONTEXT="${EVAL_MAX_CONTEXT:-0}"

[ -f "$CKPT" ] || { echo "FATAL: ckpt not found: $CKPT" >&2; exit 1; }

case "$SLURM_ARRAY_TASK_ID" in
    0) DATASET=IHDP     ;;
    1) DATASET=ACIC     ;;
    2) DATASET=CPS      ;;
    3) DATASET=PSID     ;;
    4) DATASET=PSIDbal  ;;
    *) echo "invalid task id: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac
export DATASET
export OUT="$OUT_ROOT/$DATASET"
mkdir -p "$OUT"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DoPFN-bb  DATASET=$DATASET  OUT=$OUT"
python -u "$REPO/benchmarks/l2_ihdp/eval_dopfn_bb_raw.py" \
    --repo       "$REPO" \
    --dopfn      "$DOPFN_ROOT" \
    --causalpfn  "$CAUSALPFN" \
    --checkpoint "$CKPT" \
    --dataset    "$DATASET" \
    --n-context  "$EVAL_MAX_CONTEXT" \
    --out        "$OUT/summary.npz"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] done"
