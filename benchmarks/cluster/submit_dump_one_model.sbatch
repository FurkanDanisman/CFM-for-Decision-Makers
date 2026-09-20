#!/bin/bash
#SBATCH --job-name=dump1
#SBATCH --output=logs_dump1/d1_%j.out
#SBATCH --error=logs_dump1/d1_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#
# ONE job = ONE model, all of one benchmark. No array.
#
# The existing dump sbatches index a MODELS array off SLURM_ARRAY_TASK_ID, which
# makes a model x dataset grid of array tasks. Rather than duplicate their
# dispatch (and risk it drifting), this loops the inner task indices and invokes
# the same script as plain bash -- the #SBATCH lines are comments, so `bash
# <sbatch>` runs the body directly with whatever SLURM_ARRAY_TASK_ID we set. One
# queue entry per model, every dataset done sequentially inside it.
#
# Required:
#   MODEL_NAME   label for the output root, e.g. cpfn1d_j32
#   MODEL_IDX    index into the inner MODELS array:
#                  0 dopfn_native  1 dopfn_bb  2 uwyk1d
#                  3 graph2d       4 cpfn1d    5 cpfn2d_pooled
#   CKPT_ENV     env var the harness reads, e.g. CKPT_CPFN1D / DOPFN_CKPT /
#                CKPT_DOPFN_BB / CKPT_GRAPH2D
#   CKPT         path to the weights
#   BENCH        rc | cs
# Optional:
#   OUT_ROOT     default $SCRATCH/dumps_all/<MODEL_NAME>/<BENCH>
#   CELLS        subset of inner indices (default: all 5 datasets / all 8 d)
#   SHIFT        case studies only (default 0)
#
#   MODEL_NAME=cpfn1d_j32 MODEL_IDX=4 CKPT_ENV=CKPT_CPFN1D \
#   CKPT=$REPO/Required_checkpoints/cpfn1d_j32_step50000.pt BENCH=rc \
#     sbatch benchmarks/cluster/submit_dump_one_model.sbatch

set -uo pipefail
mkdir -p logs_dump1

: "${MODEL_NAME:?MODEL_NAME required}"
: "${MODEL_IDX:?MODEL_IDX required}"
: "${CKPT_ENV:?CKPT_ENV required}"
: "${CKPT:?CKPT required}"
: "${BENCH:?BENCH required (rc|cs)}"

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
OUT_ROOT="${OUT_ROOT:-$SCRATCH/dumps_all/$MODEL_NAME/$BENCH}"
[ -f "$CKPT" ] || { echo "FATAL: no checkpoint at $CKPT" >&2; exit 1; }

case "$BENCH" in
  rc) INNER="$REPO/benchmarks/cluster/submit_realcause_density_unified.sbatch"
      STRIDE=5; DEFAULT_CELLS="0 1 2 3 4"          # IHDP ACIC CPS PSID PSID_bal
      LABELS=(IHDP ACIC CPS PSID PSID_bal) ;;
  cs) INNER="$REPO/benchmarks/cluster/submit_cs_dvar_density.sbatch"
      STRIDE=8; DEFAULT_CELLS="0 1 2 3 4 5 6 7"    # d = 2 3 5 10 20 30 40 50
      LABELS=(d2 d3 d5 d10 d20 d30 d40 d50) ;;
  *)  echo "FATAL: BENCH must be rc or cs, got '$BENCH'" >&2; exit 1 ;;
esac
CELLS="${CELLS:-$DEFAULT_CELLS}"

echo "[$(date '+%F %T')] model=$MODEL_NAME idx=$MODEL_IDX bench=$BENCH"
echo "  $CKPT_ENV=$CKPT"
echo "  OUT_ROOT=$OUT_ROOT"
echo "  cells=$CELLS  (stride $STRIDE)"
echo

FAILED=0
for c in $CELLS; do
    task=$(( MODEL_IDX * STRIDE + c ))
    label="${LABELS[$c]}"
    echo "===== [$(date '+%F %T')] $MODEL_NAME / $label   (inner task $task)"
    # New process per cell, so each gets a clean environment and one bad cell
    # cannot leave exported state behind for the next.
    SLURM_ARRAY_TASK_ID="$task" \
    env "$CKPT_ENV=$CKPT" OUT_ROOT="$OUT_ROOT" DENSITY_DUMP=1 \
        SHIFT="${SHIFT:-0}" bash "$INNER"
    st=$?
    if [ "$st" != 0 ]; then echo "  CELL FAILED $label (exit $st)"; FAILED=$((FAILED+1)); fi
done

n_npz=$(find "$OUT_ROOT" -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
echo
echo "[$(date '+%F %T')] $MODEL_NAME/$BENCH done: failed_cells=$FAILED  npz_written=$n_npz"
[ "$n_npz" = 0 ] && { echo "ERROR: no dumps written at all" >&2; exit 1; }
exit 0
