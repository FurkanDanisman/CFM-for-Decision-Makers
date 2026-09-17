#!/bin/bash
#SBATCH --job-name=dopfn-repro-1d
#SBATCH --output=logs_dopfn_repro_1d/train_%j.out
#SBATCH --error=logs_dopfn_repro_1d/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120

# Do-PFN reproduction, 1-D bar-distribution head. Task 1 of training_dopfn_repro.
#
#   python training_dopfn_repro/train.py --variant dopfn_1d --steps 150000
#
# ── CLUSTER NOTES (Fir) ──────────────────────────────────────────────────────
# --gres=gpu:h100:1 is MANDATORY here: Fir rejects a GPU job that does not name
# a model ("Specifying a GPU model is now required"). h100 also selects the
# gpubase_bygpu_* partitions, which allocate PER GPU -- unlike tamIA, where
# gpu:h100:4 whole-node is the only option and queues far worse.
#
# --partition is deliberately NOT set. Fir routes by --time into buckets
#     b1 3h | b2 12h | b3 24h | b4 3d | b5 7d
# and pinning one only creates the "Requested time limit is invalid" failure we
# hit repeatedly on Killarney (its h100_b1 caps at 3h).
#
# --account must be given at submit time; there is no usable default:
#     sbatch --account=def-rgrosse_gpu ...        (better fairshare)
#     sbatch --account=def-zhijing_gpu ...        (use for the other variant so
#                                                  the two do not queue behind
#                                                  each other in one pool)
#
# ── 150k STEPS WILL NOT FIT ONE ALLOCATION ───────────────────────────────────
# Step rate is unmeasured, so the run is built to survive the wall rather than
# to fit inside it: --requeue plus a SIGTERM trap that requeues, and --resume
# passed unconditionally. train.py writes <OUT>/<variant>/latest.pt every
# --ckpt-every steps and reloads it on --resume, so each requeue continues
# instead of restarting. --resume is a no-op on the first run.
#
# Once you know the rate (the log prints s/step), shrink --time to the smallest
# bucket that still makes progress between requeues: smaller buckets schedule
# sooner, and the only cost of a requeue is re-fitting the bar distribution at
# startup.
#
# Env:
#   DEPLOY_ROOT  default $SCRATCH/rpfn_bench_kit
#   REPO         default $DEPLOY_ROOT/R-PFN
#   VENV         default $DEPLOY_ROOT/venv
#   DOPFN_SRC    Do-PFN checkout holding artifacts/  (required by prior.py)
#   OUT          checkpoint root (default $DEPLOY_ROOT/checkpoints_dopfn_repro)
#   STEPS        default 150000
#   EXTRA        extra args appended to train.py
set -euo pipefail
mkdir -p logs_dopfn_repro_1d

VARIANT=dopfn_1d
DEPLOY_ROOT="${DEPLOY_ROOT:-$SCRATCH/rpfn_bench_kit}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
VENV="${VENV:-$DEPLOY_ROOT/venv}"
OUT="${OUT:-$DEPLOY_ROOT/checkpoints_dopfn_repro}"
STEPS="${STEPS:-150000}"
export DOPFN_SRC="${DOPFN_SRC:-$DEPLOY_ROOT/external/dopfn}"
EXTRA="${EXTRA:-}"
read -r -a EXTRA_ARGS <<< "$EXTRA"

# ── preflight: fail in seconds, not after the queue wait ─────────────────────
[ -f "$VENV/bin/activate" ]          || { echo "FATAL: no venv at $VENV" >&2; exit 1; }
[ -f "$REPO/training_dopfn_repro/train.py" ] || { echo "FATAL: no train.py under $REPO" >&2; exit 1; }
# prior.py loads these from inside DOPFN_SRC with cwd changed there; a missing
# artifact otherwise surfaces as an opaque unpickling error mid-run.
for f in artifacts/dopfn_model.pkl \
         artifacts/model_submitit_0ccc_id_171b69db_epoch_-1.cpkt; do
    [ -f "$DOPFN_SRC/$f" ] || { echo "FATAL: missing $DOPFN_SRC/$f" >&2; exit 1; }
done

source "$VENV/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] dopfn-repro variant=$VARIANT steps=$STEPS"
echo "  REPO=$REPO"
echo "  DOPFN_SRC=$DOPFN_SRC"
echo "  OUT=$OUT/$VARIANT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
LATEST="$OUT/$VARIANT/latest.pt"
[ -f "$LATEST" ] && echo "  found $LATEST -> resuming" || echo "  no latest.pt -> fresh start"

# Forward SIGTERM to the child so train.py can finish its step, then requeue.
CHILD=""
_on_term() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SIGTERM -> child $CHILD, then requeue"
    [ -n "$CHILD" ] && kill -TERM "$CHILD" 2>/dev/null || true
    wait "$CHILD" 2>/dev/null || true
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}
trap _on_term SIGTERM

cd "$REPO"
python -u training_dopfn_repro/train.py \
    --variant "$VARIANT" \
    --steps "$STEPS" \
    --out "$OUT" \
    --resume \
    "${EXTRA_ARGS[@]}" &
CHILD=$!
wait "$CHILD"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE"
