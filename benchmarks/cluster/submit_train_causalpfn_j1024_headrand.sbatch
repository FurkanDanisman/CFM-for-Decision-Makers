#!/bin/bash
#SBATCH --job-name=cpfn-orig-headrand
#SBATCH --output=logs_causalpfn_original_j1024_headrand/train_%j.out
#SBATCH --error=logs_causalpfn_original_j1024_headrand/train_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpubase_h100_b1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120

# Exp 3: CausalPFN's own trainer at their native J=1024, σ=0.01, but with
# the head re-initialized to random after backbone warmstart. Isolates
# the head-warmstart contribution for their own 1D architecture.
#
# Uses rpfn_patches.loader_with_nbins.load_pretrained_in_context_model_with_nbins
# with head_reinit=True (no nbins_override → keeps 1024, re-inits head).
#
# A1 STEP-BASED TRAJECTORY MATCH:
# - max_epochs=2048, num_model_updates=128 → 262 144 optimizer steps (originals).
# - Step-based ckpts every 2000 successful optimizer steps at
#   $OUT_DIR/step_checkpoints/step_XXXXXXX.pt, alongside per-epoch
#   Checkpoint callback (still enabled for top_k tracking).
# - Requires the step-ckpt patch installed via
#   rpfn_patches/causalpfn_step_ckpt/install.sh (see script for details).

set -euo pipefail
mkdir -p logs_causalpfn_original_j1024_headrand

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }

CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
[ -f "$CAUSALPFN/train.py" ] || { echo "FATAL: no train.py at $CAUSALPFN" >&2; exit 1; }

# Install step-checkpoint patches into CausalPFN deploy tree (idempotent).
PATCH_DIR="${PATCH_DIR:-$DEPLOY_ROOT/R-PFN/rpfn_patches/causalpfn_step_ckpt}"
if [ -x "$PATCH_DIR/install.sh" ]; then
    DEPLOY_ROOT="$DEPLOY_ROOT" CAUSALPFN="$CAUSALPFN" "$PATCH_DIR/install.sh"
else
    echo "FATAL: step-ckpt install script missing at $PATCH_DIR/install.sh" >&2
    exit 1
fi

WARMSTART_LOCAL="${WARMSTART_LOCAL:-$DEPLOY_ROOT/warmstart/tabdpt_long_context.ckpt}"
[ -f "$WARMSTART_LOCAL" ] || {
    echo "FATAL: warmstart ckpt not found at $WARMSTART_LOCAL" >&2; exit 1;
}

OUT_DIR="${OUT_DIR:-$DEPLOY_ROOT/causalpfn_j1024_headrand_output}"
mkdir -p "$OUT_DIR"

SIGMA="${SIGMA:-0.01}"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# Compile is ON (their default) — matches paper's ~2-day H100 wall-clock.
# If compile crashes on this cluster's PyTorch build, add back:
#   export TORCHDYNAMO_DISABLE=1
#   export TORCH_COMPILE_DISABLE=1
# and override compile=false on the CLI.
export HF_HOME="${HF_HOME:-$DEPLOY_ROOT/hf_cache}"
export PYTHONPATH="$DEPLOY_ROOT/R-PFN/benchmarks/uwyk_table1/shims:$CAUSALPFN/src:$CAUSALPFN:$DEPLOY_ROOT/R-PFN${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] CausalPFN Exp 3: J=1024 σ=$SIGMA head_reinit=True"
echo "  CAUSALPFN=$CAUSALPFN"
echo "  OUT_DIR=$OUT_DIR"

CHILD=""
_on_term() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SIGTERM — forwarding to child pid=$CHILD"
    [ -n "$CHILD" ] && kill -TERM "$CHILD" 2>/dev/null || true
    wait "$CHILD" 2>/dev/null || true
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}
trap _on_term SIGTERM

STEP_CKPT_EVERY="${STEP_CKPT_EVERY:-2000}"
STEP_CKPT_ROOT="${STEP_CKPT_ROOT:-$OUT_DIR/step_checkpoints}"
mkdir -p "$STEP_CKPT_ROOT"

# A1 rigor: originals (max_epochs=2048, num_model_updates=128 → 262 144 steps).
# Resume: the checkpoint at latest.pt (per-epoch) carries actual_step so
# trainer picks up the counter exactly.
RESUME_ARGS=()
LATEST_CKPT="$OUT_DIR/checkpoints/latest.pt"
# checkpoint_dir_name is set by the callback the first time; on requeue,
# find any latest.pt under the checkpoints dir.
FOUND_LATEST="$(find "$OUT_DIR/checkpoints" -maxdepth 2 -name latest.pt 2>/dev/null | head -1 || true)"
if [ -n "$FOUND_LATEST" ]; then
    LATEST_CKPT="$FOUND_LATEST"
    RESUME_ARGS+=(resume_training=enabled resume_training.checkpoint_path="$LATEST_CKPT")
    echo "[resume] found $LATEST_CKPT"
fi

cd "$CAUSALPFN"
python -u train.py \
    +experiment=simple_configuration \
    hydra.run.dir="$OUT_DIR/hydra_run" \
    +callbacks@callbacks.checkpoint=checkpoint \
    callbacks.checkpoint.checkpoint_root="$OUT_DIR/checkpoints" \
    callbacks.checkpoint.frequency=1 \
    callbacks.checkpoint.top_k=10000 \
    ~callbacks.eval_cate \
    model.obj._target_=rpfn_patches.loader_with_nbins.load_pretrained_in_context_model_with_nbins \
    +model.obj.head_reinit=true \
    sigma="$SIGMA" \
    +trainer.step_checkpoint_every="$STEP_CKPT_EVERY" \
    +trainer.step_checkpoint_root="$STEP_CKPT_ROOT" \
    "${RESUME_ARGS[@]}" &
CHILD=$!
wait "$CHILD"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE"
