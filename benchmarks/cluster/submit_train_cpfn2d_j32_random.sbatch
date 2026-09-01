#!/bin/bash
#SBATCH --job-name=cpfn2d-j32-rand
#SBATCH --output=logs_cpfn2d_j32_random/train_%j.out
#SBATCH --error=logs_cpfn2d_j32_random/train_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpubase_h100_b1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120

# A1 TRAJECTORY MATCH — 2D joint head sibling of j1024_headrand.
#
# Runs through CausalPFN's OWN train.py + trainer, so the training loop,
# DataLoader, optimizer (schedulefree AdamW), DDP, autocast, callbacks,
# and step-based checkpointing are BYTE-IDENTICAL to the 1D run. The
# ONLY difference is model.obj._target_:
#   1D headrand → rpfn_patches.loader_with_nbins.load_pretrained_in_context_model_with_nbins
#                 (native TabDPT at J=1024, head re-init)
#   2D j32rand  → rpfn_patches.cpfn2d_loader.load_cpfn2d_from_tabdpt
#                 (TabDPT backbone warmstart + our 2D joint head at J=32)
#
# 2D-head hyperparams (from prior j32_random config):
#   J=32, loss_type=hlgauss, hlgauss_sigma=0.32 (= 0.512 × 20/32 = 0.512 × bin_width),
#   y_scaling_mode=pooled_std, num_features=100
#
# Trainer config: originals (max_epochs=2048, num_model_updates=128 → 262 144
# optimizer steps). Step-based ckpts every 2000 successful steps at
# $OUT_DIR/step_checkpoints/<run>/step_XXXXXXX.pt.
#
# Requires step-ckpt + train.py patches installed via
# rpfn_patches/causalpfn_step_ckpt/install.sh (called automatically below).

set -euo pipefail
mkdir -p logs_cpfn2d_j32_random

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

# Install step-ckpt + train.py patches into CausalPFN deploy tree (idempotent).
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

OUT_DIR="${OUT_DIR:-$DEPLOY_ROOT/cpfn2d_j32_random_output}"
mkdir -p "$OUT_DIR"

# 2D-head hyperparams (bake into the CLI overrides below).
J="${J:-32}"
HLGAUSS_SIGMA="${HLGAUSS_SIGMA:-0.32}"
LOSS_TYPE="${LOSS_TYPE:-hlgauss}"
Y_SCALING_MODE="${Y_SCALING_MODE:-pooled_std}"
# NUM_FEATURES = X-feature count (backbone gets +1 for T column).
# CausalPFN's synthetic_backdoor prior emits X at post_padding_n_cols=99,
# so num_features=99 keeps backbone.num_features=100 = TabDPT ckpt native.
NUM_FEATURES="${NUM_FEATURES:-99}"

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
# rpfn_patches (loader + cpfn2d_loader) + our repo (losses, training_causalpfn2d)
# must be importable. CausalPFN src also on the path.
export PYTHONPATH="$DEPLOY_ROOT/R-PFN/benchmarks/uwyk_table1/shims:$CAUSALPFN/src:$CAUSALPFN:$DEPLOY_ROOT/R-PFN${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] cpfn2d j32_random  J=$J  σ_hl=$HLGAUSS_SIGMA  loss=$LOSS_TYPE  y_scale=$Y_SCALING_MODE"
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

# Resume-on-requeue: latest.pt carries actual_step so counter picks up exactly.
RESUME_ARGS=()
FOUND_LATEST="$(find "$OUT_DIR/checkpoints" -maxdepth 2 -name latest.pt 2>/dev/null | head -1 || true)"
if [ -n "$FOUND_LATEST" ]; then
    RESUME_ARGS+=(resume_training=enabled resume_training.checkpoint_path="$FOUND_LATEST")
    echo "[resume] found $FOUND_LATEST"
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
    model.obj._target_=rpfn_patches.cpfn2d_loader.load_cpfn2d_from_tabdpt \
    model.obj.ckpt_path="$WARMSTART_LOCAL" \
    +model.obj.J="$J" \
    +model.obj.num_features="$NUM_FEATURES" \
    +model.obj.hlgauss_sigma="$HLGAUSS_SIGMA" \
    +model.obj.loss_type="$LOSS_TYPE" \
    +model.obj.y_scaling_mode="$Y_SCALING_MODE" \
    +trainer.step_checkpoint_every="$STEP_CKPT_EVERY" \
    +trainer.step_checkpoint_root="$STEP_CKPT_ROOT" \
    "${RESUME_ARGS[@]}" &
CHILD=$!
wait "$CHILD"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE"
