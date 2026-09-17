#!/bin/bash
#SBATCH --job-name=cpfn2d-j32-eta0
#SBATCH --output=logs_cpfn2d_j32_sharednoise_y01/train_%j.out
#SBATCH --error=logs_cpfn2d_j32_sharednoise_y01/train_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpubase_h100_b1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120

# SHARED ARM NOISE + NOISY TARGETS -- ablation sibling of j32_random.
#
# Identical to submit_train_cpfn2d_j32_random.sbatch in EVERY respect --
# same TabDPT warmstart, same 2D-head hyperparams (J=32, hlgauss, sigma=0.32,
# pooled_std, num_features=99), same trainer / optimizer / DDP / autocast /
# callbacks / step-checkpointing, same hydra overrides, same resources --
# except for two environment variables read by the patched trainer:
#
#   CPFN_SHARED_ARM_NOISE=1   eta_1 := eta_0, so the two arms share one noise
#                             draw (rho = 1) instead of the prior's two
#                             independent draws (rho = 0).
#   CPFN_TRAIN_ON_Y01=1       supervise the 2D joint head on the NOISY
#                             potential outcomes y0, y1 rather than the
#                             conditional means E_y0, E_y1.
#
# Diff against the baseline is exactly:
#   diff submit_train_cpfn2d_j32_random.sbatch \
#        submit_train_cpfn2d_j32_sharednoise_y01.sbatch
# (job name, log dir, OUT_DIR, this comment, and the two exports).
#
# WHAT THIS TESTS. Stock CausalPFN trains the joint head on (E_y0, E_y1), which
# carry no arm noise at all, so the DGP's arm coupling never enters the
# objective -- the head's implied rho reflects its prior, not the data. Here
# the targets are noisy AND perfectly coupled, so the pairs (y0, y1) lie on a
# line of slope 1 offset by tau(x) and the head must represent a
# near-degenerate, strongly correlated joint.
#
# WHAT TO READ OFF IT. The primary readout is the head's IMPLIED rho
# (benchmarks/diag_joint_dependence.py), not the interval width. Note that
# y1 - y0 = E_y1 - E_y0 exactly under eta_1 = eta_0, so the tau information in
# the targets is unchanged from the baseline and a perfectly-fit head would
# give the same tau marginal. If rho moves toward 1 while length holds, the
# head CAN express coupling and the current width is epistemic; if rho stays
# near 0 even against a perfectly diagonal target, the head cannot represent
# correlation at all, which is the stronger architectural finding.
#
# Runs through CausalPFN's OWN train.py + trainer, so the training loop,
# DataLoader, optimizer (schedulefree AdamW), DDP, autocast, callbacks, and
# step-based checkpointing are BYTE-IDENTICAL to the 1D run. The only model
# difference is model.obj._target_ = rpfn_patches.cpfn2d_loader.
# load_cpfn2d_from_tabdpt (TabDPT backbone warmstart + our 2D joint head).
#
# Trainer config: originals (max_epochs=2048, num_model_updates=128 -> 262 144
# optimizer steps). Step-based ckpts every 2000 successful steps at
# $OUT_DIR/step_checkpoints/<run>/step_XXXXXXX.pt.
#
# Requires step-ckpt + train.py patches installed via
# rpfn_patches/causalpfn_step_ckpt/install.sh (called automatically below).
# That same install.sh carries the CPFN_SHARED_ARM_NOISE / CPFN_TRAIN_ON_Y01
# support; both knobs default OFF, so installing it does not disturb any
# existing run.

set -euo pipefail
mkdir -p logs_cpfn2d_j32_sharednoise_y01

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

OUT_DIR="${OUT_DIR:-$DEPLOY_ROOT/cpfn2d_j32_sharednoise_y01_output}"
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
# ── the ONLY behavioural difference from j32_random ──────────────────────────
# eta_1 := eta_0 (arms share one noise draw) and supervise on y0/y1 instead of
# E_y0/E_y1. Read by calculate_loss() in the patched trainer.py; both default
# to "0", so omitting them reproduces j32_random exactly.
export CPFN_SHARED_ARM_NOISE="${CPFN_SHARED_ARM_NOISE:-1}"
export CPFN_TRAIN_ON_Y01="${CPFN_TRAIN_ON_Y01:-1}"
# Compile is ON (their default) — matches paper's ~2-day H100 wall-clock.
# If compile crashes on this cluster's PyTorch build, add back:
#   export TORCHDYNAMO_DISABLE=1
#   export TORCH_COMPILE_DISABLE=1
# and override compile=false on the CLI.
export HF_HOME="${HF_HOME:-$DEPLOY_ROOT/hf_cache}"
# rpfn_patches (loader + cpfn2d_loader) + our repo (losses, training_causalpfn2d)
# must be importable. CausalPFN src also on the path.
export PYTHONPATH="$DEPLOY_ROOT/R-PFN/benchmarks/uwyk_table1/shims:$CAUSALPFN/src:$CAUSALPFN:$DEPLOY_ROOT/R-PFN${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] cpfn2d j32_sharednoise_y01  J=$J  σ_hl=$HLGAUSS_SIGMA  loss=$LOSS_TYPE  y_scale=$Y_SCALING_MODE"
echo "  CAUSALPFN=$CAUSALPFN"
echo "  OUT_DIR=$OUT_DIR"
echo "  CPFN_SHARED_ARM_NOISE=$CPFN_SHARED_ARM_NOISE  CPFN_TRAIN_ON_Y01=$CPFN_TRAIN_ON_Y01"

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
# Keep only the N most recent step_*.pt; older ones are deleted as each new one
# lands. LAST-N, not best-N: the Checkpoint callback already keeps a
# best-by-loss set (top_k), and what a requeued job needs is the most recent
# state. 0 = keep everything (the old behaviour, which filled the disk).
STEP_CKPT_KEEP="${STEP_CKPT_KEEP:-3}"
mkdir -p "$STEP_CKPT_ROOT"

# COMPILE=0 → append `compile=false` to hydra CLI (disables torch.compile).
# Use when torch inductor hits a bug on this pytorch build.
COMPILE="${COMPILE:-1}"
COMPILE_ARGS=()
if [ "$COMPILE" = "0" ]; then
    COMPILE_ARGS+=(compile=false)
    echo "[compile] disabled via COMPILE=0"
fi

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
    callbacks.checkpoint.checkpoint_dir_name=run \
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
    +trainer.step_checkpoint_keep="$STEP_CKPT_KEEP" \
    "${COMPILE_ARGS[@]}" \
    "${RESUME_ARGS[@]}" &
CHILD=$!
wait "$CHILD"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE"
