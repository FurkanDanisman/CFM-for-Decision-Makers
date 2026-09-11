#!/bin/bash
#SBATCH --job-name=cpfn2d-j1024-from1d
#SBATCH --output=logs_cpfn2d_j1024_from1d/train_%j.out
#SBATCH --error=logs_cpfn2d_j1024_from1d/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpubase_h100_b1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120

# J=1024 SMALL RUN — maximal-head 2D joint variant.
#
# SIZE. The 2D head emits J**2 + 13 logits, so J=1024 is 1,048,589 outputs per
# query -- 1011x the J=32 run (1,037). Concretely, with ninp=256:
#
#   head weight        384 x 1,048,599  = 402.7 M params   ~1.61 GB fp32
#                      (ninp=384 -- the cpfn2d_loader default, NOT the
#                       CausalPFN2DHead class default of 256)
#   + grad + AdamW     ~4x that                            ~6.4-8.0 GB
#   logits activation  4.19 MB PER QUERY per fp32 copy
#                      hlgauss builds a same-sized target, and softmax another,
#                      so budget ~3 copies: ~12.6 MB/query
#
# That last line is the binding constraint, not the weights: at 1000 queries in
# flight it is ~12.6 GB of activation for the head alone. If it OOMs, cut the
# per-step query/batch count first -- shrinking J defeats the purpose.
#
# COMPILE IS OFF BY DEFAULT HERE (COMPILE=${COMPILE:-0}). torch.compile on a
# 1M-class softmax can take pathological time/memory to trace. Set COMPILE=1
# once a step is known to run.
#
# THIS IS THE from-1D VARIANT. Backbone warmstart is identical to the random
# sibling; the difference is the HEAD. Given the trained 1D J=1024 head with
# logits l_i, the joint block is seeded as
#
#     joint_logit(i, j) = l_i + l_j      =>   p(i,j) = p_1d(i) * p_1d(j)
#
# so at step 0 the 2D model reproduces the 1D model's marginals EXACTLY on both
# axes with correlation exactly 0. Training then only has to learn the coupling
# -- the one thing a 1D head cannot represent -- instead of re-learning the
# marginals from scratch over 1,048,576 bins.
#
# The 9 region weights are pinned to region 0 (inner x inner) with zero weight
# rows, giving inner mass 1 - 8*exp(-20) ~ 0.999999984 and leaving the 8 tail
# regions inactive at init, as intended. The 4 tail scale raws are zeroed
# (softplus(0)=0.693 -> s = 0.693 * bin_width): finite and unused while the
# tails carry no mass. All of it is initialisation only; gradients can move it.
#
# Verified exactly (rpfn_patches/cpfn2d_head_from_1d.py self-test): joint ==
# product of marginals to 1e-16, both marginals == the 1D distribution, and
# covariance 0 to 1e-13.
#
# CKPT_1D must be a trained 1D head at the SAME J and the SAME backbone width
# (ninp=384); the loader raises if either disagrees.
#
# SIGMA TRACKS BIN WIDTH. Under y_scaling_mode=pooled_std the grid is
# [-10, +10], so bin width = 20/J and the j32 run's sigma is 0.512 x 20/32 =
# 0.32. At J=1024 bin width = 0.01953 and sigma = 0.512 x 0.01953 = 0.01.
# Leaving sigma at 0.32 would smear each label over ~33 bins on each axis and
# throw away exactly the resolution this run is buying.
#
# Everything else is byte-identical to submit_train_cpfn2d_j32_random.sbatch.
#
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
mkdir -p logs_cpfn2d_j1024_from1d

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

OUT_DIR="${OUT_DIR:-$DEPLOY_ROOT/cpfn2d_j1024_from1d_output}"
mkdir -p "$OUT_DIR"

# 2D-head hyperparams (bake into the CLI overrides below).
J="${J:-1024}"
# trained 1D J=1024 head whose marginals seed the joint
CKPT_1D="${CKPT_1D:-$DEPLOY_ROOT/causalpfn_j1024_headrand_A1_h100/step_checkpoints/run/step_0050000.pt}"
[ -f "$CKPT_1D" ] || { echo "FATAL: CKPT_1D not found: $CKPT_1D" >&2; exit 1; }
# 0.512 x bin_width, bin_width = 20/J = 0.01953125 under pooled_std
HLGAUSS_SIGMA="${HLGAUSS_SIGMA:-0.01}"
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

echo "[$(date '+%Y-%m-%d %H:%M:%S')] cpfn2d j1024_from1d  J=$J  σ_hl=$HLGAUSS_SIGMA  loss=$LOSS_TYPE  y_scale=$Y_SCALING_MODE"
echo "  CAUSALPFN=$CAUSALPFN"
echo "  OUT_DIR=$OUT_DIR"
echo "  CKPT_1D=$CKPT_1D"
python - <<PY
J=$J
n=J*J+9+4
print(f"  head logits = {n:,}  ({n/1037:.0f}x the J=32 head)")
print(f"  head weight = 384 x {n+10:,} = {384*(n+10)/1e6:.1f}M params (~{384*(n+10)*4/1e9:.2f} GB fp32)")
print(f"  logits activation = {n*4/1e6:.2f} MB per query per copy")
PY

CHILD=""
_on_term() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SIGTERM — forwarding to child pid=$CHILD"
    [ -n "$CHILD" ] && kill -TERM "$CHILD" 2>/dev/null || true
    wait "$CHILD" 2>/dev/null || true
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}
trap _on_term SIGTERM

# checkpoint every 500 steps over the 24h slot
STEP_CKPT_EVERY="${STEP_CKPT_EVERY:-500}"
STEP_CKPT_ROOT="${STEP_CKPT_ROOT:-$OUT_DIR/step_checkpoints}"
mkdir -p "$STEP_CKPT_ROOT"

# COMPILE=0 → append `compile=false` to hydra CLI (disables torch.compile).
# Use when torch inductor hits a bug on this pytorch build.
COMPILE="${COMPILE:-0}"
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
    model.obj._target_=rpfn_patches.cpfn2d_head_from_1d.load_cpfn2d_from_1d \
    model.obj.ckpt_path="$WARMSTART_LOCAL" \
    +model.obj.ckpt_1d="$CKPT_1D" \
    +model.obj.J="$J" \
    +model.obj.num_features="$NUM_FEATURES" \
    +model.obj.hlgauss_sigma="$HLGAUSS_SIGMA" \
    +model.obj.loss_type="$LOSS_TYPE" \
    +model.obj.y_scaling_mode="$Y_SCALING_MODE" \
    +trainer.step_checkpoint_every="$STEP_CKPT_EVERY" \
    +trainer.step_checkpoint_root="$STEP_CKPT_ROOT" \
    "${COMPILE_ARGS[@]}" \
    "${RESUME_ARGS[@]}" &
CHILD=$!
wait "$CHILD"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE"
