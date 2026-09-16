#!/bin/bash
#SBATCH --job-name=cpfn1d-j1024-botharms
#SBATCH --output=logs_cpfn1d_j1024_botharms/train_%j.out
#SBATCH --error=logs_cpfn1d_j1024_botharms/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120

# cpfn1d random-init at native J=1024, supervising BOTH arms of every query
# point instead of one coin-flipped arm.
#
# WHY. Upstream cepo_losses draws one treatment per query point:
#     random_treatments = torch.randint(0, 2, E_y0_query.shape, ...)
#     Ey_target = where(random_treatments == 0, E_y0_std, E_y1_std)
# so a step supervises N_q outcome values drawn from N_q covariates, and each
# point's other potential outcome is thrown away. cpfn2d supervises BOTH arms
# of all N_q points -- 2*N_q values from the same N_q covariates. At equal step
# counts the 1D head therefore sees half the outcome supervision, which
# confounds every 1D-vs-2D comparison with a data-quantity difference.
#
# This run sets CPFN_QUERY_BOTH_ARMS=1 (rpfn_patches/causalpfn_query_both_arms),
# which emits each query point twice -- t=0 targeting E_y0, t=1 targeting E_y1.
# Per step: same N_q covariates, 2*N_q query rows, 2*N_q outcome values. That is
# exactly what cpfn2d sees, so the two match at the SAME step count and 50k here
# is comparable to 50k there. Cost is ~2x per-step compute on the query half of
# the sequence; the loss is a mean over query rows, so its scale is unchanged.
#
# Everything else matches causalpfn_j1024_headrand: same trainer, same prior,
# same per-arm y-scaling, J=1024, sigma=0.01, head_reinit=True.
#
# COMPILE defaults to 0: torch.compile is broken in the current venv
# (ModuleNotFoundError: torch._thread_safe_fork, job 5464647, 2026-09-15).
# It affects speed, not results.
#
# A1 STEP-BASED TRAJECTORY MATCH:
# - max_epochs=2048, num_model_updates=128 → 262 144 optimizer steps (originals).
# - Step-based ckpts every 2000 successful optimizer steps at
#   $OUT_DIR/step_checkpoints/step_XXXXXXX.pt, alongside per-epoch
#   Checkpoint callback (still enabled for top_k tracking).
# - Requires the step-ckpt patch installed via
#   rpfn_patches/causalpfn_step_ckpt/install.sh (see script for details).

set -euo pipefail
mkdir -p logs_cpfn1d_j1024_botharms

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

OUT_DIR="${OUT_DIR:-$DEPLOY_ROOT/cpfn1d_j1024_botharms_output}"
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

echo "[$(date '+%Y-%m-%d %H:%M:%S')] cpfn1d J=1024 botharms  σ=$SIGMA  head_reinit=True"
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

# Extra hydra overrides, space separated. The both-arms run doubles the query
# rows, so the (B, 2*N_q, nbins) logits tensor in _hl_gaussian_cross_entropy_loss
# doubles too; if that OOMs, halve the batch and double the accumulation to keep
# the effective batch identical:
#   EXTRA_HYDRA="data.batch_size=<half> trainer.num_agg=<double>" sbatch ...
EXTRA_HYDRA="${EXTRA_HYDRA:-}"
read -r -a EXTRA_HYDRA_ARGS <<< "$EXTRA_HYDRA"

STEP_CKPT_EVERY="${STEP_CKPT_EVERY:-2000}"
STEP_CKPT_ROOT="${STEP_CKPT_ROOT:-$OUT_DIR/step_checkpoints}"
# Keep only the N most recent step_*.pt; older ones are deleted as each new one
# lands. LAST-N, not best-N: the Checkpoint callback already keeps a
# best-by-loss set (top_k), and what a requeued job needs is the most recent
# state. 0 = keep everything (the old behaviour, which filled the disk).
STEP_CKPT_KEEP="${STEP_CKPT_KEEP:-3}"
# Hard stop at 50k (trainer patch). max_epochs bounds EPOCHS only, and
# 50000/num_model_updates is not an integer, so the epoch ceiling cannot express
# this target. Final retained set: 46000 / 48000 / 50000.
MAX_STEP="${MAX_STEP:-50000}"

# Both-arms query expansion. Requires
#   bash rpfn_patches/causalpfn_query_both_arms/install.sh
# to have been run against $CAUSALPFN. The patch is opt-in: without this export
# the installed file reproduces upstream's coin flip exactly, so installing it
# does not disturb any other run.
export CPFN_QUERY_BOTH_ARMS=1
echo "  CPFN_QUERY_BOTH_ARMS=$CPFN_QUERY_BOTH_ARMS  (2 query rows per covariate)"
echo "  MAX_STEP=$MAX_STEP  STEP_CKPT_EVERY=$STEP_CKPT_EVERY  STEP_CKPT_KEEP=$STEP_CKPT_KEEP"
mkdir -p "$STEP_CKPT_ROOT"

# COMPILE=0 → append `compile=false` to hydra CLI (disables torch.compile).
# Use when torch inductor hits a bug on this pytorch build (see paper 2-day
# H100 estimate assumes compile=true).
COMPILE="${COMPILE:-0}"
COMPILE_ARGS=()
if [ "$COMPILE" = "0" ]; then
    COMPILE_ARGS+=(compile=false)
    echo "[compile] disabled via COMPILE=0"
fi

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
    callbacks.checkpoint.checkpoint_dir_name=run \
    callbacks.checkpoint.frequency=1 \
    callbacks.checkpoint.top_k=10000 \
    ~callbacks.eval_cate \
    model.obj._target_=rpfn_patches.loader_with_nbins.load_pretrained_in_context_model_with_nbins \
    +model.obj.head_reinit=true \
    sigma="$SIGMA" \
    +trainer.step_checkpoint_every="$STEP_CKPT_EVERY" \
    +trainer.step_checkpoint_root="$STEP_CKPT_ROOT" \
    +trainer.step_checkpoint_keep="$STEP_CKPT_KEEP" \
    +trainer.max_actual_step="$MAX_STEP" \
    "${COMPILE_ARGS[@]}" \
    "${EXTRA_HYDRA_ARGS[@]}" \
    "${RESUME_ARGS[@]}" &
CHILD=$!
wait "$CHILD"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE"
