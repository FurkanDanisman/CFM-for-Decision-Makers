#!/bin/bash
#SBATCH --job-name=cpfn1d-j32-headrand
#SBATCH --output=logs_cpfn1d_j32_headrand/train_%j.out
#SBATCH --error=logs_cpfn1d_j32_headrand/train_%j.err
#SBATCH --time=24:00:00
# Partition deliberately NOT pinned: gpubase_h100_b1 caps at 3h, so pinning it
# makes any --time above that fail at submission with "Requested time limit is
# invalid" (which is exactly what the J=1024 script does -- it asks for 4h on a
# 3h partition). Leaving it unset lets Slurm route to a partition that can take
# the full 24h. The job also requeues on SIGTERM, so a shorter placement still
# resumes from latest.pt.
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120

# cpfn1d random-init at J=32 (instead of CausalPFN's native J=1024).
#
# The bin-count ablation for the 1D head: same trainer, same prior, same
# per-arm y-scaling as causalpfn_j1024_headrand -- ONLY nbins changes. That
# makes it the 1D counterpart of cpfn2d_j32_random, so a 1D-vs-2D comparison
# at J=32 is on one lattice for both heads.
#
# Uses rpfn_patches.loader_with_nbins.load_pretrained_in_context_model_with_nbins
# with nbins_override=32: the backbone stays TabDPT-warmstarted, the final head
# Linear is replaced by a fresh 32-bin one (random init), and bin_edges / width
# / centers are rebuilt on the same [vmin, vmax]. head_reinit is not passed --
# nbins_override already re-inits the head.
#
# SIGMA=0.32, not the J=1024 default of 0.01. sigma is a width in y-units, so
# it has to track the bin width or the soft target collapses to one-hot:
#     J=1024  bin width 20/1024 = 0.0195  ->  sigma 0.01  = 0.51 bins
#     J=32    bin width 20/32   = 0.625   ->  sigma 0.32  = 0.51 bins
# 0.32 is also what cpfn2d_j32 uses, so the two J=32 models match here too.
#
# COMPILE defaults to 0 here, unlike the J=1024 script. torch.compile is broken
# in the current venv -- it dies at import with
#     ModuleNotFoundError: No module named 'torch._thread_safe_fork'
# (confirmed on job 5464647, 2026-09-15). compile affects speed, not results.
# Set COMPILE=1 once the torch install is repaired.
#
# TARGET: step 50000, enforced by +trainer.max_actual_step=$MAX_STEP. The job
# stops itself there, so the last three step checkpoints are 46k / 48k / 50k.
#
# A1 STEP-BASED TRAJECTORY MATCH:
# - max_epochs=2048, num_model_updates=128 → 262 144 optimizer steps (originals).
# - Step-based ckpts every 2000 successful optimizer steps at
#   $OUT_DIR/step_checkpoints/step_XXXXXXX.pt, alongside per-epoch
#   Checkpoint callback (still enabled for top_k tracking).
# - Requires the step-ckpt patch installed via
#   rpfn_patches/causalpfn_step_ckpt/install.sh (see script for details).

set -euo pipefail
mkdir -p logs_cpfn1d_j32_headrand

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

OUT_DIR="${OUT_DIR:-$DEPLOY_ROOT/cpfn1d_j32_headrand_output}"
mkdir -p "$OUT_DIR"

SIGMA="${SIGMA:-0.32}"

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

echo "[$(date '+%Y-%m-%d %H:%M:%S')] cpfn1d J=32 headrand  σ=$SIGMA  nbins_override=32"
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
# Hard stop at step 50000 (trainer patch: +trainer.max_actual_step). max_epochs
# only bounds EPOCHS -- 50000 / num_model_updates=128 is not an integer, so the
# epoch ceiling cannot express this target. The trainer breaks out the moment
# actual_step reaches the limit, and guarantees a checkpoint at exactly that
# step. Final retained set is 46000 / 48000 / 50000.
MAX_STEP="${MAX_STEP:-50000}"
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
    +model.obj.nbins_override=32 \
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
