#!/bin/bash
#SBATCH --job-name=uwyk-D-j32nobin
#SBATCH --output=logs_uwyk_ablation/D_j32_nobin_%j.out
#SBATCH --error=logs_uwyk_ablation/D_j32_nobin_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:4
#SBATCH --requeue
#SBATCH --signal=B:TERM@300

# ── UWYK ABLATION D: J=32 bars, NO binarization ──────────────────────────────
#
# One of four ablations on UWYK's own code. D is B's counterpart: same 32-bar
# BarDistribution, but binarize_treatment_prob left at the upstream default of
# 0.0. B minus D isolates what binarizing the treatment does at J=32; A minus D
# isolates what the bar count does without binarization in play.
#
#   A_base        BINARIZE=1.0                  (1000 bars, binarized)
#   B_j32         BINARIZE=1.0  NUM_BARS=32
#   C_botharms    BINARIZE=1.0  BOTH_ARMS=1
#   D_j32_nobin                 NUM_BARS=32     <- THIS FILE
#
# NOTE: `#SBATCH --gres`/`--time` below are KILLARNEY defaults
# (gpubase_h100_b4, 3 days). Command-line flags override file directives, so
# for another cluster just pass them:
#
#   Killarney  sbatch --account=aip-rgrosse --partition=gpubase_h100_b4 <this>
#   Fir        sbatch --account=def-rgrosse_gpu --partition=gpubase_h100_b4 <this>
#   tamIA      sbatch --account=aip-rgrosse --partition=gpubase_bynode_b3 \
#                     --time=1-00:00:00 <this>       (24h cap; resume handles it)
#
# PREREQUISITES -- all four patches must be installed against $UWYK_ROOT:
#   rpfn_patches/uwyk_pickle_shim      workers can spawn at all
#   rpfn_patches/uwyk_mp_start_method  fork, not spawn (spawned workers segfault)
#   rpfn_patches/uwyk_log_every        a Step line every 50, not every 5000
#   rpfn_patches/uwyk_resume           continue after a walltime requeue
# and `wandb` must be importable (run.py imports it even with WANDB_MODE=offline).
set -euo pipefail

DEPLOY_ROOT="${DEPLOY_ROOT:-$SCRATCH/rpfn_bench_kit}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
export UWYK_ROOT="${UWYK_ROOT:-$DEPLOY_ROOT/external/g4cfm}"

# Fail here rather than 26s into a GPU allocation.
[ -d "$UWYK_ROOT" ] || { echo "FATAL: UWYK_ROOT not found: $UWYK_ROOT" >&2; exit 1; }
source "${VENV:-$DEPLOY_ROOT/venv}/bin/activate"
python -c "import wandb" 2>/dev/null || {
    echo "FATAL: wandb not importable. run.py imports it unconditionally," >&2
    echo "       even with WANDB_MODE=offline. Fix: pip install --no-index wandb" >&2
    exit 1; }
[ -f "$UWYK_ROOT/src/utils_module.py" ] || {
    echo "FATAL: pickle shim missing. Run:" >&2
    echo "  UWYK_ROOT=$UWYK_ROOT bash $REPO/rpfn_patches/uwyk_pickle_shim/install.sh" >&2
    exit 1; }

exec env \
    TAG=D_j32_nobin \
    NUM_BARS=32 \
    NUM_WORKERS="${NUM_WORKERS:-16}" \
    BATCH_SIZE="${BATCH_SIZE:-4}" \
    ACCUM="${ACCUM:-8}" \
    UWYK_ROOT="$UWYK_ROOT" \
    DEPLOY_ROOT="$DEPLOY_ROOT" \
    bash "$REPO/training_uwyk_variants/submit_train_uwyk_ablation.sbatch"
