#!/bin/bash
#SBATCH --job-name=uwyk-C-botharms
#SBATCH --output=logs_uwyk_ablation/C_botharms_%j.out
#SBATCH --error=logs_uwyk_ablation/C_botharms_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:4
#SBATCH --requeue
#SBATCH --signal=B:TERM@300

# ── UWYK ABLATION C: both arms queried per covariate ─────────────────────────
#
# Upstream draws ONE arm per query row at random. C emits BOTH arms for each
# query x, so the model sees y0 and y1 under identical covariates and identical
# noise -- the paired-arm signal our 2D head is trained on. This is the
# ablation that asks whether the gain comes from the joint head or merely from
# having seen both arms.
#
#   A_base        BINARIZE=1.0                  (1000 bars, binarized)
#   B_j32         BINARIZE=1.0  NUM_BARS=32
#   C_botharms    BINARIZE=1.0  BOTH_ARMS=1     <- THIS FILE
#   D_j32_nobin                 NUM_BARS=32
#
# SLOWER THAN THE OTHERS: doubling query rows roughly doubles per-step cost.
# A_base took 22:42:02 for 50k steps, so budget well over a day. On a cluster
# with a 24h cap (tamIA) this one WILL requeue -- which is exactly what
# rpfn_patches/uwyk_resume exists for.
#
# `#SBATCH --gres`/`--time` are KILLARNEY defaults. Override per cluster:
#   Killarney  sbatch --account=aip-rgrosse --partition=gpubase_h100_b4 <this>
#   Fir        sbatch --account=def-rgrosse_gpu --partition=gpubase_h100_b4 <this>
#   tamIA      sbatch --account=aip-rgrosse --partition=gpubase_bynode_b3 \
#                     --time=1-00:00:00 <this>
#
# PREREQUISITES -- the four shared patches PLUS the both-arms code patch:
#   rpfn_patches/uwyk_pickle_shim
#   rpfn_patches/uwyk_mp_start_method
#   rpfn_patches/uwyk_log_every
#   rpfn_patches/uwyk_resume
#   rpfn_patches/uwyk_both_arms        <- C ONLY; without it BOTH_ARMS=1 is
#                                         silently ignored and you get a
#                                         duplicate of A, not an ablation.
set -euo pipefail

DEPLOY_ROOT="${DEPLOY_ROOT:-$SCRATCH/rpfn_bench_kit}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
export UWYK_ROOT="${UWYK_ROOT:-$DEPLOY_ROOT/external/g4cfm}"

[ -d "$UWYK_ROOT" ] || { echo "FATAL: UWYK_ROOT not found: $UWYK_ROOT" >&2; exit 1; }
source "${VENV:-$DEPLOY_ROOT/venv}/bin/activate"
python -c "import wandb" 2>/dev/null || {
    echo "FATAL: wandb not importable. Fix: pip install --no-index wandb" >&2; exit 1; }
[ -f "$UWYK_ROOT/src/utils_module.py" ] || {
    echo "FATAL: pickle shim missing. Run:" >&2
    echo "  UWYK_ROOT=$UWYK_ROOT bash $REPO/rpfn_patches/uwyk_pickle_shim/install.sh" >&2
    exit 1; }

# The one check that matters for C: BOTH_ARMS=1 against unpatched code is a
# silent no-op, and you would not find out until you compared C to A and saw
# identical curves 23 hours later.
grep -rqs "UWYK_BOTH_ARMS" "$UWYK_ROOT/src/priordata_processing/" || {
    echo "FATAL: both-arms patch NOT installed -- BOTH_ARMS=1 would be ignored" >&2
    echo "       and this run would silently duplicate A_base. Run:" >&2
    echo "  UWYK_ROOT=$UWYK_ROOT bash $REPO/rpfn_patches/uwyk_both_arms/install.sh" >&2
    exit 1; }

exec env \
    TAG=C_botharms \
    BINARIZE=1.0 \
    BOTH_ARMS=1 \
    NUM_WORKERS="${NUM_WORKERS:-16}" \
    BATCH_SIZE="${BATCH_SIZE:-4}" \
    ACCUM="${ACCUM:-8}" \
    UWYK_ROOT="$UWYK_ROOT" \
    DEPLOY_ROOT="$DEPLOY_ROOT" \
    bash "$REPO/training_uwyk_variants/submit_train_uwyk_ablation.sbatch"
