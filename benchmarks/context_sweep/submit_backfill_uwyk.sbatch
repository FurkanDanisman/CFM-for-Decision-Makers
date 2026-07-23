#!/bin/bash
#SBATCH --job-name=rpfn-backfill-uwyk
#SBATCH --output=logs_backfill/task_%A_%a.out
#SBATCH --error=logs_backfill/task_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-19%20
# ─────────────────────────────────────────────────────────────────────────────
# Backfill UWYK No-Ancestral fields into existing sweep npz files.
#
# Each array task processes 1/N_SHARDS of the todo list (interleaved stripe),
# so files finish in balanced time across the array.
#
# Usage on killarney (from your deploy dir, e.g. /scratch/furkanbd/rpfn_bench_kit):
#   sbatch --account=def-rgrosse \
#          --export=ALL,SOURCES_FILTER=poly \
#          R-PFN/benchmarks/context_sweep/submit_backfill_uwyk.sbatch
#
# Env overrides:
#   DEPLOY_ROOT — root that contains R-PFN/ external/ venv/ results_sweep/
#   SOURCES_FILTER — "prior" | "poly"    (required — one at a time)
#   N_SHARDS — must match the array size (default 20 → --array=0-19)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
mkdir -p logs_backfill

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
if [ ! -d "$REPO/benchmarks" ]; then REPO="$DEPLOY_ROOT"; fi
# external/ lives beside R-PFN on killarney (DEPLOY_ROOT/external), or inside
# R-PFN on trillium — probe both, prefer the beside-R-PFN layout.
EXTERNAL_ROOT="${EXTERNAL_ROOT:-$DEPLOY_ROOT/external}"
[ ! -d "$EXTERNAL_ROOT" ] && [ -d "$REPO/external" ] && EXTERNAL_ROOT="$REPO/external"
UWYK_SRC="${UWYK_SRC:-$EXTERNAL_ROOT/uwyk/src}"
UWYK_CKPT_DIR="${UWYK_CKPT_DIR:-$EXTERNAL_ROOT/uwyk/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0}"
CAUSALPFN="${CAUSALPFN:-$EXTERNAL_ROOT/causalpfn}"
RESULTS_DIR="${RESULTS_DIR:-$DEPLOY_ROOT/results_sweep}"
VENV="${VENV:-$DEPLOY_ROOT/.venv}"
[ ! -f "$VENV/bin/activate" ] && [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && VENV="$DEPLOY_ROOT/venv"

SRC_FILTER="${SOURCES_FILTER:-poly}"
N_SHARDS="${N_SHARDS:-20}"
SHARD_IDX="${SLURM_ARRAY_TASK_ID:-0}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] shard=$SHARD_IDX/$N_SHARDS  source=$SRC_FILTER"
echo "  REPO=$REPO"
echo "  RESULTS_DIR=$RESULTS_DIR"
echo "  UWYK_CKPT_DIR=$UWYK_CKPT_DIR"

source "$VENV/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

python "$REPO/benchmarks/context_sweep/backfill_uwyk.py" \
    --results-dir   "$RESULTS_DIR" \
    --repo          "$REPO" \
    --uwyk-src      "$UWYK_SRC" \
    --uwyk-ckpt-dir "$UWYK_CKPT_DIR" \
    --causalpfn     "$CAUSALPFN" \
    --source        "$SRC_FILTER" \
    --shard-idx     "$SHARD_IDX" \
    --n-shards      "$N_SHARDS"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE shard=$SHARD_IDX"
