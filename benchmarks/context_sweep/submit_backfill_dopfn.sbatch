#!/bin/bash
#SBATCH --job-name=rpfn-backfill-dopfn
#SBATCH --output=logs_backfill/task_%A_%a.out
#SBATCH --error=logs_backfill/task_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-19%20
# ─────────────────────────────────────────────────────────────────────────────
# Backfill Do-PFN predictions into existing context-sweep npz files.
# Same interleaved-stripe sharding as backfill_uwyk.
#
# Usage on killarney:
#   sbatch --account=aip-rgrosse --export=ALL,SOURCES_FILTER=poly \
#          R-PFN/benchmarks/context_sweep/submit_backfill_dopfn.sbatch
#
# Env overrides:
#   DEPLOY_ROOT — root that contains R-PFN/ external/ venv/ results_sweep/
#   SOURCES_FILTER — "prior" | "poly"    (required — one at a time)
#   N_SHARDS       — must match array size (default 20 → --array=0-19)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
mkdir -p logs_backfill

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
if [ ! -d "$REPO/benchmarks" ]; then REPO="$DEPLOY_ROOT"; fi
EXTERNAL_ROOT="${EXTERNAL_ROOT:-$DEPLOY_ROOT/external}"
[ ! -d "$EXTERNAL_ROOT" ] && [ -d "$REPO/external" ] && EXTERNAL_ROOT="$REPO/external"
DOPFN="${DOPFN:-$EXTERNAL_ROOT/dopfn}"
UWYK_SRC="${UWYK_SRC:-$EXTERNAL_ROOT/uwyk/src}"
CAUSALPFN="${CAUSALPFN:-$EXTERNAL_ROOT/causalpfn}"
RESULTS_DIR="${RESULTS_DIR:-$DEPLOY_ROOT/results_sweep}"
VENV="${VENV:-$DEPLOY_ROOT/.venv}"
[ ! -f "$VENV/bin/activate" ] && [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && VENV="$DEPLOY_ROOT/venv"

SRC_FILTER="${SOURCES_FILTER:-poly}"
N_SHARDS="${N_SHARDS:-20}"
SHARD_IDX="${SLURM_ARRAY_TASK_ID:-0}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] shard=$SHARD_IDX/$N_SHARDS  source=$SRC_FILTER"
echo "  REPO=$REPO"
echo "  DOPFN=$DOPFN"
echo "  RESULTS_DIR=$RESULTS_DIR"

source "$VENV/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

python "$REPO/benchmarks/context_sweep/backfill_dopfn.py" \
    --results-dir  "$RESULTS_DIR" \
    --repo         "$REPO" \
    --dopfn        "$DOPFN" \
    --uwyk-src     "$UWYK_SRC" \
    --causalpfn    "$CAUSALPFN" \
    --source       "$SRC_FILTER" \
    --shard-idx    "$SHARD_IDX" \
    --n-shards     "$N_SHARDS"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE shard=$SHARD_IDX"
