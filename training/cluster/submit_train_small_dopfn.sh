#!/bin/bash
#SBATCH --account=def-rgrosse
#SBATCH --job-name=cfm-small-dopfn
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/train_small_dopfn_%j.out
#SBATCH --error=logs/train_small_dopfn_%j.err

# Small-scale Do-PFN training smoke run — the debug counterpart to the full
# submit_train.sh. Tiny model (d_model=64, depth=2, heads=4), ~200 steps,
# plain Adam (no cosine / no bf16 / no grad accum). One H100 hour is plenty.
#
# Prereqs on the login node before submitting:
#   1) Do-PFN is at $PROJ_DIR/Do-PFN (or override DOPFN_SRC), containing
#      priors/playground_scm/…
#   2) The venv at $VENV_DIR exists and has: torch (2.9.1+computecanada), numpy,
#      scipy, networkx. See the setup commands in the repo README (uv-based).

set -e
# SLURM stages the submit script to a temp path (e.g. /var/spool/slurm/…),
# so $0 doesn't point at the repo. $SLURM_SUBMIT_DIR is the directory you ran
# `sbatch` from — use that. Fallback to $0/../.. for local `bash` runs.
PROJ_DIR="${PROJ_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"
cd "$PROJ_DIR"
mkdir -p logs

# ── Environment ─────────────────────────────────────────────────────────
# The venv was created against DRAC's python/3.11 (cp311 wheelhouse). Load
# the same module here so the venv's python resolves to the same cvmfs
# interpreter it was seeded with.
module load python/3.11

VENV_DIR="${VENV_DIR:-$PROJ_DIR/.venv}"
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "ERROR: venv not found at $VENV_DIR (override with VENV_DIR=…)"
    exit 1
fi
source "$VENV_DIR/bin/activate"

# Do-PFN source — default to a project-local checkout for consistent jobs.
export DOPFN_SRC="${DOPFN_SRC:-$PROJ_DIR/Do-PFN}"
if [ ! -d "$DOPFN_SRC/priors/playground_scm" ]; then
    echo "ERROR: DOPFN_SRC=$DOPFN_SRC does not contain priors/playground_scm/"
    exit 1
fi

echo "=== Node info ==="
hostname
date
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "================="

# ── Model (small — matches train_cfm_small.py defaults) ─────────────────
export J=100
export D_MODEL=64
export DEPTH=2
export HEADS=4
export DROPOUT=0.0

# ── Training ────────────────────────────────────────────────────────────
export N_STEPS=200
export LR=1e-3
export LOG_EVERY=20
export N_CONTEXT_TRAIN=100
export N_QUERY_TRAIN=30

# ── Do-PFN prior ────────────────────────────────────────────────────────
export NUM_FEATURES=10    # fixed by prior, no padding
export N_TRAIN=200        # per-task SCM sample sizes emitted by the dataset
export N_TEST=100

# ── Streaming ───────────────────────────────────────────────────────────
export STREAM_WORKERS=4
export STREAM_SEED=42
export STREAM_WARMUP=4

time python -u training/train_cfm_small_dopfn.py
