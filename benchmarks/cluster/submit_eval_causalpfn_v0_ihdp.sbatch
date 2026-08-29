#!/bin/bash
#SBATCH --job-name=cpfn-v0-ihdp
#SBATCH --output=logs_cpfn_v0_ihdp/eval_%j.out
#SBATCH --error=logs_cpfn_v0_ihdp/eval_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Reference eval: run vdblm/causalpfn's pretrained causalpfn_v0.pt on
# IHDP using their own CATEEstimator API. Auto-downloads the checkpoint
# from HuggingFace on first run.

set -euo pipefail
mkdir -p logs_cpfn_v0_ihdp

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    if   [ -f "$(dirname "$DEPLOY_ROOT")/venv/bin/activate" ]; then
        DEPLOY_ROOT="$(dirname "$DEPLOY_ROOT")"
    elif [ -f "$(dirname "$(dirname "$DEPLOY_ROOT")")/venv/bin/activate" ]; then
        DEPLOY_ROOT="$(dirname "$(dirname "$DEPLOY_ROOT")")"
    fi
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
export CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
export OUT="${OUT:-$DEPLOY_ROOT/results_causalpfn_v0_ihdp}"

# HF cache dir — first run downloads causalpfn_v0.pt (~75MB) here.
# Reuse across runs so we don't re-download.
export HF_HOME="${HF_HOME:-$DEPLOY_ROOT/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$HF_HUB_CACHE"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# NOTE: no shims on PYTHONPATH here. causalpfn's own CATEEstimator needs
# real huggingface_hub / faiss — not shim stubs. Our shims exist for the
# TRAINING pipeline where we only need CausalPFN's model class (not its
# inference stack). For this reference eval we use the full package.

echo "[$(date '+%Y-%m-%d %H:%M:%S')] launching causalpfn_v0 IHDP reference eval"
echo "  CAUSALPFN=$CAUSALPFN"
echo "  HF_HUB_CACHE=$HF_HUB_CACHE"
echo "  OUT=$OUT"
python -u "$REPO/benchmarks/eval_causalpfn2d/eval_causalpfn_v0_ihdp.py"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] done"
