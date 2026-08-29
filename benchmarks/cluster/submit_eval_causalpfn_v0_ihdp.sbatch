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

# Local pretrained ckpt — pre-downloaded on login node into $DEPLOY_ROOT/warmstart/.
# If missing here, eval falls back to HF download (may fail if compute
# node has no internet).
export CPFN_V0_LOCAL="${CPFN_V0_LOCAL:-$DEPLOY_ROOT/warmstart/causalpfn_v0.pt}"
if [ ! -f "$CPFN_V0_LOCAL" ]; then
    echo "WARNING: $CPFN_V0_LOCAL not found; eval will try HF download at runtime." >&2
    echo "  Pre-download on login node with:" >&2
    echo "    PYTHONPATH= python -c \"from huggingface_hub import hf_hub_download; print(hf_hub_download('vdblm/causalpfn', 'causalpfn_v0.pt'))\"" >&2
    echo "    cp <printed-path> $CPFN_V0_LOCAL" >&2
fi

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# Shim on PYTHONPATH — sitecustomize.py registers a LAST-RESORT meta_path
# finder that stubs faiss/transformers/wandb/etc. Real packages (like
# huggingface_hub installed via `pip install huggingface-hub`) take
# precedence over the stub. This lets `from causalpfn import CATEEstimator`
# succeed on Alliance Canada where faiss-cpu can't be built from source.
# We don't hit any code paths that actually call faiss functions, so the
# MagicMock stub is harmless.
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] launching causalpfn_v0 IHDP reference eval"
echo "  CAUSALPFN=$CAUSALPFN"
echo "  CPFN_V0_LOCAL=$CPFN_V0_LOCAL  (exists: $([ -f "$CPFN_V0_LOCAL" ] && echo yes || echo no))"
echo "  OUT=$OUT"
python -u "$REPO/benchmarks/eval_causalpfn2d/eval_causalpfn_v0_ihdp.py"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] done"
