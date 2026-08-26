#!/bin/bash
#SBATCH --job-name=cpfn2d-sanity
#SBATCH --output=logs_causalpfn2d/sanity_%j.out
#SBATCH --error=logs_causalpfn2d/sanity_%j.err
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

# Smoke test for the CausalPFN 2D-head trainer -- verifies:
#   1. CausalPFN's branch codex/add-training-code is checked out
#      (training/priors/BackdoorDGPMetaDataset importable).
#   2. faiss / huggingface_hub shims resolve.
#   3. A tiny model forward + backward + null_t_intv gradient.
#
# Runs in ~30s on any GPU. No writes to persistent state.

set -euo pipefail
mkdir -p logs_causalpfn2d

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
export CAUSALPFN_ROOT="${CAUSALPFN_ROOT:-$DEPLOY_ROOT/external/causalpfn}"
[ -f "$CAUSALPFN_ROOT/src/causalpfn/training/priors/backdoor_prior.py" ] || {
    echo "FATAL: CausalPFN's codex/add-training-code branch not checked out at $CAUSALPFN_ROOT" >&2
    echo "  git -C $CAUSALPFN_ROOT fetch origin codex/add-training-code && git -C $CAUSALPFN_ROOT checkout codex/add-training-code" >&2
    exit 1
}

source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1
cd "$REPO"
python -u training_causalpfn2d/sanity_check.py
