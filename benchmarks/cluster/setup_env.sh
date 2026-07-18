#!/bin/bash
# One-shot environment setup for killarney.
# Assumes the rsync payload is already in $DEPLOY_ROOT and contains:
#   R-PFN/, external/dopfn/, external/uwyk/, external/causalpfn/
#
# Usage:
#   cd /path/on/killarney/rpfn_bench_kit
#   bash R-PFN/experiments/killarney/setup_env.sh

set -euo pipefail

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
PY="${PYTHON:-python3.11}"

echo "[1/4] Creating venv at $DEPLOY_ROOT/venv (using $PY)"
if [ ! -d "$DEPLOY_ROOT/venv" ]; then
    $PY -m venv "$DEPLOY_ROOT/venv"
fi
source "$DEPLOY_ROOT/venv/bin/activate"
python -m pip install --upgrade pip wheel

echo "[2/4] Installing deps (relaxed pins so Compute Canada wheels are OK)"
# CC wheelhouse pins to +computecanada builds; use compatible-release ranges.
pip install \
    "torch~=2.11.0"        \
    "numpy~=2.4.0"         \
    "scipy~=1.15.0"        \
    "scikit-learn~=1.8.0"  \
    "pandas~=3.0.0"        \
    "matplotlib~=3.10.0"   \
    "networkx~=3.6.0"      \
    "PyYAML~=6.0.0"        \
    "einops~=0.8.0"        \
    "tqdm~=4.68.0"         \
    "scikit-uplift~=0.5.0"

echo "[3/4] Installing Do-PFN + UWYK sources (editable, so patches stay)"
pip install -e "$DEPLOY_ROOT/external/dopfn" || echo "  (dopfn editable install optional; script sets sys.path)"
# UWYK: no setup.py, we sys.path it in run_one.py — nothing to install.
# CausalPFN benchmarks: sys.path in run_one.py — nothing to install.

echo "[4/4] Smoke test"
cd "$DEPLOY_ROOT/external/dopfn"
python - <<'PY'
import sys, os
DEPLOY = os.environ.get('DEPLOY_ROOT', os.getcwd())
sys.path.insert(0, DEPLOY + '/external/dopfn')
sys.path.insert(0, DEPLOY + '/external/causalpfn')
sys.path.insert(0, DEPLOY + '/external/uwyk/src')
sys.path.insert(0, DEPLOY + '/R-PFN')
sys.path.insert(0, DEPLOY + '/R-PFN/MALC')
# Try loading each piece
from benchmarks import IHDPDataset
d = IHDPDataset()
cd, ad = d[0]
print(f"  IHDP: X_train {cd.X_train.shape}, ATE={ad.true_ate:.3f}")
import torch
ck = torch.load(DEPLOY + '/R-PFN/checkpoints/step_50000_final.pt', map_location='cpu', weights_only=False)
print(f"  OURS ckpt: J={ck['config']['J']}, d_model={ck['config']['d_model']}")
print("  All imports OK.")
PY
echo "SETUP COMPLETE."
