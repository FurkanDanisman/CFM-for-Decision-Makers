#!/bin/bash
# One-shot environment setup for Killarney (Alliance).
#
# Unlike setup_env.sh (which assumes a deploy_local.sh rsync payload), this
# builds $DEPLOY_ROOT in place from public sources:
#
#   $DEPLOY_ROOT/
#   ├── R-PFN                          -> symlink to this repo
#   ├── external/dopfn                    clone of jr2021/Do-PFN
#   ├── external/causalpfn                clone of vdblm/CausalPFN
#   ├── external/uwyk                  -> symlink to <repo>/g4cfm (+ git lfs pull)
#   ├── checkpoints_dopfn_backbone_j10/step_200000.pt
#   │                                  -> symlink to <repo>/checkpoints_shared/
#   │                                     dopfn_bb_step_200000.pt
#   └── venv/
#
# The sbatch scripts hardcode "$DEPLOY_ROOT/venv" (not .venv), hence the name.
#
# Usage:
#   export DEPLOY_ROOT=$SCRATCH/rpfn_bench_kit
#   bash /path/to/repo/benchmarks/cluster/setup_killarney.sh

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DEPLOY_ROOT="${DEPLOY_ROOT:-$SCRATCH/rpfn_bench_kit}"

DOPFN_URL="${DOPFN_URL:-https://github.com/jr2021/Do-PFN.git}"
CAUSALPFN_URL="${CAUSALPFN_URL:-https://github.com/vdblm/CausalPFN.git}"

echo "REPO=$REPO"
echo "DEPLOY_ROOT=$DEPLOY_ROOT"
mkdir -p "$DEPLOY_ROOT/external"

# ── 1. External repos ────────────────────────────────────────────────────────
# Do-PFN: needed for the `dopfn` baseline (DoPFNRegressor) AND for
# `ours_dopfn_bb`, which unpickles the PerFeatureTransformer *architecture*
# from artifacts/dopfn_model.pkl. Model artifacts are committed, not LFS.
if [ ! -d "$DEPLOY_ROOT/external/dopfn/.git" ]; then
    echo "[1/5] Cloning Do-PFN"
    git clone --depth 1 "$DOPFN_URL" "$DEPLOY_ROOT/external/dopfn"
else
    echo "[1/5] Do-PFN already present"
fi

# Patch: model/layer.py imports `Optional` from torch.nn.modules.transformer,
# which only ever worked because torch leaked its own `typing` import there.
# Removed in modern torch (gone by 2.13); every other name in that list still
# resolves. This is the "patched Do-PFN" the deploy_local.sh payload referred to.
python - "$DEPLOY_ROOT/external/dopfn/model/layer.py" <<'PY'
import sys
p = sys.argv[1]
src = open(p).read()
old = "from torch.nn.modules.transformer import (\n    _get_activation_fn,\n    Module,\n    Tensor,\n    Optional,\n"
new = "from typing import Optional\nfrom torch.nn.modules.transformer import (\n    _get_activation_fn,\n    Module,\n    Tensor,\n"
if old in src:
    open(p, 'w').write(src.replace(old, new, 1))
    print("  patched model/layer.py (Optional -> typing)")
else:
    print("  model/layer.py already patched (or upstream changed)")
PY

# Patch: base.py calls check_array(..., force_all_finite=...). sklearn renamed
# that to ensure_all_finite in 1.6 and dropped the old name in 1.8. Rename it
# forward — benchmarks/methods/dopfn.py already carries a shim that maps it
# *back* for anyone stuck on sklearn < 1.6, so both directions stay covered.
python - "$DEPLOY_ROOT/external/dopfn/scripts/transformer_prediction_interface/base.py" <<'PY'
import sys
p = sys.argv[1]
src = open(p).read()
n = src.count('force_all_finite=')
if n:
    open(p, 'w').write(src.replace('force_all_finite=', 'ensure_all_finite='))
print(f"  patched base.py check_array kwarg ({n} occurrences)")
PY

# CausalPFN: supplies the dataset loaders (IHDPDataset, ACIC2016Dataset, ...)
# and ships the IHDP NPZs under benchmarks/IHDP/.
if [ ! -d "$DEPLOY_ROOT/external/causalpfn/.git" ]; then
    echo "[2/5] Cloning CausalPFN"
    git clone --depth 1 "$CAUSALPFN_URL" "$DEPLOY_ROOT/external/causalpfn"
else
    echo "[2/5] CausalPFN already present"
fi

# UWYK is vendored in this repo as g4cfm/ (ArikReuter/Graphs4CausalFoundationModels).
# Its checkpoint is git-lfs; a fresh clone leaves a 134-byte pointer file.
echo "[3/5] Wiring UWYK (g4cfm) + pulling its LFS checkpoint"
UWYK_CKPT="$REPO/g4cfm/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0/best_model.pt"
if [ ! -f "$UWYK_CKPT" ] || [ "$(stat -c%s "$UWYK_CKPT")" -lt 1000000 ]; then
    git -C "$REPO/g4cfm" lfs pull --include="$(realpath --relative-to="$REPO/g4cfm" "$UWYK_CKPT")"
fi
ln -sfn "$REPO/g4cfm" "$DEPLOY_ROOT/external/uwyk"

# ── 2. Repo + checkpoint symlinks ────────────────────────────────────────────
echo "[4/5] Symlinking repo + checkpoints into DEPLOY_ROOT"
ln -sfn "$REPO" "$DEPLOY_ROOT/R-PFN"
mkdir -p "$DEPLOY_ROOT/checkpoints_dopfn_backbone_j10"
ln -sfn "$REPO/checkpoints_shared/dopfn_bb_step_200000.pt" \
        "$DEPLOY_ROOT/checkpoints_dopfn_backbone_j10/step_200000.pt"

# ── 3. venv ──────────────────────────────────────────────────────────────────
echo "[5/5] Building venv at $DEPLOY_ROOT/venv"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    module load python/3.11 2>/dev/null || true
    python3.11 -m venv "$DEPLOY_ROOT/venv"
fi
source "$DEPLOY_ROOT/venv/bin/activate"
python -m pip install --quiet --upgrade pip wheel

# Loose pins: the Alliance wheelhouse (find-links in the cvmfs pip config) wins
# over PyPI and only carries +computecanada builds, so exact pins fail to
# resolve. Everything here runs on CPU anyway — UWYK's loader hardcodes
# device='cpu', DoPFNRegressor defaults to cpu, and our checkpoints load with
# map_location='cpu' — so the wheelhouse torch is fine as-is.
pip install --quiet \
    "torch>=2.6" \
    "numpy>=2.0" \
    "scipy>=1.13" \
    "scikit-learn>=1.6" \
    "pandas>=2.2" \
    "matplotlib>=3.8" \
    "networkx>=3.2" \
    "PyYAML>=6.0" \
    "einops>=0.8" \
    "tqdm>=4.66" \
    "scikit-uplift>=0.5"

# clarabel is the conic solver behind MALC/log_concave_2d_fast.py. Not pulled
# in by anything above, and the wheelhouse only has 0.6.0 — take it from PyPI.
pip install --quiet "clarabel>=0.11"

echo
echo "SETUP COMPLETE. Smoke test with:"
echo "  DEPLOY_ROOT=$DEPLOY_ROOT bash $REPO/benchmarks/cluster/smoke_test.sh"
