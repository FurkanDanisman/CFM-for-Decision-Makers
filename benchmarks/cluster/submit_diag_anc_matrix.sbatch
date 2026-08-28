#!/bin/bash
#SBATCH --job-name=diag-anc
#SBATCH --output=logs_diag_anc/diag_%j.out
#SBATCH --error=logs_diag_anc/diag_%j.err
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Diagnose why graph2d anc mode fails on IHDP even though the model
# subclasses UWYK's graph-conditioned architecture.
#
# Two checks:
#   (1) training_graph2d/sanity_check.py — asserts graph_encoder receives
#       non-zero gradient (i.e. adjacency path is wired), and prints
#       actual training-time anc_matrix shape.
#   (2) inline python that samples 10 training-time anc_matrix examples
#       and compares their value distribution to the eval's build_anc_full.

set -euo pipefail
mkdir -p logs_diag_anc

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
export UWYK_SRC="${UWYK_SRC:-$DEPLOY_ROOT/external/uwyk/src}"
export UWYK_ROOT="${UWYK_ROOT:-$DEPLOY_ROOT/external/uwyk}"
export CAUSALPFN_ROOT="${CAUSALPFN_ROOT:-$DEPLOY_ROOT/external/causalpfn}"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "════════════════════════════════════════════════════════════════"
echo "  [1/2] sanity_check.py — assert graph_encoder gradient flows"
echo "════════════════════════════════════════════════════════════════"
cd "$REPO"
python -u training_graph2d/sanity_check.py || echo "SANITY_CHECK_FAILED=1"

echo
echo "════════════════════════════════════════════════════════════════"
echo "  [2/2] compare training anc_matrix vs eval build_anc_full values"
echo "════════════════════════════════════════════════════════════════"

python -u - <<'PYEOF'
import numpy as np
import sys, os

REPO = os.environ.get('REPO', '/scratch/furkanbd/rpfn_bench_kit/R-PFN')
UWYK_SRC = os.environ.get('UWYK_SRC')
sys.path.insert(0, REPO)
sys.path.insert(0, UWYK_SRC)

from training.data.PairedInterventionalDataset import PairedInterventionalDataset
from benchmarks.eval_graph2d.eval_graph2d_ihdp import build_anc_full

ds = PairedInterventionalDataset()

train_ancs = []
for i in range(10):
    b = ds[i]
    a = b['anc_matrix']
    if hasattr(a, 'numpy'):
        a = a.numpy()
    train_ancs.append(np.asarray(a))
    print(f'  sample {i}: shape={a.shape}  '
          f'unique={np.unique(a).tolist()}  '
          f'min={a.min()}  max={a.max()}')

print()
print('=== TRAINING anc_matrix aggregate distribution (10 samples) ===')
all_vals = np.concatenate([a.reshape(-1) for a in train_ancs])
print(f'  total elements: {all_vals.size}')
for v in np.unique(all_vals):
    frac = (all_vals == v).mean()
    print(f'  value={v:+.4f}  count={(all_vals==v).sum():>8d}  frac={frac:.4f}')

# Also look at the REAL block only (first (F_real+2)^2, where F_real is
# the number of real features in the first sample - we don't know exact
# per-sample sizes so just use full matrix distribution above).

print()
print('=== EVAL build_anc_full(F=100, n_real=25) distribution ===')
eval_A = build_anc_full(100, 25)
real_size = 2 + 25
eval_real = eval_A[:real_size, :real_size]
eval_pad = np.concatenate([eval_A[real_size:, :].reshape(-1),
                            eval_A[:real_size, real_size:].reshape(-1)])
print(f'  real block ({real_size}x{real_size} = {real_size*real_size} elems):')
for v in np.unique(eval_real):
    frac = (eval_real == v).mean()
    print(f'    value={v:+.4f}  count={(eval_real==v).sum():>6d}  frac={frac:.4f}')
print(f'  padded rows/cols ({eval_pad.size} elems):')
for v in np.unique(eval_pad):
    frac = (eval_pad == v).mean()
    print(f'    value={v:+.4f}  count={(eval_pad==v).sum():>6d}  frac={frac:.4f}')

# Save first training anc_matrix and eval matrix to disk for further inspection.
np.savez(
    os.path.join('/scratch/furkanbd/rpfn_bench_kit', 'diag_anc_matrices.npz'),
    train_sample_0=train_ancs[0],
    eval_anc_full_F100_n25=eval_A,
)
print()
print('Saved: /scratch/furkanbd/rpfn_bench_kit/diag_anc_matrices.npz')
PYEOF

echo
echo "[$(date '+%Y-%m-%d %H:%M:%S')] diagnostic done"
