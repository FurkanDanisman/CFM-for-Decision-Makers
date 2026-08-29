#!/bin/bash
#SBATCH --job-name=cpfn-v0-min
#SBATCH --output=logs_cpfn_v0_ihdp/eval_min_%j.out
#SBATCH --error=logs_cpfn_v0_ihdp/eval_min_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Minimal IHDP eval on causalpfn_v0.pt — loads state_dict directly into
# TabDPTLongContextModel, bypasses CATEEstimator entirely (no faiss needed).
# Pure ICL inference, per-arm y-standardisation. Matches paper's setup
# except for the retrieval step (which needs faiss and isn't installable
# on Alliance Canada).

set -euo pipefail
mkdir -p logs_cpfn_v0_ihdp

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
export CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
export CPFN_V0_LOCAL="${CPFN_V0_LOCAL:-$DEPLOY_ROOT/warmstart/causalpfn_v0.pt}"
export OUT="${OUT:-$DEPLOY_ROOT/results_causalpfn_v0_ihdp_minimal}"

[ -f "$CPFN_V0_LOCAL" ] || { echo "FATAL: ckpt not found at $CPFN_V0_LOCAL" >&2; exit 1; }

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# Shim needed for faiss stub — causalpfn/__init__.py chains through
# causal_estimator which does `import faiss` at module top. We don't
# CALL any faiss functions in this script; the stub just satisfies
# the import.
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] launching MINIMAL causalpfn_v0 eval"
echo "  CPFN_V0_LOCAL=$CPFN_V0_LOCAL"
echo "  OUT=$OUT"
python -u "$REPO/benchmarks/eval_causalpfn2d/eval_causalpfn_v0_ihdp_minimal.py"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] done"
