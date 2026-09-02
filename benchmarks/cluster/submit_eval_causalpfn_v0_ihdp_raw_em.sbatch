#!/bin/bash
#SBATCH --job-name=cpfn-1d-rawem
#SBATCH --output=logs_cpfn_v0_ihdp_raw_em/eval_%j.out
#SBATCH --error=logs_cpfn_v0_ihdp_raw_em/eval_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Pure-ICL IHDP eval of a CausalPFN 1D checkpoint — raw AND em mean.
# Bypasses CATEEstimator + faiss entirely.
#
# Env: CPFN_V0_LOCAL, OUT, CAUSALPFN

set -euo pipefail
mkdir -p logs_cpfn_v0_ihdp_raw_em

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
export CPFN_V0_LOCAL="${CPFN_V0_LOCAL:?required — path to 1D CausalPFN ckpt}"
export OUT="${OUT:?required — output dir}"

[ -f "$CPFN_V0_LOCAL" ] || { echo "FATAL: ckpt not found at $CPFN_V0_LOCAL" >&2; exit 1; }

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 1D CausalPFN pure-ICL IHDP raw+em"
echo "  CKPT=$CPFN_V0_LOCAL"
echo "  OUT=$OUT"
python -u "$REPO/benchmarks/eval_causalpfn2d/eval_causalpfn_v0_ihdp_raw_em.py"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] done"
