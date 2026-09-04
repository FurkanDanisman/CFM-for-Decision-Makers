#!/bin/bash
#SBATCH --job-name=scm-cs-regen-rho02
#SBATCH --output=logs_scm_case_studies_regen/regen_%j.out
#SBATCH --error=logs_scm_case_studies_regen/regen_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
# CPU-only: no --gres needed. Single job regenerates all 6 case studies.
#
# Regenerates DoPFN's 6 synthetic case-study pkls with bivariate Y-noise
# (rho=0.2 by default) so tau|x is a proper Gaussian rather than a delta.
#
# Env vars:
#   DOPFN_ROOT       path to the dopfn upstream repo root
#   OUT_DIR          destination root; pkls land at $OUT_DIR/prior_sampling_rho02/<Case>/
#   Y_NOISE_CORR     correlation rho (default 0.2)
#   N_PER_CASE       realizations per case (default 100)
#   SEED_BASE        base seed (default 42)
#   SEQ_LEN          rows per realization (default 500 → 200 train + 300 test)
#   NUM_FEATURES     case width (default 3)
#   NOISE_STD        Y-node noise std (default 0.01)
#   EXO_STD          exogenous noise std (default 0.1)
#   NONLINS          activation setting (default paper prior)
#   ONLY_CASES       optional space-separated subset

set -euo pipefail
mkdir -p logs_scm_case_studies_regen

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
export DOPFN_ROOT="${DOPFN_ROOT:-$DEPLOY_ROOT/external/dopfn}"
export OUT_DIR="${OUT_DIR:-$DEPLOY_ROOT/external/dopfn/data}"
Y_NOISE_CORR="${Y_NOISE_CORR:-0.2}"
N_PER_CASE="${N_PER_CASE:-100}"
SEED_BASE="${SEED_BASE:-42}"
SEQ_LEN="${SEQ_LEN:-500}"
NUM_FEATURES="${NUM_FEATURES:-3}"
NOISE_STD="${NOISE_STD:-0.01}"
EXO_STD="${EXO_STD:-0.1}"
NONLINS="${NONLINS:-sophisticated_sampling_1_rescaling_normalization}"
ONLY_CASES="${ONLY_CASES:-}"
OVERWRITE="${OVERWRITE:-0}"

[ -d "$DOPFN_ROOT" ] || { echo "FATAL: DOPFN_ROOT not a dir: $DOPFN_ROOT" >&2; exit 2; }

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

EXTRA=()
if [ -n "$ONLY_CASES" ]; then
    EXTRA+=(--only-cases $ONLY_CASES)
fi
if [ "$OVERWRITE" = "1" ]; then
    EXTRA+=(--overwrite)
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] regenerating case-study pkls"
echo "  DOPFN_ROOT=$DOPFN_ROOT"
echo "  OUT_DIR=$OUT_DIR"
echo "  Y_NOISE_CORR=$Y_NOISE_CORR  N_PER_CASE=$N_PER_CASE  SEQ_LEN=$SEQ_LEN"

python -u "$REPO/benchmarks/scripts/regen_case_study_pkls.py" \
    --dopfn-root  "$DOPFN_ROOT" \
    --out-dir     "$OUT_DIR" \
    --y-noise-corr "$Y_NOISE_CORR" \
    --n-per-case  "$N_PER_CASE" \
    --seed-base   "$SEED_BASE" \
    --seq-len     "$SEQ_LEN" \
    --num-features "$NUM_FEATURES" \
    --noise-std   "$NOISE_STD" \
    --exo-std     "$EXO_STD" \
    --nonlins     "$NONLINS" \
    "${EXTRA[@]}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] regen done"
