#!/bin/bash
#SBATCH --job-name=rc-1d-malc-ci
#SBATCH --output=logs_rc_1d_malc_ci/dump_%A_%a.out
#SBATCH --error=logs_rc_1d_malc_ci/dump_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --array=0-14

# 15-task array — 3 1D methods × 5 RealCause datasets. CPU-only.
# Builds p_joint = p_y0 ⊗ p_y1 per query from 1D marginals, fits 2D-MALC,
# diag-integrates → p(τ). Same MALC recipe as the 2D dump, differs only
# in how p_joint is sourced.
#
#   method_idx = task / 5   → 0=cpfn1d 1=dopfn 2=uwyk1d
#   ds_idx     = task % 5   → 0=IHDP 1=ACIC 2=CPS 3=PSID 4=PSID_bal
#
# Writes malc_ci_<TAG>_r<###>.npz next to each 1D density NPZ.
#
# MALC hyperparameters:
#   K = 1     (skips BIC scan, fits a single 2D log-concave component)
#   B = 100   (bandwidth, matches 2D dump default)
#   n_eval = 50
#
# Downsample: DEFAULT IS OFF (max_J=0 → no downsample, full K passed to
# MALC). Actual K per method:
#   cpfn1d  K=1024 → 1024² = 1.05M atoms per query
#   dopfn   K=100  →  100² = 10K   atoms per query
#   uwyk1d  K=1000 (after tail-trim from K+2=1002) → 1M atoms per query
# MALC compute grows with atom count; the K=1024/1000 cells are the
# tail. If a cell doesn't finish in the 6h wall, override DOWNSAMPLE_MAX_J
# (e.g. =100 → cpfn1d→J=64, uwyk1d→J=100, dopfn no-op; largest divisor
# ≤ max_J is used so mass conservation is exact).
#
# Env vars:
#   OUT_ROOT     required — root of 1D density dumps (…/results_1d_density)
#   N_WORKERS    optional (default 32)
#   MALC_K       optional (default 1)
#   MALC_MAX_K   optional (default 0)  — BIC selection if > 0
#   MALC_B       optional (default 100)
#   N_EVAL       optional (default 50)
#   DOWNSAMPLE_MAX_J  optional (default 100) — 0 disables
#   MAX_REAL     optional cap on realizations (smoke tests)
#   OVERWRITE    optional (0/1) — redo existing outputs
#   OUT_TAG      optional filename tag → malc_ci_{TAG}_r<###>.npz.
#                Default '' → malc_ci_r<###>.npz. Recommended: 'B100'
#                to disambiguate from any future B-sweep results.

set -euo pipefail
mkdir -p logs_rc_1d_malc_ci

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT required — same one the 1D density-dump job wrote to}"
N_WORKERS="${N_WORKERS:-32}"
MALC_K="${MALC_K:-1}"
MALC_MAX_K="${MALC_MAX_K:-0}"
MALC_B="${MALC_B:-500}"
N_EVAL="${N_EVAL:-50}"
DOWNSAMPLE_MAX_J="${DOWNSAMPLE_MAX_J:-0}"   # 0 = no downsample (pass full K to MALC)
MAX_REAL="${MAX_REAL:-}"
OVERWRITE="${OVERWRITE:-0}"
OUT_TAG="${OUT_TAG:-}"

METHOD_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
DS_IDX=$(( SLURM_ARRAY_TASK_ID % 5 ))
case "$METHOD_IDX" in
    0) METHOD=cpfn1d ;;
    1) METHOD=dopfn  ;;
    2) METHOD=uwyk1d ;;
    *) echo "invalid METHOD_IDX=$METHOD_IDX" >&2; exit 2 ;;
esac
case "$DS_IDX" in
    0) DATASET=IHDP     ;;
    1) DATASET=ACIC     ;;
    2) DATASET=CPS      ;;
    3) DATASET=PSID     ;;
    4) DATASET=PSID_bal ;;
    *) echo "invalid DS_IDX=$DS_IDX" >&2; exit 2 ;;
esac

METHOD_DIR="$OUT_ROOT/$METHOD"
CELL_DIR="$METHOD_DIR/$DATASET"
[ -d "$CELL_DIR" ] || { echo "FATAL: cell dir not found: $CELL_DIR" >&2; exit 1; }

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] $METHOD  $DATASET  CELL=$CELL_DIR  "\
"K=$MALC_K  B=$MALC_B  n_eval=$N_EVAL  downsample=$DOWNSAMPLE_MAX_J  workers=$N_WORKERS"

_EXTRA_ARGS=()
[ -n "$MAX_REAL" ]        && _EXTRA_ARGS+=(--max-realizations "$MAX_REAL")
[ "$OVERWRITE" = "1" ]    && _EXTRA_ARGS+=(--overwrite)
[ -n "$OUT_TAG" ]         && _EXTRA_ARGS+=(--out-tag "$OUT_TAG")

python -u "$REPO/realcause_eval/compute_malc_ci_cell_1d.py" \
    --method-dir         "$METHOD_DIR" \
    --dataset            "$DATASET" \
    --n-workers          "$N_WORKERS" \
    --malc-K             "$MALC_K" \
    --malc-max-K         "$MALC_MAX_K" \
    --malc-B             "$MALC_B" \
    --n-eval             "$N_EVAL" \
    --downsample-max-J   "$DOWNSAMPLE_MAX_J" \
    --repo               "$REPO" \
    "${_EXTRA_ARGS[@]}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] done  $METHOD  $DATASET"
