#!/bin/bash
#SBATCH --job-name=rc-2d-malc-ci
#SBATCH --output=logs_rc_2d_malc_ci/dump_%A_%a.out
#SBATCH --error=logs_rc_2d_malc_ci/dump_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --array=0-14

# 15-task array — 3 2D methods × 5 RealCause datasets. CPU-only (MALC is
# CVXPY / numpy). Each task processes all realizations for one (method,
# dataset) cell using a persistent multiprocessing Pool of 32 workers
# (pool init ~5s from MALC imports, so per-realization spawning would
# waste minutes).
#
#   method_idx = task / 5   → 0=cpfn2d 1=graph2d 2=dopfnbb
#   ds_idx     = task % 5   → 0=IHDP 1=ACIC 2=CPS 3=PSID 4=PSID_bal
#
# Writes malc_ci_r<###>.npz next to each density NPZ. Resumable — the
# compute script skips realizations whose output already exists (unless
# --overwrite).
#
# MALC hyperparameters:
#   K = 1     (skips BIC scan, fits a single 2D log-concave component)
#   B = 100   (bandwidth parameter)
#   n_eval = 50    (fine 2D evaluation grid — 50×50 = 2500 points)
#
# After the run:
#   sbatch benchmarks/cluster/submit_realcause_2d_malc_ci_summary.sbatch

set -euo pipefail
mkdir -p logs_rc_2d_malc_ci

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT required — same one the 2D density-dump job wrote to}"
N_WORKERS="${N_WORKERS:-32}"
MALC_K="${MALC_K:-1}"
MALC_MAX_K="${MALC_MAX_K:-0}"       # if > 0, BIC-select K in {1..MAX_K} instead of forced K
MALC_B="${MALC_B:-100}"
N_EVAL="${N_EVAL:-50}"
MAX_REAL="${MAX_REAL:-}"        # optional cap (mostly for smoke tests)
OVERWRITE="${OVERWRITE:-0}"     # 1 to redo realizations whose malc_ci_*.npz already exists
OUT_TAG="${OUT_TAG:-}"          # optional filename tag → malc_ci_{TAG}_r<###>.npz

METHOD_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
DS_IDX=$(( SLURM_ARRAY_TASK_ID % 5 ))
case "$METHOD_IDX" in
    0) METHOD=cpfn2d  ;;
    1) METHOD=graph2d ;;
    2) METHOD=dopfnbb ;;
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

# dopfnbb legacy dir naming (PSIDbal instead of PSID_bal).
DS_ON_DISK="$DATASET"; SPLIT_FLAG=""
if [ "$METHOD" = "dopfnbb" ]; then
    SPLIT_FLAG="--is-split"
    [ "$DATASET" = "PSID_bal" ] && DS_ON_DISK="PSIDbal"
fi

METHOD_DIR="$OUT_ROOT/$METHOD"
CELL_DIR="$METHOD_DIR/$DS_ON_DISK"
[ -d "$CELL_DIR" ] || { echo "FATAL: cell dir not found: $CELL_DIR" >&2; exit 1; }

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] $METHOD  $DATASET  CELL=$CELL_DIR  "\
"K=$MALC_K  B=$MALC_B  n_eval=$N_EVAL  workers=$N_WORKERS"

_EXTRA_ARGS=()
[ -n "$MAX_REAL" ]        && _EXTRA_ARGS+=(--max-realizations "$MAX_REAL")
[ "$OVERWRITE" = "1" ]    && _EXTRA_ARGS+=(--overwrite)
[ -n "$OUT_TAG" ]         && _EXTRA_ARGS+=(--out-tag "$OUT_TAG")

python -u "$REPO/realcause_eval/compute_malc_ci_cell.py" \
    --method-dir "$METHOD_DIR" \
    --dataset    "$DS_ON_DISK" \
    $SPLIT_FLAG \
    --n-workers  "$N_WORKERS" \
    --malc-K     "$MALC_K" \
    --malc-max-K "$MALC_MAX_K" \
    --malc-B     "$MALC_B" \
    --n-eval     "$N_EVAL" \
    --repo       "$REPO" \
    "${_EXTRA_ARGS[@]}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] done  $METHOD  $DATASET"
