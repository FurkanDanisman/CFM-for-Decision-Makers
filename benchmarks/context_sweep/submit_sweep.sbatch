#!/bin/bash
#SBATCH --job-name=rpfn-ctxsweep
#SBATCH --output=logs_sweep/task_%A_%a.out
#SBATCH --error=logs_sweep/task_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --array=0-99%50
# On Trillium:
#   - Do NOT set --mem: whole-node allocation, 745 GiB always available.
#   - Submit from a scratch directory (e.g. $SCRATCH/rpfn_sweep) — /project is
#     read-only on compute nodes, so both #SBATCH --output/--error and any
#     files the job writes must live under $SCRATCH.
#   - Account: def-rgrosse or def-zhijing (pass at sbatch time, no #SBATCH default).
# On Killarney / other Alliance clusters that DO allow per-task mem: uncomment
# the two lines below.
##SBATCH --cpus-per-task=8
##SBATCH --mem=64G
# ── Array layout ─────────────────────────────────────────────────────────────
# Alliance Canada limits array size to 1000. To cover 500 SCMs × 5 N × 2 sources
# = 5000 SCM-N-source configurations, we batch N_SCMS_PER_TASK SCMs per SLURM
# task. Total tasks = |SOURCES| × N_CONTEXTS × N_BATCHES.
#
# Default: SOURCES=(prior poly) → 2×5×10 = 100 tasks, array=0-99.
# Poly-only mode:  set SOURCES_FILTER=poly (or "prior") to run only one source.
#                  Poly-only: 50 tasks — submit with --array=0-49.
# Prior-only:      same, submit --array=0-49.
#
# Layout (given the SOURCES you've filtered to):
#   src_idx   = TASK / (N_CONTEXTS × N_BATCHES)   ∈ {0..|SOURCES|-1}
#   remain    = TASK % (N_CONTEXTS × N_BATCHES)
#   N_idx     = remain / N_BATCHES                ∈ {0..N_CONTEXTS-1}
#   batch_idx = remain % N_BATCHES                ∈ {0..N_BATCHES-1}
#   seeds     = batch_idx * N_SCMS_PER_TASK .. (batch_idx+1) * N_SCMS_PER_TASK - 1

set -euo pipefail
mkdir -p logs_sweep results_sweep

# ── Path layout (all env-overridable) ────────────────────────────────────────
# Two supported layouts:
#   (a) killarney-style: DEPLOY_ROOT/{R-PFN, external, venv, results_sweep, ...}
#   (b) trillium-style : R-PFN is DEPLOY_ROOT itself; external is R-PFN/external
DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
# If R-PFN dir isn't at $DEPLOY_ROOT/R-PFN, fall back to $DEPLOY_ROOT itself
if [ ! -d "$REPO/benchmarks" ]; then
    REPO="$DEPLOY_ROOT"
fi
# Prefer $DEPLOY_ROOT/external/... (killarney layout) with a fallback to
# $REPO/external/... (trillium layout where R-PFN IS the deploy root).
if [ -d "$DEPLOY_ROOT/external/uwyk/src" ]; then
    UWYK_SRC="${UWYK_SRC:-$DEPLOY_ROOT/external/uwyk/src}"
else
    UWYK_SRC="${UWYK_SRC:-$REPO/external/uwyk/src}"
fi
if [ -d "$DEPLOY_ROOT/external/causalpfn" ]; then
    CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
else
    CAUSALPFN="${CAUSALPFN:-$REPO/external/causalpfn}"
fi
DOPFN="${DOPFN:-$DEPLOY_ROOT/external/dopfn}"
CHECKPOINT="${CHECKPOINT:-$REPO/checkpoints/step_50000_final.pt}"
if [ -d "$DEPLOY_ROOT/external/uwyk/experiments" ]; then
    UWYK_CKPT_DIR="${UWYK_CKPT_DIR:-$DEPLOY_ROOT/external/uwyk/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0}"
else
    UWYK_CKPT_DIR="${UWYK_CKPT_DIR:-$REPO/external/uwyk/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0}"
fi
OUTDIR="${OUTDIR:-$DEPLOY_ROOT/results_sweep}"
VENV="${VENV:-$DEPLOY_ROOT/.venv}"
# Legacy Killarney venv path
[ ! -f "$VENV/bin/activate" ] && [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && VENV="$DEPLOY_ROOT/venv"

N_SCMS_TOTAL="${N_SCMS_TOTAL:-500}"
N_SCMS_PER_TASK="${N_SCMS_PER_TASK:-50}"
OURS_ONLY_FLAG=""
[ "${OURS_ONLY:-0}" = "1" ] && OURS_ONLY_FLAG="--ours-only"
BACKBONE="${BACKBONE:-ipfn}"                        # ipfn | dopfn_bb | graph2d
ANC_MODE="${ANC_MODE:-v6a}"                          # v6a | noanc  (only if BACKBONE=graph2d)
OURS_MAX_N_TRAIN="${OURS_MAX_N_TRAIN:-1000}"         # ≥ N_context disables 1000-cluster cap
MALC_B="${MALC_B:-500}"                              # MALC bootstrap draws
MALC_MAX_K="${MALC_MAX_K:-3}"                        # MALC mixture max K
Y_LOG_TRANSFORM="${Y_LOG_TRANSFORM:-0}"              # 1 → log(y - y_min + 1) inference
Y_POWER_TRANSFORM="${Y_POWER_TRANSFORM:-0}"          # 1 → Yeo-Johnson on Y at inference
Y_WINSORIZE="${Y_WINSORIZE:-0}"                      # e.g. 5 → (5%, 95%) Y quantile scaling
SKIP_MALC="${SKIP_MALC:-0}"                          # 1 → skip MALC pool (raw-mean only)
N_BATCHES=$(( N_SCMS_TOTAL / N_SCMS_PER_TASK ))    # 500 / 50 = 10
CONTEXTS=(50 250 500 1000 5000)
N_CONTEXTS=${#CONTEXTS[@]}                          # 5

# ── SOURCES_FILTER: run only prior, only poly, or both ────────────────────────
case "${SOURCES_FILTER:-both}" in
    prior) SOURCES=(prior) ;;
    poly)  SOURCES=(poly) ;;
    both|"") SOURCES=(prior poly) ;;
    *) echo "unknown SOURCES_FILTER: ${SOURCES_FILTER}" >&2; exit 2 ;;
esac
N_SOURCES=${#SOURCES[@]}
WORKERS="${SLURM_CPUS_PER_TASK:-8}"

TASK=${SLURM_ARRAY_TASK_ID:-0}
SRC_IDX=$(( TASK / (N_CONTEXTS * N_BATCHES) ))
REMAIN=$(( TASK % (N_CONTEXTS * N_BATCHES) ))
N_IDX=$(( REMAIN / N_BATCHES ))
BATCH_IDX=$(( REMAIN % N_BATCHES ))
SOURCE="${SOURCES[$SRC_IDX]}"
N_CTX="${CONTEXTS[$N_IDX]}"
SEED_START=$(( BATCH_IDX * N_SCMS_PER_TASK ))
SEED_END=$(( SEED_START + N_SCMS_PER_TASK - 1 ))

echo "[$(date '+%Y-%m-%d %H:%M:%S')] TASK=$TASK  SOURCE=$SOURCE  N=$N_CTX  seeds=[$SEED_START..$SEED_END]"
echo "  REPO=$REPO"
echo "  UWYK_SRC=$UWYK_SRC"
echo "  UWYK_CKPT_DIR=$UWYK_CKPT_DIR"
echo "  CAUSALPFN=$CAUSALPFN"
echo "  OUTDIR=$OUTDIR"

source "$VENV/bin/activate"
export OMP_NUM_THREADS="$WORKERS"
export MKL_NUM_THREADS="$WORKERS"
export OPENBLAS_NUM_THREADS="$WORKERS"
export PYTHONUNBUFFERED=1

for SEED in $(seq "$SEED_START" "$SEED_END"); do
    OUT_FILE="$OUTDIR/${SOURCE}_seed$(printf '%04d' "$SEED")_N$(printf '%06d' "$N_CTX").npz"
    if [ -f "$OUT_FILE" ]; then
        echo "[skip] $OUT_FILE exists"
        continue
    fi
    python "$REPO/benchmarks/context_sweep/run_one.py" \
        --source        "$SOURCE" \
        --seed          "$SEED" \
        --n-context     "$N_CTX" \
        --outdir        "$OUTDIR" \
        --repo          "$REPO" \
        --uwyk-src      "$UWYK_SRC" \
        --uwyk-ckpt-dir "$UWYK_CKPT_DIR" \
        --causalpfn     "$CAUSALPFN" \
        --checkpoint    "$CHECKPOINT" \
        --dopfn         "$DOPFN" \
        --workers       "$WORKERS" \
        --backbone      "$BACKBONE" \
        --anc-mode      "$ANC_MODE" \
        --ours-max-n-train "$OURS_MAX_N_TRAIN" \
        --malc-B        "$MALC_B" \
        --malc-max-K    "$MALC_MAX_K" \
        --y-winsorize   "$Y_WINSORIZE" \
        $( [ "$Y_LOG_TRANSFORM"   = "1" ] && echo "--y-log-transform" ) \
        $( [ "$Y_POWER_TRANSFORM" = "1" ] && echo "--y-power-transform" ) \
        $( [ "$SKIP_MALC"         = "1" ] && echo "--skip-malc" ) \
        $OURS_ONLY_FLAG || echo "[warn] seed=$SEED failed, continuing"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE task=$TASK"
