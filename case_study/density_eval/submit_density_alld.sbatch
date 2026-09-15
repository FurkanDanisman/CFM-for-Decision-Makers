#!/bin/bash
#SBATCH --output=logs_density/%x_%j.out
#SBATCH --error=logs_density/%x_%j.err
#SBATCH --time=23:59:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#
# ONE (shift, model) density-dump job at a fixed context: loops ALL d and ALL
# cases, dumping up to REAL_END realizations per cell. Driven by env from
# 07_backfill_density_1000.sh.
#
# RECOVERABLE ON TIMEOUT.  eval_density_tauC.py writes each realization as its
# own <Case>_r###.npz the moment it finishes, so a wall-clock kill loses only
# the single in-flight realization -- every completed one is on disk. This job
# is IDEMPOTENT: for each (d, case) it counts the realizations already present
# and resumes at REAL_START = that count, so simply RE-SUBMITTING the same job
# after a timeout backfills only what is missing (no recompute of finished work).
#
# Required env: REPO HERE MODEL SHIFT CTX DATA RES CASES MODEL_FAMILY
#               (+ per-model: UWYK_CKPT/UWYK_CFG/CKPT for uwyk-family/graph2d,
#                 DOPFN_JOINT_CKPT for dopfn_bb)
# Optional:     DS (default all), REAL_END (default 100), SCM_N_QUERY (default 100)
set -uo pipefail                     # NOT -e: one bad cell must not abort the rest
mkdir -p logs_density

: "${REPO:?}" "${HERE:?}" "${MODEL:?}" "${SHIFT:?}" "${CTX:?}" "${DATA:?}" \
  "${RES:?}" "${CASES:?}" "${MODEL_FAMILY:?}" "${DEPLOY_ROOT:?}"
source "${VENV:-$DEPLOY_ROOT/venv}/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
# Same causalpfn/faiss shim the other density jobs use.
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
[ -f "$REPO/benchmarks/uwyk_table1/shims/sitecustomize.py" ] || {
    echo "FATAL: shim missing at $REPO/benchmarks/uwyk_table1/shims" >&2; exit 1; }

DS="${DS:-2 3 5 10 20 30 40 50}"
REAL_END="${REAL_END:-100}"
IFS=',' read -ra CASE_LIST <<< "$CASES"

echo "[alld] shift$SHIFT model=$MODEL ctx=$CTX ds=[$DS] real_end=$REAL_END"
for d in $DS; do
  DROOT="$DATA/shift$SHIFT/d$d"
  if [ ! -d "$DROOT" ]; then echo "  skip (no data): $DROOT"; continue; fi
  for CASE in "${CASE_LIST[@]}"; do
    CELL="$RES/shift$SHIFT/d$d/ctx${CTX}/$MODEL/$CASE"
    mkdir -p "$CELL"
    # Resume: highest existing DENSITY dump index + 1. Count predictions/ ONLY.
    # The cell top level also holds <CASE>_r###.npz from the earlier POINT
    # sweep, and those go up to r099 in cells where no density has ever been
    # dumped -- counting them made the cell look complete and skipped it
    # forever (graph2d/uwyk at d5 had 120 point npz and 21 dumps, and every
    # backfill exited in 2 s having written nothing). predictions/ is written
    # in both SCORE_INLINE modes, so this is the artifact that tracks progress.
    last=$( ls "$CELL"/predictions/${CASE}_r*.npz 2>/dev/null \
           | sed -E 's/.*_r0*([0-9]+)\.npz$/\1/' | sort -n | tail -1)
    start=$(( ${last:--1} + 1 ))
    if [ "$start" -ge "$REAL_END" ]; then
      echo "  d$d $CASE complete (have $start)"; continue
    fi
    echo "  === d$d $CASE  resume@$start -> $REAL_END ==="
    env DATASET="$CASE" OUT="$CELL" KIND=tauC MODEL_FAMILY="$MODEL_FAMILY" \
        CASE_STUDY_DATA_ROOT="$DROOT" CASE_STUDY_N="$CTX" DATA_ROOT="$DROOT" \
        DENSITY_DUMP=1 REAL_START="$start" REAL_END="$REAL_END" \
        SCM_N_QUERY="${SCM_N_QUERY:-100}" \
        python -u "$HERE/eval_density_tauC.py" \
      || echo "  WARN: d$d $CASE exited nonzero (completed dumps kept; resume next run)"
  done
done
echo "[alld] done shift$SHIFT $MODEL"
