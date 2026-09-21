#!/bin/bash
#SBATCH --job-name=cm-dump1
#SBATCH --output=logs_cmech_dump/cm_%j.out
#SBATCH --error=logs_cmech_dump/cm_%j.err
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# No type here on purpose: nibi rejects a bare gpu:N, and the right type varies by
# cluster, so the driver passes --gres explicitly and overrides this line.
#SBATCH --gres=gpu:1
#
# ONE job = ONE model on ComplexMech, walking 6 node counts x 2 subsets at N=1000.
#
# Wraps submit_cmech_pehe_1d_vs_2d.sbatch rather than duplicating its dispatch: the
# #SBATCH lines there are comments, so running it under plain bash with
# SLURM_ARRAY_TASK_ID set executes exactly the same body the array would.
#
# Inner index = model_idx * (n_nodes * n_contexts) + node_idx * n_contexts + ctx_idx.
# With CMECH_CONTEXTS pinned to one value, n_contexts = 1 and it reduces to
# model_idx * n_nodes + node_idx.
#
# Required: MODEL_NAME MODEL_IDX CKPT_ENV CKPT_PATH OUT_ROOT UWYK_FIG34_DATA
# Optional: EXTRA_ENV (space-separated VAR=VAL), CMECH_NODES, SUBSETS

set -uo pipefail
mkdir -p logs_cmech_dump
: "${MODEL_NAME:?}"; : "${MODEL_IDX:?}"; : "${CKPT_ENV:?}"
: "${OUT_ROOT:?}"; : "${UWYK_FIG34_DATA:?}"

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
INNER="$REPO/benchmarks/cluster/submit_cmech_pehe_1d_vs_2d.sbatch"
read -r -a NODES <<<"${CMECH_NODES:-5 10 20 30 40 50}"
read -r -a SUBSETS <<<"${SUBSETS:-nonzero zero}"
N_NODES=${#NODES[@]}

echo "[$(date '+%F %T')] $MODEL_NAME  harness_idx=$MODEL_IDX"
echo "  $CKPT_ENV=${CKPT_PATH:-（none)}"
echo "  data=$UWYK_FIG34_DATA  out=$OUT_ROOT"
echo "  nodes=${NODES[*]}  subsets=${SUBSETS[*]}  N=1000"
echo

FAILED=0
for sub in "${SUBSETS[@]}"; do
    for i in "${!NODES[@]}"; do
        task=$(( MODEL_IDX * N_NODES + i ))
        echo "===== [$(date '+%F %T')] n=${NODES[$i]} subset=$sub (inner task $task)"
        # Fresh process per cell: one bad cell cannot leave exported state behind.
        ENVS=(OUT_ROOT="$OUT_ROOT" UWYK_FIG34_DATA="$UWYK_FIG34_DATA"
              DENSITY_DUMP=1 SUBSET="$sub" CMECH_CONTEXTS=1000
              CMECH_NODES="${NODES[*]}")
        [ "$CKPT_ENV" != NONE ] && ENVS+=("$CKPT_ENV=$CKPT_PATH")
        # shellcheck disable=SC2086
        SLURM_ARRAY_TASK_ID="$task" env "${ENVS[@]}" ${EXTRA_ENV:-} bash "$INNER"
        st=$?
        [ "$st" != 0 ] && { echo "  CELL FAILED n=${NODES[$i]} $sub (exit $st)"; FAILED=$((FAILED+1)); }
    done
done

n=$(find "$OUT_ROOT" -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
echo
echo "[$(date '+%F %T')] $MODEL_NAME done: failed_cells=$FAILED  npz=$n"
[ "$n" = 0 ] && { echo "ERROR: no dumps written" >&2; exit 1; }
exit 0
