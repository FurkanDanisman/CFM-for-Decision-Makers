#!/bin/bash
#SBATCH --job-name=cmech-dens-score
#SBATCH --output=logs_cmech_score/score_%j.out
#SBATCH --error=logs_cmech_score/score_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#
# CPU ONLY -- no --gres. Scores the ComplexMech density dumps with
# UWYK_Fig3_4/cate_density_metrics.py for BOTH targets.
#
# Memory matters here: the ATE path barycenters every query of a realization,
# and dopfn_native's atom grid can reach ATOM_GRID_MAX bins -> a tau support
# of 2*J-1. At 4096 that is 8191 atoms per query, which OOMs a login node.
#
# Env:
#   ROOT            dump root      (default $SCRATCH/cmech_1d2d)
#   CTX             context        (default 1000)
#   NODES           node counts    (default "5 10 20 30 40 50")
#   SUBSET          nonzero|zero|total   (default total)
#   METHODS         optional subset of method labels
#   TARGETS         "cate ate" (default), or just one of them --
#                   the CATE pass is the slow one and is often
#                   already done
#   DATA_ROOT       benchmark data; --target ate needs it to map each npz back
#                   to its source realization when pooling the two subsets
#   ATOM_GRID_MAX   dopfn_native tau grid cap (default 4096; 1024 is 4x
#                   cheaper and coarser)
#
#   sbatch benchmarks/cluster/submit_cmech_density_score.sbatch

set -uo pipefail
mkdir -p logs_cmech_score

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
ROOT="${ROOT:-$SCRATCH/cmech_1d2d}"
CTX="${CTX:-1000}"
NODES="${NODES:-5 10 20 30 40 50}"
SUBSET="${SUBSET:-total}"
DATA_ROOT="${DATA_ROOT:-$REPO/UWYK_Fig3_4/data}"
export ATOM_GRID_MAX="${ATOM_GRID_MAX:-4096}"

source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
# aggregate_cmech_methods lives in benchmarks/ and --target ate imports it by
# name to resolve which source realization each npz came from.
export PYTHONPATH="$REPO/benchmarks:$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

MD=()
[ -n "${METHODS:-}" ] && MD=(--methods ${METHODS})

echo "[score] root=$ROOT ctx=$CTX nodes=[$NODES] subset=$SUBSET"
echo "        ATOM_GRID_MAX=$ATOM_GRID_MAX  data_root=$DATA_ROOT"

for T in ${TARGETS:-cate ate}; do
    echo "=== target=$T ==="
    python -u "$REPO/UWYK_Fig3_4/cate_density_metrics.py" \
        --root "$ROOT" --context "$CTX" --nodes $NODES \
        --subset "$SUBSET" --target "$T" \
        --data-root "$DATA_ROOT" "${MD[@]}" \
      || echo "WARN: target=$T exited nonzero"
done
echo "[$(date '+%F %T')] done"
