#!/bin/bash
#SBATCH --job-name=ate-malcT
#SBATCH --output=logs_density_ate_malcT/eval_%A_%a.out
#SBATCH --error=logs_density_ate_malcT/eval_%A_%a.err
#SBATCH --account=aip-rgrosse
#SBATCH --time=02:59:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --array=0-23

# Repeat the existing MALC-T fits, retaining each query's density long enough
# to compute its W2-barycenter ATE. No model inference or GPU is needed.
# Four families x (four 25-realization IHDP slices + two 5-realization ACIC
# slices). This keeps the costly Do-PFN jobs inside the three-hour bucket.
# Existing MALC-T files remain the reference; new outputs go to a separate tree.
#
# From the repository root:
#   mkdir -p logs_density_ate_malcT
#   REPO="$PWD" sbatch benchmarks/cluster/submit_density_ate_malcT.sbatch
# One realization smoke test:
#   REPO="$PWD" SMOKE=1 sbatch --array=0 benchmarks/cluster/submit_density_ate_malcT.sbatch
# Then (once all outputs have been synced to the checkout containing the doc):
#   python benchmarks/eval_graph2d/summarize_density_malcT.py --write

set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
source "$REPO/.venv/bin/activate"

SHARDS=(5312882 5312884 dopfn_refresh 5571187)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if (( IDX < 0 || IDX >= 24 )); then
    echo "Invalid task index: $IDX" >&2
    exit 2
fi
SHARD="${SHARDS[$((IDX / 6))]}"
SLICE=$((IDX % 6))
if (( SLICE < 4 )); then
    DATASET=IHDP
    export REAL_START=$((SLICE * 25)) REAL_END=$((SLICE * 25 + 25))
else
    DATASET=ACIC
    export REAL_START=$(((SLICE - 4) * 5)) REAL_END=$(((SLICE - 4) * 5 + 5))
fi
if [[ "${SMOKE:-0}" == 1 ]]; then
    export REAL_END=$((REAL_START + 1))
fi

export DUMPS="$REPO/results_density_tauC/$SHARD/$DATASET/predictions"
export OUT="${OUT_ROOT:-$REPO/results_density_tauC_malcT_ate}/$SHARD/$DATASET"
export REFERENCE_ROOT="$REPO/results_density_tauC_malcT/$SHARD"
# Optimizer/library environments can change a seeded fit. Record any drift;
# the summarizer switches BOTH CATE and ATE to a complete new realization set.
export REFERENCE_STRICT=0
export COMPUTE_ATE=1
export MALC_B=1000 MALC_K=1 MALC_SEED=0 MALC_N_TAU=12001 N_Y0=4096
export DOPFN_MALC_BINS=1024 MODEL_FAMILY=auto
export N_WORKERS="${N_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1

[[ -d "$DUMPS" && -d "$REFERENCE_ROOT/$DATASET" ]] || {
    echo "Missing prediction dumps or MALC-T reference for $SHARD/$DATASET" >&2
    exit 1
}
echo "[ate-malcT] shard=$SHARD dataset=$DATASET realizations=[$REAL_START,$REAL_END) out=$OUT"
python -u benchmarks/eval_graph2d/eval_density_tauC_malcT.py
