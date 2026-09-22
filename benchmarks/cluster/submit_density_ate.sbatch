#!/bin/bash
#SBATCH --job-name=ate-dens
#SBATCH --output=logs_density_ate/eval_%A_%a.out
#SBATCH --error=logs_density_ate/eval_%A_%a.err
#SBATCH --account=aip-rgrosse
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-7

# NO --partition AND NO --gres, same reasoning as submit_density_tauC.sbatch --
# and here it is not even a judgement call: this job loads NO model and runs NO
# forward pass. It is pure numpy over prediction dumps that eval_density_tauC.py
# already wrote, so it is CPU-only by construction. <=3h is the shallowest
# (fastest-draining) time bucket; if a task outgrows the wall clock, WIDEN THE
# ARRAY rather than lengthening --time.
#
# Tier: p_ATE(tau), the 2-Wasserstein barycenter over queries of p(tau | x_q).
# Feeds the f_ATE column of Table 2 in FOR_FURKAN.md.
#
# NOT A SHORTCUT. The per-query tau densities are rebuilt with the SAME exact
# routines and the SAME 12001-point TAU_CENTERS grid the tau tier used --
# uwyk/dopfn/causalpfn_tau_density for 1D heads, joint_tau_density's full
# 9-region mixture (tail quadrature at the dump's own N_Y0) for 2D heads. The
# cheap alternatives were rejected on purpose: an anti-diagonal sum of the
# joint interior drops the tail mixture, and convolving gridded marginals is
# not the analytic convolution f_tau was scored with. So f_ATE and f_tau rest
# on identical densities.
#
# PARALLELISM: the array splits over (run, dataset); realizations inside a task
# go through a multiprocessing pool of SLURM_CPUS_PER_TASK workers. Realizations
# are independent by construction -- each prediction dump is self-contained and
# carries its own context, scaling and truth -- so this is safe. BLAS threads
# are pinned to 1 below: the pool already owns every core, and letting numpy
# also thread would oversubscribe them.
#
# COST. Measured on a laptop: ~0.24 s/query for uwyk_tau_density and ~0.17 s for
# joint_tau_density at N_Y0=4096; the histogram convolutions (causalpfn, dopfn
# 1D) are much cheaper. Budget the sum over the methods in each dump:
#   causalpfn IHDP  90 real x  75 q x 5 methods  ~= 1.7 h at 1 worker, ~13 min at 8
#   causalpfn ACIC  10 real x 481 q x 5 methods  ~= 1.2 h at 1 worker, ~ 9 min at 8
#   dopfn     IHDP  90 real x  75 q x 4 methods  ~= 1.4 h at 1 worker, ~11 min at 8
#   uwyk_*    IHDP 100 real x  75 q x 2 methods  ~= 45 min at 1 worker, ~ 6 min at 8
# Every task fits 03:00:00 even single-threaded, so the wall clock is not the
# binding constraint; the pool is there to keep the queue short.
#
# SUBMIT (the mkdir is NOT optional on a first run -- Slurm resolves
# --output/--error at SUBMIT time, before this script's own mkdir runs, so a
# missing logs_density_ate/ fails the task before any of it executes):
#   mkdir -p logs_density_ate
#   sbatch benchmarks/cluster/submit_density_ate.sbatch
# SMOKE (one realization per task, same code path, ~1 min/task):
#   SMOKE=1 sbatch benchmarks/cluster/submit_density_ate.sbatch
# ONE COMBO ONLY:
#   sbatch --array=4 benchmarks/cluster/submit_density_ate.sbatch
# THEN:
#   python benchmarks/eval_graph2d/summarize_density_ate.py --dataset IHDP

set -euo pipefail
mkdir -p logs_density_ate

REPO="${REPO:-/home/lukez/projects/aip-rgrosse/lukez/CFM-for-Decision-Makers}"
cd "$REPO"

# Prediction-dump shards written by earlier eval_density_tauC.py runs. These are
# the SAME shards that back the f_tau column, so the two tiers cannot drift.
#   run name    shard id    what it holds
TAUC_ROOT="${TAUC_ROOT:-$REPO/results_density_tauC}"
COMBOS=(
    "uwyk_v3a:5312882:IHDP"      # UWYK-1D (anc=v3a) + graph2d Joint-2D
    "uwyk_v3a:5312882:ACIC"
    "uwyk_noanc:5312884:IHDP"    # UWYK-1D (anc=noanc) + graph2d Joint-2D
    "uwyk_noanc:5312884:ACIC"
    "dopfn:60508900:IHDP"        # DoPFN native + repro_1d_J10/J100 + repro_joint2d
    "dopfn:60508900:ACIC"
    "causalpfn:5571187:IHDP"     # all five CausalPFN checkpoints
    "causalpfn:5571187:ACIC"
)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
if (( IDX >= ${#COMBOS[@]} )); then
    echo "[ate] array index $IDX past the end of COMBOS (${#COMBOS[@]} entries)" >&2
    exit 1
fi
IFS=: read -r RUN SHARD DATASET <<< "${COMBOS[$IDX]}"

PREDICTIONS="$TAUC_ROOT/$SHARD/$DATASET/predictions"
OUT_ROOT="${OUT_ROOT:-$REPO/results_density_ate}"
OUT="$OUT_ROOT/$RUN/$DATASET"

# Fail here, not 40 minutes in, if the dumps are missing or empty.
if [[ ! -d "$PREDICTIONS" ]]; then
    echo "[ate] no prediction dir: $PREDICTIONS" >&2
    echo "[ate] eval_density_tauC.py writes it when SAVE_PREDICTIONS=1 (the default)." >&2
    exit 1
fi
N_DUMPS=$(find "$PREDICTIONS" -maxdepth 1 -name "${DATASET}_r[0-9]*.npz" | wc -l)
if (( N_DUMPS == 0 )); then
    echo "[ate] $PREDICTIONS holds no ${DATASET}_r*.npz dumps" >&2
    exit 1
fi

source "$REPO/.venv/bin/activate"
# One BLAS thread per process: the pool below already owns every core.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
# --limit 0 means all dumps; a scalar also works with Bash 3.2's nounset mode,
# which treats an empty array as unset when expanding it.
LIMIT=0
[[ "${SMOKE:-0}" == "1" ]] && LIMIT=1

echo "[ate] task $IDX  run=$RUN shard=$SHARD dataset=$DATASET"
echo "[ate] dumps=$N_DUMPS  workers=$WORKERS  out=$OUT"

mkdir -p "$OUT"
python -u "$REPO/benchmarks/eval_graph2d/eval_density_ate.py" \
    --predictions "$PREDICTIONS" \
    --out "$OUT" \
    --workers "$WORKERS" \
    --limit "$LIMIT"

echo "[ate] done: $(find "$OUT" -name "${DATASET}_r[0-9]*.npz" | wc -l) realizations written"
