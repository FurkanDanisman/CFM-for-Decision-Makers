#!/bin/bash
#SBATCH --job-name=tauC-dens
#SBATCH --output=logs_density_tauC/eval_%A_%a.out
#SBATCH --error=logs_density_tauC/eval_%A_%a.err
#SBATCH --partition=gpubase_h100_b1
#SBATCH --account=aip-rgrosse
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-11

# Tier-C density eval: p(tau | x) for UWYK-1D vs Joint-2D on IHDP + ACIC.
# v1 = RAW path only (no MALC). Spec: density_eval_pipeline.md.
#
# PARALLELISM: the array splits over REALIZATIONS, not over models.
#   0-9   IHDP realizations in blocks of 10   (100 total)
#   10-11 ACIC realizations in blocks of 5    (10 total)
# All selected models run in ONE process per task, on purpose. Splitting them
# would be safe -- every shared preprocessing step is deterministic given
# (dataset, r, EVAL_MAX_CONTEXT, EVAL_CONTEXT_SEED) -- but it buys only 2x and
# it trades a *guarantee* that models saw byte-identical context and outcome
# scaling for an *assumption* that separate processes recompute them identically.
# The UWYK pair receives the selected adjacency (the wrapper may propagate it);
# neither DoPFN model accepts or receives graph information.
# Realizations are independent by
# construction, so splitting there is both wider (12x) and strictly safer.
#
# Cost. The tau grid is now 12001 points (dtau=0.0005, knot-aligned: it
# divides both models' bin widths, 0.002 for UWYK's K=1000 bars and 0.0625 for
# the J=32 joint, so the trapezoid in kl/l2/mass is exact on the piecewise-
# linear interiors). Per QUERY the eval runs three methods -- uwyk_native,
# uwyk_matched, joint -- so budget the sum, not one method:
#
#   N_Y0     s/query (3 methods)      int p(tau), joint path
#   1024     0.32                     0.99922
#   2048     0.67                     0.99922
#   4096     1.60                     0.99990   <- default
#   8192     3.70                     0.99995
#
# Those are login-node numbers. The 5253193 logs put the OLD code (600-point
# grid, N_Y0=2048) at 0.22 s/query on a compute node vs 0.31 s/query measured
# the same way here, i.e. the compute node runs ~1.4x faster, so expect
# ~1.1 s/query at N_Y0=4096:
#   IHDP task = 10 real x 75 q  ~= 14 min  + model load/forwards
#   ACIC task =  5 real x 480 q ~= 44 min  + model load/forwards
# That is ~5x the old cost per query, not "about the same"; --time=03:00:00
# still leaves >3x headroom on the ACIC tasks.
#
# N_Y0=4096 is set by the JOINT path, not UWYK: UWYK's tail quadrature is at
# 0.99998 by N_Y0=1024, while the joint only reaches 1e-4 mass error at 4096.
#
# Submit:  sbatch benchmarks/cluster/submit_density_tauC.sbatch
# Smoke:   sbatch --array=0 benchmarks/cluster/submit_density_tauC.sbatch
#          (task 0 also runs the model-free gates and aborts if any fail)
# 1 real:  REAL_END=1 sbatch --array=0 benchmarks/cluster/submit_density_tauC.sbatch
# Then:    python benchmarks/eval_graph2d/summarize_density_tauC.py results_density_tauC/<JOBID>
# Point errors are now computed from the same logits, with no extra forwards.
# SAVE_PREDICTIONS=1 (default) retains raw outputs under <OUT>/predictions for
# future CPU checks. Use a fresh OUT_ROOT to keep the previous run intact.

set -euo pipefail
mkdir -p logs_density_tauC

REPO="${REPO:-/home/lukez/projects/aip-rgrosse/lukez/CFM-for-Decision-Makers}"

export UWYK="${UWYK:-$REPO/g4cfm}"
export CAUSALPFN="${CAUSALPFN:-/home/lukez/projects/aip-rgrosse/lukez/CausalPFN}"

# Select uwyk (legacy default), dopfn, or all. The DoPFN checkout must include
# its pretrained artifacts even when loading our joint checkpoint.
# Example: MODEL_FAMILY=dopfn DOPFN_ROOT=/path/to/Do-PFN sbatch ...
export MODEL_FAMILY="${MODEL_FAMILY:-uwyk}"
export DOPFN_ROOT="${DOPFN_ROOT:-${DOPFN:-}}"
export DOPFN_JOINT_CKPT="${DOPFN_JOINT_CKPT:-$REPO/Required_checkpoints/dopfn_bb_j10_step_150000.pt}"
export DOPFN_QUERY_CHUNK="${DOPFN_QUERY_CHUNK:-20}"

# ── checkpoints ────────────────────────────────────────────────────────────
# Joint-2D: our graph2d head. step_50000 matches the "Ours V6A - 5206104" row
# in LATEST_RESULTS.md; override CKPT to compare step_58000.
export CKPT="${CKPT:-$REPO/Required_checkpoints/graph2d_step_50000.pt}"

# UWYK-1D: the Table-1 full-conditioning checkpoint. NOT
# Required_checkpoints/uwyk_reproduce_best_model.pt -- that file is a copy of
# experiments/checkpoints/model/best_model.pt, a different experiment's run
# (upstream REPRODUCE_REALCAUSE_RESULTS.md says explicitly not to use it), and
# it ships without the config yaml the loader requires.
UWYK_CKPT_DIR="$UWYK/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0"
export UWYK_CKPT="${UWYK_CKPT:-$UWYK_CKPT_DIR/best_model.pt}"
export UWYK_CFG="${UWYK_CFG:-$UWYK_CKPT_DIR/best_model_config.yaml}"

# ── eval configuration ─────────────────────────────────────────────────────
# Must match whatever the PEHE table you are comparing against used, or the
# two are not measuring the same conditioning.
# ANC_MODE selects the family H.build_mode_list emits; ANC_TAG picks one
# matrix out of it. They must be set as a PAIR:
#   ANC_MODE=v6a_only ANC_TAG=v6a   no +1 edges; only the -1s implied by
#                                   unconfoundedness (diag, Y->T, Y->X, T->X)
#   ANC_MODE=full     ANC_TAG=anc   build_anc_full: T->Y, X_i->T, X_i->Y
#                                   asserted +1, then propagate fills the -1s
# The +/-1 matrix is handed only to the UWYK pair. Neither DoPFN model receives
# it. ANC_TAG is ignored in a DoPFN-only run.
export ANC_MODE="${ANC_MODE:-v6a_only}"
export ANC_TAG="${ANC_TAG:-v6a}"
# UWYK's config caps training context at 1000; feeding more is outside what it
# saw in training. IHDP's 672 train rows are under the cap anyway (no-op);
# ACIC's ~4.3k are not, which is exactly why the truth loader must be handed
# the POST-subsample y -- the driver does that.
export EVAL_MAX_CONTEXT="${EVAL_MAX_CONTEXT-1000}"
export N_Y0="${N_Y0:-4096}"
export ACIC_CACHE_DIR="${ACIC_CACHE_DIR:-$REPO/data/acic_cache}"

export OUT_ROOT="${OUT_ROOT:-$REPO/results_density_tauC/$SLURM_ARRAY_JOB_ID}"

inputs=()
case "$MODEL_FAMILY" in
    uwyk|all) inputs+=("$CKPT" "$UWYK_CKPT" "$UWYK_CFG") ;;
    dopfn) ;;
    *) echo "FATAL: MODEL_FAMILY must be uwyk, dopfn, or all" >&2; exit 1 ;;
esac
if [[ "$MODEL_FAMILY" == dopfn || "$MODEL_FAMILY" == all ]]; then
    [ -n "$DOPFN_ROOT" ] || { echo "FATAL: set DOPFN_ROOT" >&2; exit 1; }
    inputs+=("$DOPFN_JOINT_CKPT" "$DOPFN_ROOT/artifacts/dopfn_model.pkl"
             "$DOPFN_ROOT/artifacts/dopfn_config.pkl"
             "$DOPFN_ROOT/artifacts/model_submitit_0ccc_id_171b69db_epoch_-1.cpkt")
fi
for f in "${inputs[@]}"; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done
# load_acic_truth reads x.csv purely for a row count, but will reach out to
# raw.githubusercontent.com if it is not cached -- which fails on a compute
# node with no egress. Fetch it from a login node if this trips.
[ -f "$ACIC_CACHE_DIR/x.csv" ] || {
    echo "FATAL: $ACIC_CACHE_DIR/x.csv missing; ACIC truth would try to download." >&2
    echo "  curl -sSfL https://raw.githubusercontent.com/BiomedSciAI/causallib/master/causallib/datasets/data/acic_challenge_2016/x.csv -o $ACIC_CACHE_DIR/x.csv" >&2
    exit 1; }

# ── array -> (dataset, realization slice) ──────────────────────────────────
# Capture caller overrides BEFORE the task-id mapping computes its own values,
# so `REAL_END=1 sbatch --array=0 ...` actually trims the slice.
_RS_IN="${REAL_START:-}"
_RE_IN="${REAL_END:-}"
_DS_IN="${DATASET:-}"

T=$SLURM_ARRAY_TASK_ID
if   [ "$T" -lt 10 ]; then
    DATASET=IHDP;  REAL_START=$(( T * 10 ));        REAL_END=$(( T * 10 + 10 ))
elif [ "$T" -lt 12 ]; then
    DATASET=ACIC;  REAL_START=$(( (T - 10) * 5 ));  REAL_END=$(( (T - 10) * 5 + 5 ))
else
    echo "invalid task id: $T" >&2; exit 2
fi
DATASET="${_DS_IN:-$DATASET}"
REAL_START="${_RS_IN:-$REAL_START}"
REAL_END="${_RE_IN:-$REAL_END}"
export DATASET REAL_START REAL_END
export OUT="$OUT_ROOT/$DATASET"

source "$REPO/.venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
# shims/ = sitecustomize denylist that keeps causalpfn's __init__ from dragging
# in deps we do not have; same one the graph2d and uwyk1d evals use.
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$MODEL_FAMILY" == dopfn ]]; then
    GRAPH_DESC="UWYK_graph=n/a DoPFN_graph=none"
elif [[ "$MODEL_FAMILY" == all ]]; then
    GRAPH_DESC="UWYK_graph=$ANC_TAG DoPFN_graph=none"
else
    GRAPH_DESC="UWYK_graph=$ANC_TAG"
fi
echo "[$(date '+%F %T')] tauC density  dataset=$DATASET  r=[$REAL_START,$REAL_END)" \
     " family=$MODEL_FAMILY  $GRAPH_DESC  N_Y0=$N_Y0  OUT=$OUT"

# The model-free validation gates are cheap (~2 min) and every number below is
# meaningless if they fail. Run them once, on the first task only.
if [ "$T" -eq 0 ]; then
    echo "--- validation gates ---"
    python -u "$REPO/benchmarks/eval_graph2d/test_density_common.py" \
        || { echo "FATAL: density gates failed; not running the eval." >&2; exit 3; }
    python -m unittest discover -s "$REPO/benchmarks/eval_graph2d" -p 'test_density_dopfn.py'
    python -m unittest discover -s "$REPO/benchmarks/methods" -p 'test_dopfn_compat.py'
    echo "--- gates passed ---"
fi

python -u "$REPO/benchmarks/eval_graph2d/eval_density_tauC.py"
echo "[$(date '+%F %T')] done"
