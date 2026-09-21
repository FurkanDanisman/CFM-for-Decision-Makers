#!/bin/bash
#SBATCH --job-name=tauC-dens
#SBATCH --output=logs_density_tauC/eval_%A_%a.out
#SBATCH --error=logs_density_tauC/eval_%A_%a.err
#SBATCH --account=aip-rgrosse
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-11

# NO --partition AND NO --gres ON PURPOSE (see RUN_KILLARNEY.md "Killarney
# specifics"). The trailing _b1.._bN of gpubase_* is a TIME BUCKET: pinning
# gpubase_h100_b1 caps --time at 3:00:00 and makes anything longer a
# SUBMISSION ERROR, not a slower queue. Unset, Slurm routes by --time, and
# <=3h is the shallowest (fastest-draining) bucket -- so if a task outgrows
# the wall clock, WIDEN THE ARRAY rather than lengthening --time.
# The GPU is dropped because the per-query cost here is the numpy tau
# quadrature, not the forwards: h100 has 8-10 nodes against l40s's 168 and a
# --test-only probe scheduled the same job ~1.5h later on h100. The eval still
# uses cuda when it finds it (H.DEVICE), so to take the GPU tier anyway:
#   sbatch --partition=gpubase_h100_b2 --gres=gpu:h100:1 --time=... ...
# Measure before assuming either way: the smoke run below prints s/realization.

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
# no DoPFN model accepts or receives graph information.
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
# still leaves >3x headroom on the ACIC tasks -- but that headroom was measured
# for the THREE UWYK methods. A DoPFN run pays per DOPFN_MODELS entry, and each
# JOINT entry costs about what `joint` above does, so budget the sum. ACIC is
# the long pole (5 realizations x ~481 queries per task, against IHDP's
# 10 x 75); split it finer in the array map below before raising --time.
#
# N_Y0=4096 is set by the JOINT path, not UWYK: UWYK's tail quadrature is at
# 0.99998 by N_Y0=1024, while the joint only reaches 1e-4 mass error at 4096.
#
# MODEL_FAMILY=dopfn runs one method per DOPFN_MODELS entry (four by default,
# see below). Joint entries pay the same N_Y0 tail quadrature as `joint`; 1D
# entries use DoPFN's exact independence convolution. The timings above were
# measured for UWYK only, so check the first DoPFN task before trusting --time.
#
# MODEL_FAMILY=causalpfn works the same way over CAUSALPFN_MODELS: five entries
# by default, TWO of them cpfn2d joints. Each joint pays the full N_Y0 tail
# quadrature -- the dominant per-query cost -- so the default list is roughly
# 2x the quadrature of the old single-joint run, on top of ~2.7x the forwards
# (three 1D heads x two arms + two joints per query chunk, against one 1D head
# and one joint before). The 1D convolutions themselves stay cheap even at
# K=1024. Nothing here has been timed on a compute node: WATCH THE FIRST ACIC
# TASK (array id 10 or 11, ~481 queries x 5 realizations) against the 3h cap.
# If it runs out, the run is resumable -- see RESUMABLE below -- so re-submitting
# the same array finishes it; widen the array before touching --time.
#
# Submit:  sbatch benchmarks/cluster/submit_density_tauC.sbatch
# CausalPFN: MODEL_FAMILY=causalpfn sbatch benchmarks/cluster/submit_density_tauC.sbatch
# Smoke:   sbatch --array=0 benchmarks/cluster/submit_density_tauC.sbatch
#          (task 0 also runs the model-free gates and aborts if any fail)
# 1 real:  REAL_END=1 sbatch --array=0 benchmarks/cluster/submit_density_tauC.sbatch
# Then:    python benchmarks/eval_graph2d/summarize_density_tauC.py results_density_tauC/<JOBID>
# Point errors are now computed from the same logits, with no extra forwards.
# SAVE_PREDICTIONS=1 (default) retains raw outputs under <OUT>/predictions for
# future CPU checks. Use a fresh OUT_ROOT to keep the previous run intact.
# RESUMABLE: the eval skips any realization whose shard already exists, so a
# task killed at the wall clock costs one realization and re-submitting the
# same array finishes the rest. The flip side is that it trusts the shard and
# not the config that made it -- change a checkpoint, DOPFN_MODELS,
# CAUSALPFN_MODELS, ANC_TAG or N_Y0 and you MUST use a fresh OUT_ROOT, or stale
# shards are silently kept.

set -euo pipefail
mkdir -p logs_density_tauC

REPO="${REPO:-/home/lukez/projects/aip-rgrosse/lukez/CFM-for-Decision-Makers}"

export UWYK="${UWYK:-$REPO/g4cfm}"
export CAUSALPFN="${CAUSALPFN:-/home/lukez/projects/aip-rgrosse/lukez/CausalPFN}"

# Select uwyk (legacy default), dopfn, causalpfn, or all. The DoPFN checkout must include
# its pretrained artifacts even when only our checkpoints are listed below.
# Example: MODEL_FAMILY=dopfn DOPFN_ROOT=/path/to/Do-PFN sbatch ...
export MODEL_FAMILY="${MODEL_FAMILY:-uwyk}"
export DOPFN_ROOT="${DOPFN_ROOT:-${DOPFN:-}}"

# ── DoPFN models (MODEL_FAMILY=dopfn or all) ───────────────────────────────
# Every entry becomes its own density row, dopfn_<name>, and all of them see the
# same context. Add, drop or repoint models here; nothing else needs editing.
#   native          the Do-PFN library regressor under $DOPFN_ROOT; no path
#   <name>=<ckpt>   a checkpoint. Its kind is read from the file, not the name:
#                     training_dopfn_repro dopfn_1d  -> 1D bars, arms convolved
#                                                       under independence
#                     training_dopfn_repro joint_2d  -> J x J joint
#                     training_dopfn_base (dopfn_bb_*) -> J x J joint
# Replace the whole list for one submission with DOPFN_MODELS (commas, no spaces):
#   DOPFN_MODELS="native,bb_j10=$REPO/Required_checkpoints/old/dopfn_bb_j10_step_150000.pt" sbatch ...
DOPFN_MODEL_LIST=(
    "native"
    "repro_1d_J10=$REPO/Required_checkpoints/new/dopfn_repro_1d_J10_step150000.pt"
    "repro_1d_J100=$REPO/Required_checkpoints/new/dopfn_repro_1d_J100_step150000.pt"
    "repro_joint2d=$REPO/Required_checkpoints/new/dopfn_repro_joint2d_step150000.pt"
)
export DOPFN_MODELS="${DOPFN_MODELS:-$(IFS=,; echo "${DOPFN_MODEL_LIST[*]}")}"
export DOPFN_QUERY_CHUNK="${DOPFN_QUERY_CHUNK:-20}"

# ── CausalPFN models (MODEL_FAMILY=causalpfn or all) ───────────────────────
# Same contract as DOPFN_MODELS above: every entry becomes its own density row,
# causalpfn_<name>, and all of them see the same context, the same pooled
# standardization and the same scoring axis. Two differences from DoPFN:
#   * every entry needs a checkpoint -- there is no library model to name on
#     its own, so a bare word is rejected;
#   * a 1D head and a cpfn2d joint are told apart by reading the FILE
#     (model_type/edges vs bin_edges), never the name. The _1d / _2d suffixes
#     below are for the summary tables' benefit only.
# Replace the whole list for one submission with CAUSALPFN_MODELS (commas, no
# spaces):
#   CAUSALPFN_MODELS="j32_random_2d=$CPFN_DIR/cpfn2d_j32_random_step_50000.pt" sbatch ...
CPFN_DIR="${CPFN_DIR:-$REPO/Required_checkpoints/new_checkpoints}"
CAUSALPFN_MODEL_LIST=(
    "j32_random_2d=$CPFN_DIR/cpfn2d_j32_random_step_50000.pt"
    "j32_eta0_y01_2d=$CPFN_DIR/cpfn2d_j32_eta0_y01_step50000.pt"
    "j1024_headrand_1d=$CPFN_DIR/cpfn1d_j1024_headrand_step_50000.pt"
    "j32_1d=$CPFN_DIR/cpfn1d_j32_step50000.pt"
    "botharms_1d=$CPFN_DIR/cpfn1d_botharms_step50000.pt"
)
export CAUSALPFN_MODELS="${CAUSALPFN_MODELS:-$(IFS=,; echo "${CAUSALPFN_MODEL_LIST[*]}")}"
export CAUSALPFN_QUERY_CHUNK="${CAUSALPFN_QUERY_CHUNK:-512}"
# NOT SET ON PURPOSE: CAUSALPFN_CKPT / CAUSALPFN_JOINT_CKPT. Setting either one
# puts the eval back on its pre-list two-row layout (causalpfn_native +
# causalpfn_joint) and ignores everything above.

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
# The +/-1 matrix is handed only to the UWYK pair. No DoPFN model receives
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
    dopfn|causalpfn) ;;
    *) echo "FATAL: MODEL_FAMILY must be uwyk, dopfn, causalpfn, or all" >&2; exit 1 ;;
esac
if [[ "$MODEL_FAMILY" == causalpfn || "$MODEL_FAMILY" == all ]]; then
    inputs+=("$CAUSALPFN/src/causalpfn/models/model.py")
    IFS=, read -r -a _causalpfn_entries <<< "$CAUSALPFN_MODELS"
    for entry in "${_causalpfn_entries[@]}"; do
        [[ "$entry" == *=* ]] || {
            echo "FATAL: CAUSALPFN_MODELS entry '$entry' must be name=checkpoint" >&2
            exit 1; }
        inputs+=("${entry#*=}")
    done
fi
if [[ "$MODEL_FAMILY" == dopfn || "$MODEL_FAMILY" == all ]]; then
    [ -n "$DOPFN_ROOT" ] || { echo "FATAL: set DOPFN_ROOT" >&2; exit 1; }
    inputs+=("$DOPFN_ROOT/artifacts/dopfn_model.pkl"
             "$DOPFN_ROOT/artifacts/dopfn_config.pkl"
             "$DOPFN_ROOT/artifacts/model_submitit_0ccc_id_171b69db_epoch_-1.cpkt")
    IFS=, read -r -a _dopfn_entries <<< "$DOPFN_MODELS"
    for entry in "${_dopfn_entries[@]}"; do
        if [[ "$entry" == *=* ]]; then inputs+=("${entry#*=}"); fi
    done
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
elif [[ "$MODEL_FAMILY" == causalpfn ]]; then
    GRAPH_DESC="UWYK_graph=n/a CausalPFN_graph=none"
elif [[ "$MODEL_FAMILY" == all ]]; then
    GRAPH_DESC="UWYK_graph=$ANC_TAG DoPFN_graph=none CausalPFN_graph=none"
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
    python -m unittest discover -s "$REPO/benchmarks/eval_graph2d" -p 'test_density_causalpfn.py'
    python -m unittest discover -s "$REPO/benchmarks/methods" -p 'test_dopfn_compat.py'
    echo "--- gates passed ---"
fi

python -u "$REPO/benchmarks/eval_graph2d/eval_density_tauC.py"
echo "[$(date '+%F %T')] done"
