#!/bin/bash
#SBATCH --job-name=tauC-repair
#SBATCH --output=logs_density_tauC_repair/repair_%A_%a.out
#SBATCH --error=logs_density_tauC_repair/repair_%A_%a.err
#SBATCH --account=aip-rgrosse
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-2

# Back-fill the realizations missing from the tau-C prediction shards, so that
# every method is scored on the SAME realization set before any MALC arm runs.
#
# THE PROBLEM THIS SOLVES. The four shards do not cover the same realizations:
#
#     family        IHDP          ACIC        missing
#     UWYK v3a      100  (0-99)   10 (0-9)    --
#     UWYK noanc    100  (0-99)   10 (0-9)    --
#     Do-PFN         90           8 (2-9)     IHDP 80-84, 90-94; ACIC 0-1
#     CausalPFN      90 (10-99)   10 (0-9)    IHDP 0-9
#
# so the common subset is 80 IHDP / 8 ACIC, and every cross-family mean in the
# tau tables is currently taken over a different set of datasets. The gaps are
# not random: Do-PFN's are the tails of array tasks 8 and 9 plus the head of
# task 10, and CausalPFN's are exactly array task 0 -- three tasks that died,
# not a deliberate subset. Filling them makes all 11 rows 100 IHDP / 10 ACIC.
#
# RUN THIS BEFORE THE MALC ARMS. Both submit_density_tauC_malc.sbatch (J) and
# submit_density_tauC_malcT.sbatch (T) read these prediction dumps, so a repair
# afterwards would mean re-running them. Order:
#
#     1. this job            fills the prediction shards in place
#     2. summarize_density_tauC.py   confirm 100/10 everywhere
#     3. submit_density_tauC_malcT.sbatch   (and _malc for J)
#
# NO --gres, BY THE SAME REASONING AS THE RAW JOB. The per-query cost is the
# numpy tau quadrature, not the forwards, and the eval still uses cuda if it
# finds one (H.DEVICE). To take the GPU tier anyway:
#   sbatch --partition=gpubase_h100_b2 --gres=gpu:h100:1 ...
#
# ── WHY THIS IS SAFE TO RUN INTO AN EXISTING SHARD DIRECTORY ───────────────
# eval_density_tauC.py skips any realization whose shard already exists, so
# this job is purely additive: it cannot overwrite one of the 90 realizations
# that are already there, and the REAL_START/REAL_END ranges below simply let
# that skip do the set arithmetic (task 0 asks for 80-94 and the eval silently
# steps over the ten that exist).
#
# The flip side, which the raw sbatch warns about and which matters far more
# here: the resume logic TRUSTS THE SHARD, NOT THE CONFIG THAT MADE IT. These
# shards were written months ago on other clusters -- their own metadata
# records three different repo roots (/home/lukez/..., /home/lukez/projects/
# aip-rgrosse/lukez/..., /project/6105522/lukez/...). Re-running realization 80
# with a different checkpoint, a different Do-PFN library version or a
# different context seed would write a shard indistinguishable from its
# neighbours and comparable to none of them.
#
# Do-PFN makes that unverifiable from metadata alone: its `native` row is the
# pretrained LIBRARY model, recorded as dopfn_sources[0] == 'library' with no
# version or hash to check.
#
# SO EVERY TASK FIRST RE-RUNS ONE REALIZATION THAT ALREADY EXISTS into a
# scratch directory and diffs the predictions against the stored copy. Passing
# is evidence that this environment reproduces the original run; failing stops
# the repair before it contaminates the shard. REPAIR_PROBE=0 skips it -- do
# that only when you have already proved reproduction for that family in this
# same environment.
#
# The config below is the harness DEFAULT set, which is what the existing
# shards recorded: EVAL_CONTEXT_SEED=1, Y_SCALING=minmax, STD_TARGET=0.3,
# BIAS_EDGE_SCALE=1.0, X_CLIP_QUANTILE unset, N_Y0=4096, anc_tag=none. Verified
# against results_density_tauC/60508900 and .../5571187. Do not change any of
# them here -- a change makes the new realizations incomparable to the old ones
# and the probe will (correctly) refuse to proceed.
#
# Cost, extrapolated from the raw job's ~1.1 s/query for three UWYK methods and
# scaled by model count (Do-PFN 4 entries, CausalPFN 5 with two joints):
#     task 0  probe 1 + fill 10 IHDP realizations x 75 q      ~30 min
#     task 1  probe 1 + fill  2 ACIC realizations x ~481 q    ~50 min
#     task 2  probe 1 + fill 10 IHDP realizations x 75 q      ~40 min
# All inside the 3h bucket. The job is resumable for the same reason the raw
# one is, so a task that hits the wall clock costs one realization.
#
# Submit:  sbatch benchmarks/cluster/submit_density_tauC_repair.sbatch
# Probe only, no writes into the shard:
#          REPAIR_DRY_RUN=1 sbatch benchmarks/cluster/submit_density_tauC_repair.sbatch
# One family:
#          sbatch --array=2 benchmarks/cluster/submit_density_tauC_repair.sbatch
# Then:    python benchmarks/eval_graph2d/summarize_density_tauC.py \
#            results_density_tauC/<SHARD>

set -euo pipefail
mkdir -p logs_density_tauC_repair

REPO="${REPO:-/home/lukez/projects/aip-rgrosse/lukez/CFM-for-Decision-Makers}"

export UWYK="${UWYK:-$REPO/g4cfm}"
export CAUSALPFN="${CAUSALPFN:-/home/lukez/projects/aip-rgrosse/lukez/CausalPFN}"
export DOPFN_ROOT="${DOPFN_ROOT:-${DOPFN:-$REPO/Do-PFN}}"

# The shard directories being repaired -- the raw jobs' OUT_ROOTs.
DOPFN_SHARD="${DOPFN_SHARD:-$REPO/results_density_tauC/60508900}"
CPFN_SHARD="${CPFN_SHARD:-$REPO/results_density_tauC/5571187}"

REPAIR_PROBE="${REPAIR_PROBE:-1}"
REPAIR_DRY_RUN="${REPAIR_DRY_RUN:-0}"
PROBE_ROOT="${PROBE_ROOT:-${SLURM_TMPDIR:-/tmp}/tauC_repair_probe_$SLURM_ARRAY_JOB_ID}"

# ── model lists: must match what the shards recorded ───────────────────────
DOPFN_MODEL_LIST=(
    "native"
    "repro_1d_J10=$REPO/Required_checkpoints/new/dopfn_repro_1d_J10_step150000.pt"
    "repro_1d_J100=$REPO/Required_checkpoints/new/dopfn_repro_1d_J100_step150000.pt"
    "repro_joint2d=$REPO/Required_checkpoints/new/dopfn_repro_joint2d_step150000.pt"
)
export DOPFN_MODELS="${DOPFN_MODELS:-$(IFS=,; echo "${DOPFN_MODEL_LIST[*]}")}"
export DOPFN_QUERY_CHUNK="${DOPFN_QUERY_CHUNK:-20}"

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

# ── eval configuration: harness defaults, matching the shards ──────────────
export EVAL_CONTEXT_SEED="${EVAL_CONTEXT_SEED:-1}"
export EVAL_MAX_CONTEXT="${EVAL_MAX_CONTEXT-1000}"
export Y_SCALING="${Y_SCALING:-minmax}"
export STD_TARGET="${STD_TARGET:-0.3}"
export BIAS_EDGE_SCALE="${BIAS_EDGE_SCALE:-1.0}"
export N_Y0="${N_Y0:-4096}"
export SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-1}"
export ACIC_CACHE_DIR="${ACIC_CACHE_DIR:-$REPO/data/acic_cache}"

# ── array -> (family, dataset, realization slice, shard, probe) ────────────
# The ranges span the gaps; the eval's skip-if-exists steps over the
# realizations already present (task 0 asks 80-94 and computes only 80-84,
# 90-94). PROBE_R is a realization that ALREADY EXISTS in that shard.
#
# EXPECT is the realization count that dataset must hold when the task
# finishes. It is what makes this job safe to chain: a downstream
# --dependency=afterok gates on exit status only, so a task that filled just
# half its gap and exited 0 would let the MALC arms run on a still-uneven set.
# Asserting the count turns that into a failure the dependency can see.
T=$SLURM_ARRAY_TASK_ID
case "$T" in
    0) FAMILY=dopfn;     DATASET=IHDP; RS=80; RE=95; SHARD="$DOPFN_SHARD"; PROBE_R=0;  EXPECT=100 ;;
    1) FAMILY=dopfn;     DATASET=ACIC; RS=0;  RE=2;  SHARD="$DOPFN_SHARD"; PROBE_R=2;  EXPECT=10  ;;
    2) FAMILY=causalpfn; DATASET=IHDP; RS=0;  RE=10; SHARD="$CPFN_SHARD";  PROBE_R=10; EXPECT=100 ;;
    *) echo "invalid task id: $T" >&2; exit 2 ;;
esac

export MODEL_FAMILY="$FAMILY"
export DATASET

# ── input gates (same set the raw job checks) ──────────────────────────────
inputs=()
if [[ "$FAMILY" == causalpfn ]]; then
    inputs+=("$CAUSALPFN/src/causalpfn/models/model.py")
    IFS=, read -r -a _entries <<< "$CAUSALPFN_MODELS"
    for entry in "${_entries[@]}"; do
        [[ "$entry" == *=* ]] || {
            echo "FATAL: CAUSALPFN_MODELS entry '$entry' must be name=checkpoint" >&2
            exit 1; }
        inputs+=("${entry#*=}")
    done
else
    [ -n "$DOPFN_ROOT" ] || { echo "FATAL: set DOPFN_ROOT" >&2; exit 1; }
    inputs+=("$DOPFN_ROOT/artifacts/dopfn_model.pkl"
             "$DOPFN_ROOT/artifacts/dopfn_config.pkl"
             "$DOPFN_ROOT/artifacts/model_submitit_0ccc_id_171b69db_epoch_-1.cpkt")
    IFS=, read -r -a _entries <<< "$DOPFN_MODELS"
    for entry in "${_entries[@]}"; do
        if [[ "$entry" == *=* ]]; then inputs+=("${entry#*=}"); fi
    done
fi
for f in "${inputs[@]}"; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done
[ -d "$SHARD/$DATASET/predictions" ] || {
    echo "FATAL: no shard to repair at $SHARD/$DATASET/predictions" >&2; exit 1; }
if [[ "$DATASET" == ACIC ]]; then
    [ -f "$ACIC_CACHE_DIR/x.csv" ] || {
        echo "FATAL: $ACIC_CACHE_DIR/x.csv missing; ACIC truth would try to download." >&2
        exit 1; }
fi

source "$REPO/.venv/bin/activate"
cd "$REPO"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

printf -v PROBE_TAG 'r%03d' "$PROBE_R"
REF_NPZ="$SHARD/$DATASET/predictions/${DATASET}_${PROBE_TAG}.npz"

echo "[$(date '+%F %T')] tauC REPAIR  family=$FAMILY  dataset=$DATASET" \
     " fill=[$RS,$RE)  shard=$SHARD"
echo "[repair] existing realizations: $(ls "$SHARD/$DATASET/predictions"/*.npz | wc -l)"

# ── phase 1: prove this environment reproduces the original run ────────────
if [ "$REPAIR_PROBE" = "1" ]; then
    [ -f "$REF_NPZ" ] || {
        echo "FATAL: probe reference $REF_NPZ does not exist; pick a PROBE_R that does" >&2
        exit 1; }
    echo "--- reproduction probe: recomputing $DATASET $PROBE_TAG into scratch ---"
    REAL_START=$PROBE_R REAL_END=$(( PROBE_R + 1 )) \
    OUT="$PROBE_ROOT/$DATASET" \
        python -u benchmarks/eval_graph2d/eval_density_tauC.py

    PROBE_NPZ="$PROBE_ROOT/$DATASET/predictions/${DATASET}_${PROBE_TAG}.npz"
    python -u benchmarks/eval_graph2d/diff_prediction_shards.py \
        "$REF_NPZ" "$PROBE_NPZ" || {
        echo "FATAL: this environment does NOT reproduce the existing shard." >&2
        echo "  Back-filling now would mix two configurations in one row." >&2
        exit 1; }
    echo "--- probe passed; the back-fill is comparable to what is already there ---"
else
    echo "[repair] WARNING: reproduction probe SKIPPED (REPAIR_PROBE=0)."
    echo "[repair] The back-filled realizations are assumed comparable. They are"
    echo "[repair] only comparable if this is the same environment that wrote"
    echo "[repair] $SHARD."
fi

if [ "$REPAIR_DRY_RUN" = "1" ]; then
    echo "[repair] REPAIR_DRY_RUN=1 -- probe only, nothing written to the shard."
    exit 0
fi

# ── phase 2: fill the gaps, in place ───────────────────────────────────────
# OUT points at the EXISTING directory on purpose: realizations already there
# are skipped, so only the gaps are computed.
echo "--- filling $DATASET [$RS,$RE) into $SHARD ---"
REAL_START=$RS REAL_END=$RE OUT="$SHARD/$DATASET" \
    python -u benchmarks/eval_graph2d/eval_density_tauC.py

# ── phase 3: assert the gap is actually closed ─────────────────────────────
# Both the prediction dump and the scored shard, because the MALC arms read the
# former and summarize_density_tauC.py reads the latter; a run killed between
# the two writes would leave them disagreeing.
N_PRED=$(ls "$SHARD/$DATASET/predictions"/*.npz 2>/dev/null | wc -l | tr -d ' ')
N_SCORED=$(ls "$SHARD/$DATASET"/*.npz 2>/dev/null | wc -l | tr -d ' ')
echo "[repair] $DATASET now has $N_PRED prediction / $N_SCORED scored realizations (expected $EXPECT)"

if [ "$N_PRED" -ne "$EXPECT" ] || [ "$N_SCORED" -ne "$EXPECT" ]; then
    echo "FATAL: $DATASET is still incomplete -- expected $EXPECT, have" \
         "$N_PRED prediction / $N_SCORED scored." >&2
    echo "  Re-submit this task: the eval is resumable, so it resumes where it" >&2
    echo "  stopped. Exiting non-zero so a --dependency=afterok MALC job does" >&2
    echo "  NOT run on an incomplete set." >&2
    exit 1
fi

echo "[repair] OK -- $DATASET complete at $EXPECT realizations."
echo "[repair] next: run the MALC arms over the repaired shard --"
echo "[repair]   DUMPS_ROOT=$SHARD sbatch benchmarks/cluster/submit_density_tauC_malcT.sbatch"
