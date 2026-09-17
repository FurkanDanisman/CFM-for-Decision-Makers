#!/bin/bash
#SBATCH --job-name=tauC-dens
#SBATCH --output=logs_density_tauC/eval_%A_%a.out
#SBATCH --error=logs_density_tauC/eval_%A_%a.err
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --account=aip-rgrosse
#SBATCH --time=03:00:00
#SBATCH --gpus-per-node=h100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --array=0-3

# TamIA requires a whole H100 node (4 GPUs) per array task.
# This version launches up to four independent realization workers per array
# task, one pinned to each H100, so the whole node is actually used.
#
# Tier-C density eval: p(tau | x) for UWYK-1D vs Joint-2D on IHDP + ACIC.
# v1 = RAW path only (no MALC). Spec: density_eval_pipeline.md.
#
# PARALLELISM: split only over REALIZATIONS, never over models.
#   array 0: IHDP r=[0,33)
#   array 1: IHDP r=[33,66)
#   array 2: IHDP r=[66,100)
#   array 3: ACIC r=[0,10)
#
# Each array task then divides its realization range as evenly as possible
# across the 4 local H100s.  Each worker still evaluates all requested models
# for its realizations in one process, preserving the byte-identical
# preprocessing/model-comparison guarantee.  Realizations are independent and
# deterministic given (dataset, r, EVAL_MAX_CONTEXT, EVAL_CONTEXT_SEED), so
# this changes scheduling only, not the intended inputs or realization set.
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
# With four workers per node, the largest per-GPU slices are:
#   IHDP: at most 9 realizations/GPU
#   ACIC: at most 3 realizations/GPU
# so --time=03:00:00 remains conservative while using all four H100s.
#
# N_Y0=4096 is set by the JOINT path, not UWYK: UWYK's tail quadrature is at
# 0.99998 by N_Y0=1024, while the joint only reaches 1e-4 mass error at 4096.
#
# TamIA: submit from the repository root. Create logs BEFORE sbatch opens them.
# Setup:   mkdir -p logs_density_tauC
# Runtime defaults: $REPO/.venv and $REPO/../CausalPFN (source + datasets).
# These must be installed before submission; checkpoint files are separate.
# CausalPFN: MODEL_FAMILY=causalpfn sbatch benchmarks/cluster/submit_density_tauC_tamia.sbatch
# Submit:  sbatch benchmarks/cluster/submit_density_tauC_tamia.sbatch
# Smoke:   sbatch --array=0 benchmarks/cluster/submit_density_tauC_tamia.sbatch
#          (task 0 also runs the model-free gates and aborts if any fail)
# 1 real:  REAL_END=1 sbatch --array=0 benchmarks/cluster/submit_density_tauC_tamia.sbatch
# Then:    python benchmarks/eval_graph2d/summarize_density_tauC.py results_density_tauC/<JOBID>

set -euo pipefail
mkdir -p logs_density_tauC

REPO="${REPO:-${SLURM_SUBMIT_DIR:?Submit from the repository root or export REPO}}"
[ -f "$REPO/benchmarks/eval_graph2d/eval_density_tauC.py" ] || {
    echo "FATAL: REPO=$REPO is not the repository root; export REPO before sbatch." >&2
    exit 1
}
cd "$REPO"

export UWYK="${UWYK:-$REPO/g4cfm}"
export CAUSALPFN="${CAUSALPFN:-$REPO/../CausalPFN}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$REPO/.venv/bin/activate}"
[ -f "$VENV_ACTIVATE" ] || {
    echo "FATAL: missing $VENV_ACTIVATE; export VENV_ACTIVATE for your TamIA environment." >&2
    exit 1
}
[ -d "$CAUSALPFN/benchmarks/IHDP" ] || {
    echo "FATAL: missing $CAUSALPFN/benchmarks/IHDP; export CAUSALPFN for your dataset checkout." >&2
    exit 1
}

# Select uwyk (legacy default), dopfn, causalpfn, or all. The DoPFN checkout must include
# its pretrained artifacts even when loading our joint checkpoint.
# Example: MODEL_FAMILY=dopfn DOPFN_ROOT=/path/to/Do-PFN sbatch ...
export MODEL_FAMILY="${MODEL_FAMILY:-uwyk}"
export DOPFN_ROOT="${DOPFN_ROOT:-${DOPFN:-}}"
export DOPFN_JOINT_CKPT="${DOPFN_JOINT_CKPT:-$REPO/Required_checkpoints/dopfn_bb_j10_step_150000.pt}"
export DOPFN_QUERY_CHUNK="${DOPFN_QUERY_CHUNK:-20}"
export CAUSALPFN_CKPT="${CAUSALPFN_CKPT:-$REPO/Required_checkpoints/cpfn1d_j1024_headrand_step_50000.pt}"
export CAUSALPFN_JOINT_CKPT="${CAUSALPFN_JOINT_CKPT:-$REPO/Required_checkpoints/cpfn2d_j32_random_step_50000.pt}"
export CAUSALPFN_QUERY_CHUNK="${CAUSALPFN_QUERY_CHUNK:-512}"

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
# Same +/-1 matrix is handed to BOTH models either way. Setting ANC_MODE alone
# leaves ANC_TAG=v6a, which the eval now rejects up front rather than at the
# first realization.
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
    inputs+=("$CAUSALPFN_CKPT" "$CAUSALPFN_JOINT_CKPT"
             "$CAUSALPFN/src/causalpfn/models/model.py")
fi
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
if [ "$T" -lt 3 ]; then
    # Integer partition of the same 100 IHDP realizations over three nodes:
    # [0,33), [33,66), [66,100).
    DATASET=IHDP
    REAL_START=$(( 100 * T / 3 ))
    REAL_END=$(( 100 * (T + 1) / 3 ))
elif [ "$T" -eq 3 ]; then
    DATASET=ACIC
    REAL_START=0
    REAL_END=10
else
    echo "invalid task id: $T" >&2; exit 2
fi
DATASET="${_DS_IN:-$DATASET}"
REAL_START="${_RS_IN:-$REAL_START}"
REAL_END="${_RE_IN:-$REAL_END}"
export DATASET REAL_START REAL_END

source "$VENV_ACTIVATE"
export PYTHONUNBUFFERED=1
# shims/ = sitecustomize denylist that keeps causalpfn's __init__ from dragging
# in deps we do not have; same one the graph2d and uwyk1d evals use.
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

FINAL_OUT="$OUT_ROOT/$DATASET"
echo "[$(date '+%F %T')] tauC density  dataset=$DATASET  r=[$REAL_START,$REAL_END)" \
     " family=$MODEL_FAMILY  anc=$ANC_TAG  N_Y0=$N_Y0  OUT=$FINAL_OUT"

# The model-free validation gates are cheap (~2 min) and every number below is
# meaningless if they fail. Run them once, on the first task only.
if [ "$T" -eq 0 ]; then
    echo "--- validation gates ---"
    python -u "$REPO/benchmarks/eval_graph2d/test_density_common.py" \
        || { echo "FATAL: density gates failed; not running the eval." >&2; exit 3; }
    echo "--- gates passed ---"
fi

# ── split this array task across the node's 4 H100s ─────────────────────────
TOTAL=$(( REAL_END - REAL_START ))
if [ "$TOTAL" -le 0 ]; then
    echo "FATAL: empty realization range [$REAL_START,$REAL_END)" >&2
    exit 4
fi

NWORKERS=4
if [ "$TOTAL" -lt "$NWORKERS" ]; then
    NWORKERS="$TOTAL"
fi

# Match the original per-process CPU budget: 4 CPU threads per evaluator.
THREADS_PER_WORKER=4

# Workers write to isolated directories first.  After all workers succeed,
# their trees are merged into the original final output path.  If two workers
# unexpectedly produce the same relative filename with different contents, the
# merge fails instead of silently overwriting data.
WORKER_ROOT="$OUT_ROOT/.workers/task_${T}"
rm -rf "$WORKER_ROOT"
mkdir -p "$WORKER_ROOT" "$FINAL_OUT"

declare -a PIDS=()
declare -a WOUTS=()
declare -a WSTARTS=()
declare -a WENDS=()
declare -a WGPUS=()

BASE=$(( TOTAL / NWORKERS ))
REM=$(( TOTAL % NWORKERS ))
CUR="$REAL_START"

for ((w=0; w<NWORKERS; w++)); do
    LEN="$BASE"
    if [ "$w" -lt "$REM" ]; then
        LEN=$(( LEN + 1 ))
    fi

    WS="$CUR"
    WE=$(( CUR + LEN ))
    CUR="$WE"

    GPU="$w"
    WOUT="$WORKER_ROOT/gpu${GPU}/$DATASET"
    mkdir -p "$WOUT"

    WLOG="logs_density_tauC/eval_${SLURM_ARRAY_JOB_ID}_${T}_gpu${GPU}.out"
    WERR="logs_density_tauC/eval_${SLURM_ARRAY_JOB_ID}_${T}_gpu${GPU}.err"

    echo "[$(date '+%F %T')] launch gpu=$GPU  dataset=$DATASET  r=[$WS,$WE)  OUT=$WOUT"

    (
        export CUDA_VISIBLE_DEVICES="$GPU"
        export REAL_START="$WS"
        export REAL_END="$WE"
        export OUT="$WOUT"

        export OMP_NUM_THREADS="$THREADS_PER_WORKER"
        export MKL_NUM_THREADS="$THREADS_PER_WORKER"
        export OPENBLAS_NUM_THREADS="$THREADS_PER_WORKER"

        python -u "$REPO/benchmarks/eval_graph2d/eval_density_tauC.py"
    ) >"$WLOG" 2>"$WERR" &

    PIDS+=("$!")
    WOUTS+=("$WOUT")
    WSTARTS+=("$WS")
    WENDS+=("$WE")
    WGPUS+=("$GPU")
done

FAIL=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "[$(date '+%F %T')] finished gpu=${WGPUS[$i]}  r=[${WSTARTS[$i]},${WENDS[$i]})"
    else
        STATUS=$?
        echo "ERROR: worker gpu=${WGPUS[$i]} r=[${WSTARTS[$i]},${WENDS[$i]}) exited $STATUS" >&2
        echo "       logs: logs_density_tauC/eval_${SLURM_ARRAY_JOB_ID}_${T}_gpu${WGPUS[$i]}.{out,err}" >&2
        FAIL=1
    fi
done

if [ "$FAIL" -ne 0 ]; then
    echo "FATAL: at least one GPU worker failed; leaving $WORKER_ROOT for debugging." >&2
    exit 5
fi

# Merge each worker tree back into the exact output location used by the old
# script. Multiple IHDP array tasks may finish together, so serialize only this
# short merge phase. GPU evaluation remains fully parallel.
MERGE_LOCK="$OUT_ROOT/.merge.lock"
exec 9>"$MERGE_LOCK"
flock 9

# Unique per-realization files are copied normally. Duplicate files are
# accepted only if byte-identical; differing duplicates are treated as a bug.
for WOUT in "${WOUTS[@]}"; do
    while IFS= read -r -d '' SRC; do
        REL="${SRC#"$WOUT"/}"
        DST="$FINAL_OUT/$REL"

        if [ -d "$SRC" ]; then
            mkdir -p "$DST"
        elif [ -f "$SRC" ]; then
            mkdir -p "$(dirname "$DST")"
            if [ -e "$DST" ]; then
                if ! cmp -s "$SRC" "$DST"; then
                    echo "FATAL: conflicting worker outputs for $DST" >&2
                    echo "       source: $SRC" >&2
                    exit 6
                fi
            else
                cp -p "$SRC" "$DST"
            fi
        else
            echo "FATAL: unsupported output type while merging: $SRC" >&2
            exit 7
        fi
    done < <(find "$WOUT" -mindepth 1 -print0)
done

rm -rf "$WORKER_ROOT"
flock -u 9

echo "[$(date '+%F %T')] done  dataset=$DATASET  r=[$REAL_START,$REAL_END)  OUT=$FINAL_OUT"
