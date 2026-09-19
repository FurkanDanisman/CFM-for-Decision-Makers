#!/bin/bash
#SBATCH --job-name=tauC-dens
#SBATCH --output=logs_density_tauC/eval_%A_%a.out
#SBATCH --error=logs_density_tauC/eval_%A_%a.err
#SBATCH --account=def-rgrosse_gpu
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --array=0-29

# Tier-C density eval on Fir: p(tau | x) for UWYK, DoPFN and CausalPFN on IHDP + ACIC.
# Same eval, inputs and outputs as submit_density_tauC.sbatch (Killarney); only
# the scheduling and the paths are Fir's. v1 = RAW path only (no MALC).
#
# ── HOW FIR DIFFERS (all read off this scheduler, 2026-09) ─────────────────
# Account   def-rgrosse_gpu is the only GPU association (aip-rgrosse is Killarney).
# Partition deliberately NOT set. Fir routes by --time: <=3h -> gpubase_bygpu_b1,
#           <=12h -> _b2, <=24h -> _b3, ... Pinning one only invites "Requested
#           time limit is invalid". 03:00:00 is the whole b1 bucket, the shortest.
# GPU       must be named ("Specifying a GPU model is now required"). Default is
#           a 3g.40gb MIG slice of an H100, not a whole H100. This job is
#           CPU-bound: the tau quadrature is numpy, and the GPU only runs a few
#           small forward passes (the models are 7M-60M params). Billing is
#           MAX_TRES, per unit:
#             h100 (whole)  12200     = 12 cores or 288G   <- Killarney request
#             3g.40gb        5229     =  5 cores or 123G   <- default here
#             2g.20gb        3486     =  3 cores or  82G
#             1g.10gb        1743     =  1 core  or  41G
#           so the slice costs 43% of the fairshare a whole H100 does, and 5
#           CPUs + 32G stay inside what it already pays for. 40 GB of GPU memory
#           is ample. b1 has 60 MIG nodes (4x 3g.40gb each) and 30 whole-H100
#           nodes; on 2026-09-18 `sbatch --test-only` started both immediately.
#           Whole H100 instead (e.g. to rule the slice out while debugging):
#             sbatch --gres=gpu:h100:1 --cpus-per-task=12 --mem=64G <this file>
# Memory    DefMemPerCPU is 256M on Fir, so --mem must stay explicit.
# Network   compute nodes DO reach GitHub (HTTP 200 from a b1 node, 2026-09-18);
#           CausalPFN's ACIC2016Dataset downloads its CSVs at every load.
#
# ── PARALLELISM: REALIZATIONS, never models ────────────────────────────────
#   0-19   IHDP realizations in blocks of 5   (100 total)
#   20-29  ACIC realizations one per task     (10 total)
# Killarney used 12 tasks (IHDP x10, ACIC x5). MIG slices are plentiful, so
# Fir gets 30 smaller tasks: the ACIC tasks were the long pole, and at 5
# realizations x 480 queries MODEL_FAMILY=all would run close to the 3h cap.
# All selected models still run in ONE process per task, so every model sees
# byte-identical context and outcome scaling. Realizations are deterministic
# given (dataset, r, EVAL_MAX_CONTEXT, EVAL_CONTEXT_SEED), so the split changes
# scheduling only, never results.
#
# Cost (extrapolated from Killarney: ~1.1 s/query for the 3 UWYK methods at
# N_Y0=4096, i.e. ~0.37 s/query/method; DoPFN 1D rows are cheaper, joint rows
# pay the same N_Y0 quadrature as `joint`):
#   IHDP task = 5 real x 75 q  = 375 q:  uwyk ~7 min,  all (9 methods) ~21 min
#   ACIC task = 1 real x 480 q = 480 q:  uwyk ~9 min,  all (9 methods) ~26 min
# plus model loading and the forwards. Unmeasured on Fir's CPUs and on a MIG
# slice, so the 3h cap is left in place as headroom.
#
# RESUME. A task first skips the contiguous run of its realizations whose
# shard already exists in OUT, so a requeued or resubmitted task continues
# where it stopped. The default OUT_ROOT is per job id, so this only engages
# on a requeue or when you pass an old OUT_ROOT explicitly:
#   OUT_ROOT=$PWD/results_density_tauC/<JOBID> sbatch --array=<ids> <this file>
# A shard written by a different MODEL_FAMILY aborts the task. Use a fresh
# OUT_ROOT whenever the model list or the eval configuration changes.
#
# ── USAGE (from the repository root; logs/ must exist before sbatch opens it)
# Preflight on a login node, reports every missing input at once:
#          MODEL_FAMILY=dopfn bash benchmarks/cluster/submit_density_tauC_fir.sbatch
# Submit:  mkdir -p logs_density_tauC
#          MODEL_FAMILY=dopfn sbatch benchmarks/cluster/submit_density_tauC_fir.sbatch
# Smoke:   sbatch --array=0 benchmarks/cluster/submit_density_tauC_fir.sbatch
#          (task 0 also runs the model-free gates and aborts if any fail)
# 1 real:  REAL_END=1 sbatch --array=0 benchmarks/cluster/submit_density_tauC_fir.sbatch
# Then:    python benchmarks/eval_graph2d/summarize_density_tauC.py results_density_tauC/<JOBID>
# SAVE_PREDICTIONS=1 (default) retains raw outputs under <OUT>/predictions.

set -euo pipefail

# Run with plain `bash` (no Slurm job) = preflight only.
PREFLIGHT=0
[ -n "${SLURM_JOB_ID:-}" ] || PREFLIGHT=1

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$REPO/benchmarks/eval_graph2d/eval_density_tauC.py" ] || {
    echo "FATAL: REPO=$REPO is not the repository root; cd there or export REPO." >&2
    exit 1; }
cd "$REPO"
mkdir -p logs_density_tauC

# ── environment and external checkouts ─────────────────────────────────────
# Defaults are siblings of the repo (e.g. ~/Do-PFN next to
# ~/CFM-for-Decision-Makers). Any of them can be exported instead.
# benchmarks/cluster/setup_killarney.sh builds the same set (clones, the two
# Do-PFN patches, the venv) and runs on Fir unchanged; point these at its output.
VENV="${VENV:-$REPO/.venv}"
export UWYK="${UWYK:-$REPO/g4cfm}"
export CAUSALPFN="${CAUSALPFN:-$REPO/CausalPFN}"

# Select uwyk (legacy default), dopfn, causalpfn, or all. The DoPFN checkout must include
# its pretrained artifacts even when only our checkpoints are listed below.
export MODEL_FAMILY="${MODEL_FAMILY:-uwyk}"
export DOPFN_ROOT="${DOPFN_ROOT:-${DOPFN:-$REPO/Do-PFN}}"

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

# ── checkpoints ────────────────────────────────────────────────────────────
# Same checkpoints as the Killarney script, at their Required_checkpoints/old/
# location in this checkout.
# Joint-2D: our graph2d head. step_50000 matches the "Ours V6A - 5206104" row
# in LATEST_RESULTS.md; override CKPT to compare step_58000.
export CKPT="${CKPT:-$REPO/Required_checkpoints/old/graph2d_step_50000.pt}"
export CAUSALPFN_CKPT="${CAUSALPFN_CKPT:-$REPO/Required_checkpoints/old/cpfn1d_j1024_headrand_step_50000.pt}"
export CAUSALPFN_JOINT_CKPT="${CAUSALPFN_JOINT_CKPT:-$REPO/Required_checkpoints/old/cpfn2d_j32_random_step_50000.pt}"
export CAUSALPFN_QUERY_CHUNK="${CAUSALPFN_QUERY_CHUNK:-512}"

# UWYK-1D: the Table-1 full-conditioning checkpoint (git-lfs inside g4cfm). NOT
# Required_checkpoints/old/uwyk_reproduce_best_model.pt -- that file is a copy of
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

# ── inputs: collect every miss, then fail once ─────────────────────────────
case "$MODEL_FAMILY" in
    uwyk|dopfn|causalpfn|all) ;;
    *) echo "FATAL: MODEL_FAMILY must be uwyk, dopfn, causalpfn, or all" >&2; exit 1 ;;
esac
missing=()
need() { [ -e "$1" ] || missing+=("$1${2:+   <- $2}"); }

need "$VENV/bin/activate" "export VENV=<your venv>"
# The harness imports UWYK's graph code and the RealCause loaders for EVERY
# family, so these two are needed even for DoPFN-only runs.
need "$UWYK/src/models/GraphConditionedInterventionalPFN_sklearn.py" "export UWYK=<g4cfm checkout>"
need "$CAUSALPFN/benchmarks/IHDP" "export CAUSALPFN=<CausalPFN checkout>; ships the IHDP data"
# load_acic_truth reads x.csv purely for a row count, and reaches out to
# raw.githubusercontent.com if it is not cached. Kept required so a run never
# depends on GitHub for the truth's row order.
need "$ACIC_CACHE_DIR/x.csv" "on a login node: mkdir -p $ACIC_CACHE_DIR && curl -sSfL https://raw.githubusercontent.com/BiomedSciAI/causallib/master/causallib/datasets/data/acic_challenge_2016/x.csv -o $ACIC_CACHE_DIR/x.csv"
if [[ "$MODEL_FAMILY" == uwyk || "$MODEL_FAMILY" == all ]]; then
    need "$CKPT"
    need "$UWYK_CKPT" "git -C $UWYK lfs pull"
    need "$UWYK_CFG"
fi
if [[ "$MODEL_FAMILY" == causalpfn || "$MODEL_FAMILY" == all ]]; then
    need "$CAUSALPFN_CKPT"
    need "$CAUSALPFN_JOINT_CKPT"
    need "$CAUSALPFN/src/causalpfn/models/model.py"
fi
if [[ "$MODEL_FAMILY" == dopfn || "$MODEL_FAMILY" == all ]]; then
    for f in dopfn_model.pkl dopfn_config.pkl model_submitit_0ccc_id_171b69db_epoch_-1.cpkt; do
        need "$DOPFN_ROOT/artifacts/$f" "export DOPFN_ROOT=<patched Do-PFN checkout>"
    done
    IFS=, read -r -a _dopfn_entries <<< "$DOPFN_MODELS"
    for entry in "${_dopfn_entries[@]}"; do
        if [[ "$entry" == *=* ]]; then need "${entry#*=}" "DOPFN_MODELS entry ${entry%%=*}"; fi
    done
fi
if [ "${#missing[@]}" -gt 0 ]; then
    echo "FATAL: ${#missing[@]} missing input(s) for MODEL_FAMILY=$MODEL_FAMILY:" >&2
    printf '  %s\n' "${missing[@]}" >&2
    exit 1
fi
if [ "$PREFLIGHT" -eq 1 ]; then
    echo "preflight OK: MODEL_FAMILY=$MODEL_FAMILY REPO=$REPO"
    [ "$MODEL_FAMILY" = uwyk ] || [ "$MODEL_FAMILY" = causalpfn ] \
        || echo "  DoPFN rows: $DOPFN_MODELS"
    exit 0
fi

# ── array -> (dataset, realization slice) ──────────────────────────────────
# Capture caller overrides BEFORE the task-id mapping computes its own values,
# so `REAL_END=1 sbatch --array=0 ...` actually trims the slice.
_RS_IN="${REAL_START:-}"
_RE_IN="${REAL_END:-}"
_DS_IN="${DATASET:-}"

T=$SLURM_ARRAY_TASK_ID
if   [ "$T" -lt 20 ]; then
    DATASET=IHDP;  REAL_START=$(( T * 5 ));  REAL_END=$(( T * 5 + 5 ))
elif [ "$T" -lt 30 ]; then
    DATASET=ACIC;  REAL_START=$(( T - 20 )); REAL_END=$(( T - 19 ))
else
    echo "invalid task id: $T" >&2; exit 2
fi
DATASET="${_DS_IN:-$DATASET}"
REAL_START="${_RS_IN:-$REAL_START}"
REAL_END="${_RE_IN:-$REAL_END}"
export OUT_ROOT="${OUT_ROOT:-$REPO/results_density_tauC/$SLURM_ARRAY_JOB_ID}"
export OUT="$OUT_ROOT/$DATASET"

# ── software ───────────────────────────────────────────────────────────────
# A venv built from an Alliance python module points into cvmfs and needs that
# module loaded; a uv-built venv (home under ~/.local/share/uv) is standalone.
PYVER=$(sed -nE 's/^version(_info)? *= *([0-9]+\.[0-9]+).*/\2/p' "$VENV/pyvenv.cfg" 2>/dev/null | head -1)
if [ -n "$PYVER" ] && grep -q '^home *= */cvmfs/' "$VENV/pyvenv.cfg" \
        && command -v module >/dev/null 2>&1; then
    set +u; module load StdEnv/2023 "python/$PYVER"; set -u
fi
set +u; source "$VENV/bin/activate"; set -u
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export FLEXIBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
# shims/ = sitecustomize denylist that keeps causalpfn's __init__ from dragging
# in deps we do not have; same one the graph2d and uwyk1d evals use.
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

# ── resume: skip this slice's leading realizations that already have a shard
if [ "${RESUME:-1}" = 1 ]; then
    REAL_START=$(python - "$OUT" "$DATASET" "$REAL_START" "$REAL_END" "$MODEL_FAMILY" <<'PY'
import os, sys, zipfile
import numpy as np
out, dataset, r, hi, family = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
while r < hi:
    path = os.path.join(out, f'{dataset}_r{r:03d}.npz')
    if not os.path.exists(path):
        break
    try:
        with np.load(path) as z:
            found = str(z['model_family'])
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        break                       # cut short mid-write: redo it
    if found != family:
        sys.exit(f'FATAL: {path} was written by MODEL_FAMILY={found}, this run is '
                 f'{family}. Use a fresh OUT_ROOT.')
    r += 1
print(r)
PY
    )
fi
export DATASET REAL_START REAL_END
if [ "$REAL_START" -ge "$REAL_END" ]; then
    echo "[$(date '+%F %T')] $DATASET r<$REAL_END: every shard already in $OUT; nothing to do"
    exit 0
fi

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
echo "  node=$(hostname)  cpus=$SLURM_CPUS_PER_TASK  gpu: $(nvidia-smi -L 2>/dev/null | tr '\n' ' ')"

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
