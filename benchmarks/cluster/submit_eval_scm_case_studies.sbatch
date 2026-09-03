#!/bin/bash
#SBATCH --job-name=scm-cs-eval
#SBATCH --output=logs_scm_case_studies/eval_%A_%a.out
#SBATCH --error=logs_scm_case_studies/eval_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-5

# 6-task array — one per DoPFN synthetic case study.
#   0: Observed_Confounder                1: Observed_Mediator
#   2: Observed_Mediator_and_Confounder   3: Unobserved_Confounder
#   4: Frontdoor_Criterion                5: Backdoor_Criterion
#
# Env vars:
#   MODEL       cpfn2d | graph2d | cpfn1d   (which realcause script to invoke)
#   CKPT        checkpoint path
#   OUT_ROOT    per-case-study dir will be created as OUT_ROOT/<case>
#   ANC_MODE    (graph2d only): noanc | v6a — passed via GRAPH2D_ANC_MODE env
#   STD_MODE    (cpfn2d/cpfn1d only): pooled | log | ...
#   DOPFN_DATA_ROOT  parent of prior_sampling/  (default set below)

set -euo pipefail
mkdir -p logs_scm_case_studies

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
export CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
export UWYK="${UWYK:-$DEPLOY_ROOT/external/uwyk}"
export CKPT="${CKPT:?CKPT required}"
export OUT_ROOT="${OUT_ROOT:?OUT_ROOT required}"
export MODEL="${MODEL:?MODEL required (cpfn2d | graph2d | cpfn1d)}"
export STD_MODE="${STD_MODE:-}"
export GRAPH2D_ANC_MODE="${GRAPH2D_ANC_MODE:-noanc}"
export DOPFN_DATA_ROOT="${DOPFN_DATA_ROOT:-$DEPLOY_ROOT/external/dopfn/data/prior_sampling}"
# SCM case studies: only need raw mean CATE, skip EM everywhere (huge speedup)
export SKIP_EM="${SKIP_EM:-1}"
# DOPFN_ROOT = parent of data/prior_sampling — needed to unpickle case-study pkls
# (they carry DoPFN's `datasets.InterventionalDataset` class refs).
export DOPFN_ROOT="${DOPFN_ROOT:-${DOPFN_DATA_ROOT%/data/prior_sampling}}"

# dopfn_native uses DoPFNRegressor().__init__ artifacts — no ckpt needed
if [ "$MODEL" != "dopfn_native" ]; then
    [ -f "$CKPT" ] || { echo "FATAL: ckpt not found: $CKPT" >&2; exit 1; }
fi

case "$SLURM_ARRAY_TASK_ID" in
    0) DATASET=Observed_Confounder              ;;
    1) DATASET=Observed_Mediator                ;;
    2) DATASET=Observed_Mediator_and_Confounder ;;
    3) DATASET=Unobserved_Confounder            ;;
    4) DATASET=Frontdoor_Criterion              ;;
    5) DATASET=Backdoor_Criterion               ;;
    *) echo "invalid task id: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac
export DATASET
export OUT="$OUT_ROOT/$DATASET"

case "$MODEL" in
    cpfn2d)     SCRIPT="$REPO/benchmarks/eval_causalpfn2d/eval_cpfn2d_realcause.py" ;;
    graph2d)    SCRIPT="$REPO/benchmarks/eval_graph2d/eval_graph2d_realcause.py" ;;
    cpfn1d)     SCRIPT="$REPO/benchmarks/eval_causalpfn2d/eval_causalpfn_v0_realcause.py" ;;
    dopfn_native)
        # native DoPFNRegressor — no --dataset flag needed (env var is enough)
        SCRIPT="$REPO/benchmarks/eval_scm_case_studies/eval_native_dopfn.py"
        export DOPFN_ROOT="${DOPFN_ROOT:-$DEPLOY_ROOT/external/dopfn}"
        ;;
    dopfn_bb)
        # DoPFN-bb reused eval_dopfn_bb_raw.py; different arg style (positional flags)
        SCRIPT="$REPO/benchmarks/l2_ihdp/eval_dopfn_bb_raw.py"
        DOPFN_ROOT="${DOPFN_ROOT:-$DEPLOY_ROOT/external/dopfn}"
        EXTRA_ARGS="--repo $REPO --dopfn $DOPFN_ROOT --causalpfn $CAUSALPFN --checkpoint $CKPT --dataset $DATASET --out $OUT/summary.npz"
        ;;
    fn50)
        # fn=50: InterventionalPFN architecture (distinct from cpfn2d).
        SCRIPT="$REPO/benchmarks/eval_scm_case_studies/eval_fn50.py"
        ;;
    uwyk)
        SCRIPT="$REPO/benchmarks/eval_scm_case_studies/eval_uwyk.py"
        export UWYK_SRC="${UWYK_SRC:-$DEPLOY_ROOT/external/uwyk/src}"
        export UWYK_CKPT_DIR="${UWYK_CKPT_DIR:-$(dirname "$CKPT")}"
        # Explicit .pt / .yaml paths (avoids the fragile filename-guessing fallback):
        export UWYK_CKPT="${UWYK_CKPT:-$CKPT}"
        export UWYK_CONFIG="${UWYK_CONFIG:-}"
        export ANC_MODE="${UWYK_ANC_MODE:-noanc}"
        ;;
    *) echo "unknown MODEL=$MODEL" >&2; exit 3 ;;
esac
: "${EXTRA_ARGS:=}"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] SCM case-study eval  MODEL=$MODEL  DATASET=$DATASET"
echo "  CKPT=$CKPT"
echo "  OUT=$OUT  STD_MODE=${STD_MODE:-<default>}  GRAPH2D_ANC_MODE=$GRAPH2D_ANC_MODE"
echo "  DOPFN_DATA_ROOT=$DOPFN_DATA_ROOT"
if [ "$MODEL" = "dopfn_bb" ]; then
    python -u "$SCRIPT" $EXTRA_ARGS
else
    python -u "$SCRIPT" --dataset "$DATASET"
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] done"
