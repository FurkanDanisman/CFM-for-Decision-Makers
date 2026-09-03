#!/bin/bash
#SBATCH --job-name=scm-cs-eval
#SBATCH --output=logs_scm_case_studies/eval_%A_%a.out
#SBATCH --error=logs_scm_case_studies/eval_%A_%a.err
#SBATCH --time=00:30:00
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

[ -f "$CKPT" ] || { echo "FATAL: ckpt not found: $CKPT" >&2; exit 1; }

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
    cpfn2d)   SCRIPT="$REPO/benchmarks/eval_causalpfn2d/eval_cpfn2d_realcause.py"    ;;
    graph2d)  SCRIPT="$REPO/benchmarks/eval_graph2d/eval_graph2d_realcause.py"       ;;
    cpfn1d)   SCRIPT="$REPO/benchmarks/eval_causalpfn2d/eval_causalpfn_v0_realcause.py" ;;
    *) echo "unknown MODEL=$MODEL" >&2; exit 3 ;;
esac

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
python -u "$SCRIPT" --dataset "$DATASET"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] done"
