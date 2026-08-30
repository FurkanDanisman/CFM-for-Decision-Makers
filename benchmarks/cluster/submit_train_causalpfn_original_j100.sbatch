#!/bin/bash
#SBATCH --job-name=cpfn-orig-j100
#SBATCH --output=logs_causalpfn_original_j100/train_%j.out
#SBATCH --error=logs_causalpfn_original_j100/train_%j.err
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120

# CausalPFN's own trainer with nbins=100 instead of default 1024.
# Isolates the bin-resolution variable: same 1D single-arm HL-Gauss
# head, same per-arm y-standardisation, same everything except J.
# Compare vs cpfn2d J=100 (PEHE 1.39 at step_6k) and vs causalpfn
# orig J=1024 (PEHE 0.67 at step_6k).
#
# σ (HL-Gauss smoothing) also scaled from 0.01 → 0.1 to keep the
# smoothing:bin-width ratio comparable (~0.5). Without this scale-up
# HL-Gauss degenerates to hard one-hot at coarser bins.

set -euo pipefail
mkdir -p logs_causalpfn_original_j100

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }

CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
[ -f "$CAUSALPFN/train.py" ] || { echo "FATAL: no train.py at $CAUSALPFN" >&2; exit 1; }

WARMSTART_LOCAL="${WARMSTART_LOCAL:-$DEPLOY_ROOT/warmstart/tabdpt_long_context.ckpt}"
[ -f "$WARMSTART_LOCAL" ] || {
    echo "FATAL: warmstart ckpt not found at $WARMSTART_LOCAL" >&2; exit 1;
}

# Separate output dir so we don't collide with the J=1024 run.
OUT_DIR="${OUT_DIR:-$DEPLOY_ROOT/causalpfn_original_j100_output}"
mkdir -p "$OUT_DIR"

# Overridable: sigma scaling. Default 0.1 matches ratio σ/bin_width ≈ 0.5
# at J=100 (comparable to CausalPFN's default 0.01/0.0195 ≈ 0.51 at J=1024).
# Set SIGMA=0.01 explicitly to run the pure bin-count experiment (degenerate
# one-hot HL-Gauss).
SIGMA="${SIGMA:-0.1}"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export HF_HOME="${HF_HOME:-$DEPLOY_ROOT/hf_cache}"
export PYTHONPATH="$DEPLOY_ROOT/R-PFN/benchmarks/uwyk_table1/shims:$CAUSALPFN/src:$CAUSALPFN${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] launching CausalPFN's trainer with nbins=100"
echo "  CAUSALPFN=$CAUSALPFN"
echo "  OUT_DIR=$OUT_DIR"
echo "  WARMSTART_LOCAL=$WARMSTART_LOCAL"
echo "  SIGMA=$SIGMA  (default 0.1 — scaled from 0.01 to match bin_width jump 0.02→0.2)"

CHILD=""
_on_term() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SIGTERM — forwarding to child pid=$CHILD"
    [ -n "$CHILD" ] && kill -TERM "$CHILD" 2>/dev/null || true
    wait "$CHILD" 2>/dev/null || true
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}
trap _on_term SIGTERM

cd "$CAUSALPFN"
python -u train.py \
    +experiment=simple_configuration \
    hydra.run.dir="$OUT_DIR/hydra_run" \
    +callbacks@callbacks.checkpoint=checkpoint \
    callbacks.checkpoint.checkpoint_root="$OUT_DIR/checkpoints" \
    ~callbacks.eval_cate \
    model.obj.model.nbins=100 \
    sigma="$SIGMA" \
    &
CHILD=$!
wait "$CHILD"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE"
