#!/bin/bash
#SBATCH --job-name=cpfn-orig-full
#SBATCH --output=logs_causalpfn_original_j1024_full/train_%j.out
#SBATCH --error=logs_causalpfn_original_j1024_full/train_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120

# Exp 5: CausalPFN's own trainer, unmodified. J=1024, σ=0.01, full
# warmstart (backbone + head from TabDPT). Reference baseline — should
# reproduce the canonical CausalPFN training trajectory.
#
# Keep all checkpoints (top_k=10000). Frequency=1 → every epoch.

set -euo pipefail
mkdir -p logs_causalpfn_original_j1024_full

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

OUT_DIR="${OUT_DIR:-$DEPLOY_ROOT/causalpfn_j1024_full_output}"
mkdir -p "$OUT_DIR"

SIGMA="${SIGMA:-0.01}"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export HF_HOME="${HF_HOME:-$DEPLOY_ROOT/hf_cache}"
export PYTHONPATH="$DEPLOY_ROOT/R-PFN/benchmarks/uwyk_table1/shims:$CAUSALPFN/src:$CAUSALPFN:$DEPLOY_ROOT/R-PFN${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] CausalPFN Exp 5: J=1024 σ=$SIGMA full warmstart"
echo "  CAUSALPFN=$CAUSALPFN"
echo "  OUT_DIR=$OUT_DIR"

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
    callbacks.checkpoint.frequency=1 \
    callbacks.checkpoint.top_k=10000 \
    ~callbacks.eval_cate \
    sigma="$SIGMA" &
CHILD=$!
wait "$CHILD"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE"
