#!/bin/bash
#SBATCH --job-name=cpfn-orig
#SBATCH --output=logs_causalpfn_original/train_%j.out
#SBATCH --error=logs_causalpfn_original/train_%j.err
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120

# Run CausalPFN's OWN trainer to produce a comparison trajectory of
# intermediate checkpoints. Same backbone + warmstart + optimizer as
# our cpfn2d — the only difference is the head (their 1D single-arm
# HL-Gauss vs our 2D joint bar-distribution).
#
# Use their published train.py + hydra config. Output goes to
# $DEPLOY_ROOT/causalpfn_original_output/, checkpoints under
# .../checkpoints/.

set -euo pipefail
mkdir -p logs_causalpfn_original

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

# Local warmstart — same one our cpfn2d uses.
WARMSTART_LOCAL="${WARMSTART_LOCAL:-$DEPLOY_ROOT/warmstart/tabdpt_long_context.ckpt}"
[ -f "$WARMSTART_LOCAL" ] || {
    echo "FATAL: warmstart ckpt not found at $WARMSTART_LOCAL" >&2
    exit 1
}

OUT_DIR="${OUT_DIR:-$DEPLOY_ROOT/causalpfn_original_output}"
mkdir -p "$OUT_DIR"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# Disable torch.compile / dynamo. Their trainer wraps the model in
# torch.compile by default, which triggers the same Inductor tiling
# assertion we hit ourselves (job 5032910, torch._inductor.exc.
# InductorError in tiling_utils.py get_pw_red_splits). Skip compile
# entirely — training is slower but functional.
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
# PYTHONPATH:
#   1. shims first — faiss stub (their trainer imports causalpfn which
#      chains through causal_estimator's `import faiss`)
#   2. $CAUSALPFN/src — the `causalpfn` package lives there
#   3. $CAUSALPFN — for `benchmarks` and other top-level modules
export PYTHONPATH="$DEPLOY_ROOT/R-PFN/benchmarks/uwyk_table1/shims:$CAUSALPFN/src:$CAUSALPFN${PYTHONPATH:+:$PYTHONPATH}"

# Point HF cache into $DEPLOY_ROOT in case anything wants to download.
export HF_HOME="${HF_HOME:-$DEPLOY_ROOT/hf_cache}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] launching CausalPFN's own trainer"
echo "  CAUSALPFN=$CAUSALPFN"
echo "  OUT_DIR=$OUT_DIR"
echo "  WARMSTART_LOCAL=$WARMSTART_LOCAL"

# Forward SIGTERM to python so their trainer can save on wall-clock timeout,
# then requeue.
CHILD=""
_on_term() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SIGTERM — forwarding to child pid=$CHILD"
    [ -n "$CHILD" ] && kill -TERM "$CHILD" 2>/dev/null || true
    wait "$CHILD" 2>/dev/null || true
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] requeueing $SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}
trap _on_term SIGTERM

cd "$CAUSALPFN"
# Hydra overrides:
#   +experiment=simple_configuration  → their default training preset
#   hydra.run.dir=...                 → point outputs to our dir
#   +callbacks@callbacks.checkpoint=checkpoint  → enable checkpoint callback
#   callbacks.checkpoint.checkpoint_root=$OUT_DIR/checkpoints
#   callbacks.checkpoint.save_every_n_steps=2000  → matches our cadence
# If any of these keys don't exist in their yaml, hydra will error out
# with the correct name; we can iterate.
python -u train.py \
    +experiment=simple_configuration \
    hydra.run.dir="$OUT_DIR/hydra_run" \
    +callbacks@callbacks.checkpoint=checkpoint \
    callbacks.checkpoint.checkpoint_root="$OUT_DIR/checkpoints" \
    ~callbacks.eval \
    &
    # ~callbacks.eval removes the in-training CATE eval callback which
    # calls CATEEstimator.estimate_cate → needs faiss (unavailable here).
    # We only want raw checkpoints; standalone eval via our minimal script
    # after training completes.
CHILD=$!
wait "$CHILD"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE"
