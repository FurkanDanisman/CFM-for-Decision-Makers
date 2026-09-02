#!/bin/bash
#SBATCH --job-name=cpfn2d-smoke
#SBATCH --output=logs_cpfn2d_smoke/train_%j.out
#SBATCH --error=logs_cpfn2d_smoke/train_%j.err
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Smoke test — retrain cpfn2d with a TIGHTER inner region so the 9-region
# tail head is activated. Runs ~600 steps, saves a step ckpt every 100
# steps. Purpose: verify the loss is stable and the tail params start
# receiving gradient (region-weight distribution shifts away from
# w_inner=1.0).
#
# Required env:
#   EDGE_LO, EDGE_HI       inner-region bounds (e.g. -3, 3 or -4, 4)
#   J                      inner-grid side (32 or 100)
#   HLGAUSS_SIGMA          HL-Gauss target σ (target: σ/bin_width ≈ 0.5)
#   OUT_TAG                unique suffix for out dir + jobname
#
# Optional env:
#   N_STEPS                default 600
#   STEP_CKPT_EVERY        default 100

set -euo pipefail
mkdir -p logs_cpfn2d_smoke

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

PATCH_DIR="${PATCH_DIR:-$DEPLOY_ROOT/R-PFN/rpfn_patches/causalpfn_step_ckpt}"
[ -x "$PATCH_DIR/install.sh" ] || { echo "FATAL: step-ckpt install missing" >&2; exit 1; }
DEPLOY_ROOT="$DEPLOY_ROOT" CAUSALPFN="$CAUSALPFN" "$PATCH_DIR/install.sh"

WARMSTART_LOCAL="${WARMSTART_LOCAL:-$DEPLOY_ROOT/warmstart/tabdpt_long_context.ckpt}"
[ -f "$WARMSTART_LOCAL" ] || { echo "FATAL: warmstart not found" >&2; exit 1; }

# Required knobs (fail fast if unset)
EDGE_LO="${EDGE_LO:?EDGE_LO required}"
EDGE_HI="${EDGE_HI:?EDGE_HI required}"
J="${J:?J required (32 or 100)}"
HLGAUSS_SIGMA="${HLGAUSS_SIGMA:?HLGAUSS_SIGMA required}"
OUT_TAG="${OUT_TAG:?OUT_TAG required (e.g. e3_j32)}"

OUT_DIR="${OUT_DIR:-$DEPLOY_ROOT/cpfn2d_smoke_${OUT_TAG}}"
mkdir -p "$OUT_DIR"

N_STEPS="${N_STEPS:-600}"
STEP_CKPT_EVERY="${STEP_CKPT_EVERY:-100}"
LOSS_TYPE="${LOSS_TYPE:-hlgauss}"
Y_SCALING_MODE="${Y_SCALING_MODE:-pooled_std}"
NUM_FEATURES="${NUM_FEATURES:-99}"

BIN_WIDTH=$(python -c "print(($EDGE_HI - ($EDGE_LO)) / $J)")
SIGMA_OVER_BW=$(python -c "print($HLGAUSS_SIGMA / (($EDGE_HI - ($EDGE_LO)) / $J))")

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$DEPLOY_ROOT/hf_cache}"
export PYTHONPATH="$DEPLOY_ROOT/R-PFN/benchmarks/uwyk_table1/shims:$CAUSALPFN/src:$CAUSALPFN:$DEPLOY_ROOT/R-PFN${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] SMOKE cpfn2d  tag=$OUT_TAG"
echo "  edges=[$EDGE_LO, $EDGE_HI]   J=$J   bin_width=$BIN_WIDTH"
echo "  σ_hl=$HLGAUSS_SIGMA         σ/bw=$SIGMA_OVER_BW"
echo "  N_STEPS=$N_STEPS   STEP_CKPT_EVERY=$STEP_CKPT_EVERY"
echo "  OUT_DIR=$OUT_DIR"

# COMPILE=0 disables torch.compile (safer for short smoke).
COMPILE_ARGS=(compile=false)

# Resume-on-restart:
#   1. Prefer latest.pt from CausalPFN's per-epoch Checkpoint callback.
#   2. If that doesn't exist (short smoke didn't complete an epoch), fall back
#      to the highest-numbered step_XXXX.pt from the step-ckpt patch.
RESUME_ARGS=()
FOUND_LATEST="$(find "$OUT_DIR/checkpoints" -maxdepth 2 -name latest.pt 2>/dev/null | head -1 || true)"
if [ -z "$FOUND_LATEST" ]; then
    FOUND_LATEST="$(ls -1 "$OUT_DIR/step_checkpoints/run"/step_*.pt 2>/dev/null | sort -V | tail -1 || true)"
fi
if [ -n "$FOUND_LATEST" ]; then
    RESUME_ARGS+=(resume_training=enabled resume_training.checkpoint_path="$FOUND_LATEST")
    echo "[resume] found $FOUND_LATEST"
else
    echo "[resume] no prior checkpoint — starting from scratch"
fi

cd "$CAUSALPFN"
python -u train.py \
    +experiment=simple_configuration \
    hydra.run.dir="$OUT_DIR/hydra_run" \
    +callbacks@callbacks.checkpoint=checkpoint \
    callbacks.checkpoint.checkpoint_root="$OUT_DIR/checkpoints" \
    callbacks.checkpoint.checkpoint_dir_name=run \
    callbacks.checkpoint.frequency=1 \
    callbacks.checkpoint.top_k=10000 \
    ~callbacks.eval_cate \
    trainer.max_epochs=1 \
    trainer.num_model_updates="$N_STEPS" \
    model.obj._target_=rpfn_patches.cpfn2d_loader.load_cpfn2d_from_tabdpt \
    model.obj.ckpt_path="$WARMSTART_LOCAL" \
    +model.obj.J="$J" \
    +model.obj.num_features="$NUM_FEATURES" \
    +model.obj.hlgauss_sigma="$HLGAUSS_SIGMA" \
    +model.obj.loss_type="$LOSS_TYPE" \
    +model.obj.y_scaling_mode="$Y_SCALING_MODE" \
    +model.obj.edge_lo="$EDGE_LO" \
    +model.obj.edge_hi="$EDGE_HI" \
    +trainer.step_checkpoint_every="$STEP_CKPT_EVERY" \
    +trainer.step_checkpoint_root="$OUT_DIR/step_checkpoints" \
    "${COMPILE_ARGS[@]}" \
    "${RESUME_ARGS[@]}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE"
