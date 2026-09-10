#!/bin/bash
# Launch all models × 5 context sizes for ONE case-study dataset version.
#
# Usage:
#   DEPLOY_ROOT=/scratch/.../rpfn_bench_kit \
#     bash benchmarks/cluster/launch_scm_ctx_sweep.sh <DATA_SUBDIR> <SWEEP_ROOT>
#
#   DATA_SUBDIR : subdir under $DEPLOY_ROOT/external/dopfn/data/
#                 (e.g. prior_sampling_norm | prior_sampling_ate5 | prior_sampling_ate50)
#   SWEEP_ROOT  : where per-model outputs land (ctx<N>/<model>/<case>/)
#
# Models launched (7 sbatch arrays × 5 contexts = 35 array-jobs):
#   dopfn_native, dopfn_bb, cpfn2d, cpfn1d, graph2d (emits v3b+noanc),
#   uwyk (v3b = ANC_VARIANT=full), uwyk_noanc (ANC_VARIANT=noanc)
set -euo pipefail

SUBDIR="${1:?DATA_SUBDIR required}"
SWEEP="${2:?SWEEP_ROOT required}"

DEPLOY_ROOT="${DEPLOY_ROOT:?DEPLOY_ROOT required}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
SBATCH="$REPO/benchmarks/cluster/submit_eval_scm_case_studies.sbatch"

export DOPFN_DATA_ROOT="$DEPLOY_ROOT/external/dopfn/data/$SUBDIR"
export DOPFN_ROOT="$DEPLOY_ROOT/external/dopfn"
export CAUSALPFN="$DEPLOY_ROOT/external/causalpfn"

[ -d "$DOPFN_DATA_ROOT" ] || { echo "FATAL: data dir not found: $DOPFN_DATA_ROOT" >&2; exit 1; }

CPFN2D_CKPT="$DEPLOY_ROOT/cpfn2d_j32_random_A1_h100/step_checkpoints/run/step_0050000.pt"
CPFN1D_CKPT="$DEPLOY_ROOT/causalpfn_j1024_headrand_A1_h100/step_checkpoints/run/step_0050000.pt"
GRAPH2D_CKPT="$REPO/Required_checkpoints/graph2d_step_50000.pt"
DOPFNBB_CKPT="$REPO/Required_checkpoints/dopfn_bb_j10_step_150000.pt"
UWYK_DIR="$DEPLOY_ROOT/external/uwyk/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0"

echo "[launch] dataset=$SUBDIR  sweep=$SWEEP"

for CTX in 50 100 250 500 1000; do
  common="SCM_N_TRAIN=$CTX SCM_N_QUERY=100 DOPFN_DATA_ROOT=$DOPFN_DATA_ROOT DOPFN_ROOT=$DOPFN_ROOT CAUSALPFN=$CAUSALPFN DEPLOY_ROOT=$DEPLOY_ROOT"

  env $common MODEL=dopfn_native CKPT=none \
      OUT_ROOT="$SWEEP/ctx${CTX}/dopfn_native" sbatch "$SBATCH"

  env $common MODEL=dopfn_bb CKPT="$DOPFNBB_CKPT" \
      OUT_ROOT="$SWEEP/ctx${CTX}/dopfn_bb" sbatch "$SBATCH"

  env $common MODEL=cpfn2d CKPT="$CPFN2D_CKPT" STD_MODE=pooled \
      OUT_ROOT="$SWEEP/ctx${CTX}/cpfn2d" sbatch "$SBATCH"

  env $common MODEL=cpfn1d CKPT="$CPFN1D_CKPT" STD_MODE=per_arm \
      OUT_ROOT="$SWEEP/ctx${CTX}/cpfn1d" sbatch "$SBATCH"

  env $common MODEL=graph2d CKPT="$GRAPH2D_CKPT" GRAPH2D_ANC_MODE=v3b_only \
      OUT_ROOT="$SWEEP/ctx${CTX}/graph2d" sbatch "$SBATCH"

  env $common MODEL=uwyk CKPT="$UWYK_DIR/best_model.pt" \
      UWYK_CONFIG="$UWYK_DIR/best_model_config.yaml" ANC_VARIANT=full \
      UWYK="$DEPLOY_ROOT/external/uwyk" UWYK_SRC="$DEPLOY_ROOT/external/uwyk/src" \
      OUT_ROOT="$SWEEP/ctx${CTX}/uwyk" sbatch "$SBATCH"

  env $common MODEL=uwyk CKPT="$UWYK_DIR/best_model.pt" \
      UWYK_CONFIG="$UWYK_DIR/best_model_config.yaml" ANC_VARIANT=noanc \
      UWYK="$DEPLOY_ROOT/external/uwyk" UWYK_SRC="$DEPLOY_ROOT/external/uwyk/src" \
      OUT_ROOT="$SWEEP/ctx${CTX}/uwyk_noanc" sbatch "$SBATCH"
done

echo "[launch] submitted 7 models × 5 contexts for $SUBDIR"
