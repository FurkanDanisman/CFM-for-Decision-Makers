#!/bin/bash
# Submit the case-study evaluation sweep: all models × context sizes × 6 cases,
# on OUR generated npz. Mirrors benchmarks/cluster/launch_scm_ctx_sweep.sh but
# swaps the data source from DoPFN pkls (DOPFN_DATA_ROOT) to our npz — the
# benchmarks/scm_case_study_dataset.py chokepoint switches to the npz backend
# whenever CASE_STUDY_DATA_ROOT is set (context size = CASE_STUDY_N).
#
# It reuses the SAME per-model dispatcher sbatch the RealCause sweep uses
# (benchmarks/cluster/submit_eval_scm_case_studies.sbatch), so every model runs
# through its verbatim eval logic. Output layout matches aggregate:
#     $SWEEP/ctx<N>/<model>/<case>/<case>_r###.npz
#
# Usage:
#   DEPLOY_ROOT=/scratch/$USER/rpfn_bench_kit \
#     bash case_study/cluster/02_submit_eval.sh <DATA_ROOT> <SWEEP_ROOT>
#
#   DATA_ROOT  : the npz root written by 01_generate.sbatch
#   SWEEP_ROOT : where per-model outputs land (ctx<N>/<model>/<case>/)
#
# Env vars:
#   CONTEXTS   context sizes to evaluate. Default "200 500 1000" (must be a
#              subset of what 01_generate produced).
#   MODELS     which model variants to submit (space separated). Default all:
#              "dopfn_native dopfn_bb cpfn2d cpfn1d graph2d uwyk uwyk_v3a uwyk_noanc"
#   DEP        optional slurm jobid — added as --dependency=afterok:$DEP so the
#              eval waits for 01_generate to finish.
#   SCM_N_QUERY  CATE query cap per realization. Default 100.
#   *_CKPT     checkpoint overrides (see defaults below).
set -euo pipefail

DATA_ROOT="${1:?DATA_ROOT required (npz root from 01_generate)}"
SWEEP="${2:?SWEEP_ROOT required}"

DEPLOY_ROOT="${DEPLOY_ROOT:?DEPLOY_ROOT required}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
SBATCH="$REPO/benchmarks/cluster/submit_eval_scm_case_studies.sbatch"
[ -f "$SBATCH" ] || { echo "FATAL: dispatcher sbatch not found: $SBATCH" >&2; exit 1; }
[ -d "$DATA_ROOT" ] || echo "WARN: DATA_ROOT not present yet: $DATA_ROOT (ok if DEP set)"

CONTEXTS="${CONTEXTS:-200 500 1000}"
MODELS="${MODELS:-dopfn_native dopfn_bb cpfn2d cpfn1d graph2d uwyk uwyk_v3a uwyk_noanc}"
DEP_ARG=""; [ -n "${DEP:-}" ] && DEP_ARG="--dependency=afterok:$DEP"

export DOPFN_ROOT="${DOPFN_ROOT:-$DEPLOY_ROOT/external/dopfn}"
export CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"

CPFN2D_CKPT="${CPFN2D_CKPT:-$DEPLOY_ROOT/cpfn2d_j32_random_A1_h100/step_checkpoints/run/step_0050000.pt}"
CPFN1D_CKPT="${CPFN1D_CKPT:-$DEPLOY_ROOT/causalpfn_j1024_headrand_A1_h100/step_checkpoints/run/step_0050000.pt}"
GRAPH2D_CKPT="${GRAPH2D_CKPT:-$REPO/Required_checkpoints/graph2d_step_50000.pt}"
DOPFNBB_CKPT="${DOPFNBB_CKPT:-$REPO/Required_checkpoints/dopfn_bb_j10_step_150000.pt}"
UWYK_DIR="${UWYK_DIR:-$DEPLOY_ROOT/external/uwyk/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0}"

want() { case " $MODELS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

echo "[launch] data_root=$DATA_ROOT  sweep=$SWEEP  contexts=[$CONTEXTS]  models=[$MODELS]"

for CTX in $CONTEXTS; do
  # npz backend selection: root + which N-subdir. SCM_TRUE_ATE_SHIFT=0 because
  # the shift is already baked into the npz at generation time.
  common="CASE_STUDY_DATA_ROOT=$DATA_ROOT CASE_STUDY_N=$CTX SCM_TRUE_ATE_SHIFT=0 \
SCM_N_QUERY=${SCM_N_QUERY:-100} DOPFN_ROOT=$DOPFN_ROOT CAUSALPFN=$CAUSALPFN DEPLOY_ROOT=$DEPLOY_ROOT REPO=$REPO"

  want dopfn_native && env $common MODEL=dopfn_native CKPT=none \
      OUT_ROOT="$SWEEP/ctx${CTX}/dopfn_native" sbatch $DEP_ARG "$SBATCH"

  want dopfn_bb && env $common MODEL=dopfn_bb CKPT="$DOPFNBB_CKPT" \
      OUT_ROOT="$SWEEP/ctx${CTX}/dopfn_bb" sbatch $DEP_ARG "$SBATCH"

  want cpfn2d && env $common MODEL=cpfn2d CKPT="$CPFN2D_CKPT" STD_MODE=pooled \
      OUT_ROOT="$SWEEP/ctx${CTX}/cpfn2d" sbatch $DEP_ARG "$SBATCH"

  want cpfn1d && env $common MODEL=cpfn1d CKPT="$CPFN1D_CKPT" STD_MODE=per_arm \
      OUT_ROOT="$SWEEP/ctx${CTX}/cpfn1d" sbatch $DEP_ARG "$SBATCH"

  # graph2d case_family emits case-correct noanc + v3a + v3b keys in one npz.
  want graph2d && env $common MODEL=graph2d CKPT="$GRAPH2D_CKPT" GRAPH2D_ANC_MODE=case_family \
      OUT_ROOT="$SWEEP/ctx${CTX}/graph2d" sbatch $DEP_ARG "$SBATCH"

  # uwyk: three ancestor variants — full(=v3b), paper_anc(=v3a), noanc.
  UWYK_COMMON="UWYK=$DEPLOY_ROOT/external/uwyk UWYK_SRC=$DEPLOY_ROOT/external/uwyk/src \
CKPT=$UWYK_DIR/best_model.pt UWYK_CONFIG=$UWYK_DIR/best_model_config.yaml"
  want uwyk && env $common $UWYK_COMMON MODEL=uwyk ANC_VARIANT=full \
      OUT_ROOT="$SWEEP/ctx${CTX}/uwyk" sbatch $DEP_ARG "$SBATCH"
  want uwyk_v3a && env $common $UWYK_COMMON MODEL=uwyk ANC_VARIANT=paper_anc \
      OUT_ROOT="$SWEEP/ctx${CTX}/uwyk_v3a" sbatch $DEP_ARG "$SBATCH"
  want uwyk_noanc && env $common $UWYK_COMMON MODEL=uwyk ANC_VARIANT=noanc \
      OUT_ROOT="$SWEEP/ctx${CTX}/uwyk_noanc" sbatch $DEP_ARG "$SBATCH"
done

echo "[launch] submitted models=[$MODELS] × contexts=[$CONTEXTS] for $DATA_ROOT"
