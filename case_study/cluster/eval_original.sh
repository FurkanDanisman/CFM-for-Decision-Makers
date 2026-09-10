#!/bin/bash
# Evaluate our models on the ORIGINAL DoPFN case-study pkls (not our npz).
# Key difference from the other launchers: CASE_STUDY_DATA_ROOT is NOT set, so
# benchmarks/scm_case_study_dataset.py uses its default DoPFN-pkl loader
# (DOPFN_DATA_ROOT). No shift (SCM_TRUE_ATE_SHIFT=0) — original data as-is.
#
# Standardization: dopfn_bb std/0.3 (paper/Table-1 setting); cpfn2d pooled
# (its established setting in every prior run — NOT changed here); cpfn1d run
# BOTH per_arm and pooled. Override cpfn2d via CPFN2D_STD_MODE / CPFN2D_STD_TARGET.
#
# Usage:
#   DEPLOY_ROOT=$SCRATCH/rpfn_bench_kit \
#     bash case_study/cluster/eval_original.sh <RESULTS_ROOT>
# Env: N (SCM_N_TRAIN context, default 200), MODELS (default all).
set -euo pipefail

ROOT="${1:?RESULTS_ROOT required}"
DEPLOY_ROOT="${DEPLOY_ROOT:?DEPLOY_ROOT required}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
SB="$REPO/benchmarks/cluster/submit_eval_scm_case_studies.sbatch"
N="${N:-200}"

export DOPFN_ROOT="${DOPFN_ROOT:-$DEPLOY_ROOT/external/dopfn}"
export DOPFN_DATA_ROOT="${DOPFN_DATA_ROOT:-$DOPFN_ROOT/data/prior_sampling}"
export CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
[ -d "$DOPFN_DATA_ROOT" ] || { echo "FATAL: original data not found: $DOPFN_DATA_ROOT" >&2; exit 1; }

CPFN2D_CKPT="${CPFN2D_CKPT:-$DEPLOY_ROOT/cpfn2d_j32_random_A1_h100/step_checkpoints/run/step_0050000.pt}"
CPFN1D_CKPT="${CPFN1D_CKPT:-$DEPLOY_ROOT/causalpfn_j1024_headrand_A1_h100/step_checkpoints/run/step_0050000.pt}"
GRAPH2D_CKPT="${GRAPH2D_CKPT:-$REPO/Required_checkpoints/graph2d_step_50000.pt}"
DOPFNBB_CKPT="${DOPFNBB_CKPT:-$REPO/Required_checkpoints/dopfn_bb_j10_step_150000.pt}"
UWYK_DIR="${UWYK_DIR:-$DEPLOY_ROOT/external/uwyk/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0}"
MODELS="${MODELS:-native bb cpfn2d cpfn1d graph2d uwyk}"

# ORIGINAL data: no CASE_STUDY_DATA_ROOT, no shift. Context via SCM_N_TRAIN.
common="SCM_TRUE_ATE_SHIFT=0 SCM_N_TRAIN=$N SCM_N_QUERY=100 \
DOPFN_ROOT=$DOPFN_ROOT DOPFN_DATA_ROOT=$DOPFN_DATA_ROOT CAUSALPFN=$CAUSALPFN \
DEPLOY_ROOT=$DEPLOY_ROOT REPO=$REPO"
sub() { env $common "$@" sbatch --array=0-5 "$SB" >/dev/null; }
want() { case " $MODELS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
C="$ROOT/ctx${N}"

echo "[eval_original] data=$DOPFN_DATA_ROOT  N=$N  out=$C  models=[$MODELS]"

want native && sub MODEL=dopfn_native CKPT=none OUT_ROOT="$C/dopfn_native"

want bb && sub MODEL=dopfn_bb CKPT="$DOPFNBB_CKPT" BB_Y_SCALING=std BB_STD_TARGET=0.3 \
    OUT_ROOT="$C/dopfn_bb"

want cpfn2d && sub MODEL=cpfn2d CKPT="$CPFN2D_CKPT" \
    STD_MODE="${CPFN2D_STD_MODE:-pooled}" STD_TARGET="${CPFN2D_STD_TARGET:-1}" \
    OUT_ROOT="$C/cpfn2d"

# cpfn1d: BOTH per-arm and pooled.
want cpfn1d && sub MODEL=cpfn1d CKPT="$CPFN1D_CKPT" STD_MODE=per_arm \
    OUT_ROOT="$C/cpfn1d_perarm"
want cpfn1d && sub MODEL=cpfn1d CKPT="$CPFN1D_CKPT" STD_MODE=pooled \
    OUT_ROOT="$C/cpfn1d_pooled"

want graph2d && sub MODEL=graph2d CKPT="$GRAPH2D_CKPT" GRAPH2D_ANC_MODE=case_family \
    OUT_ROOT="$C/graph2d"

if want uwyk; then
  UW="UWYK=$DEPLOY_ROOT/external/uwyk UWYK_SRC=$DEPLOY_ROOT/external/uwyk/src \
CKPT=$UWYK_DIR/best_model.pt UWYK_CONFIG=$UWYK_DIR/best_model_config.yaml"
  sub MODEL=uwyk $UW ANC_VARIANT=full      OUT_ROOT="$C/uwyk"
  sub MODEL=uwyk $UW ANC_VARIANT=paper_anc OUT_ROOT="$C/uwyk_v3a"
  sub MODEL=uwyk $UW ANC_VARIANT=noanc     OUT_ROOT="$C/uwyk_noanc"
fi

echo "[eval_original] submitted. Report when done:"
echo "  python $REPO/case_study/cluster/orig_report.py --root $ROOT --n $N"
