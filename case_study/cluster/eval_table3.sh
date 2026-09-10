#!/bin/bash
# Table-3-format eval on EITHER the original DoPFN pkls OR our shifted npz.
# Same model+standardization lineup for both, so the two are directly comparable.
#
#   DATA=original     -> DoPFN-pkl loader (no shift); context via SCM_N_TRAIN=N
#   DATA=<npz root>   -> our npz (shift baked in);    context = CASE_STUDY_N=N (N-subdir)
#
# Lineup:
#   dopfn_native            (1D ref, no std knob)
#   dopfn_bb    std/0.3
#   cpfn2d      pooled  AND  per_arm      (two variants)
#   cpfn1d      per_arm AND  pooled       (two variants)
#   graph2d     case_family (emits noanc/v3a/v3b)
#   uwyk        full / paper_anc / noanc
#
# Outputs: <RESULTS_ROOT>/ctx<N>/<model>/<case>/...  (raw + em PEHE, ATE means,
# err_raw/err_em L1 for uniform models, ate_pred/ate_pred_em for dopfn_bb).
#
# Usage:
#   DATA=original                              DEPLOY_ROOT=$SCRATCH/rpfn_bench_kit \
#     bash case_study/cluster/eval_table3.sh  $RES/original
#   DATA=$DEPLOY_ROOT/case_study_data/shift+2 DEPLOY_ROOT=$SCRATCH/rpfn_bench_kit \
#     bash case_study/cluster/eval_table3.sh  $RES/shift+2
# Env: N (default 500), MODELS.
set -euo pipefail

ROOT="${1:?RESULTS_ROOT required}"
DATA="${DATA:?DATA required: 'original' or an npz root path}"
DEPLOY_ROOT="${DEPLOY_ROOT:?DEPLOY_ROOT required}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
SB="$REPO/benchmarks/cluster/submit_eval_scm_case_studies.sbatch"
N="${N:-500}"

export DOPFN_ROOT="${DOPFN_ROOT:-$DEPLOY_ROOT/external/dopfn}"
export CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"

# ── data-source selection ──
if [ "$DATA" = "original" ]; then
  export DOPFN_DATA_ROOT="${DOPFN_DATA_ROOT:-$DOPFN_ROOT/data/prior_sampling}"
  [ -d "$DOPFN_DATA_ROOT" ] || { echo "FATAL: original data not found: $DOPFN_DATA_ROOT" >&2; exit 1; }
  DATA_ENV="DOPFN_DATA_ROOT=$DOPFN_DATA_ROOT SCM_N_TRAIN=$N"
  echo "[eval_table3] ORIGINAL DoPFN pkls  $DOPFN_DATA_ROOT  N(SCM_N_TRAIN)=$N"
else
  [ -d "$DATA" ] || { echo "FATAL: npz root not found: $DATA" >&2; exit 1; }
  DATA_ENV="CASE_STUDY_DATA_ROOT=$DATA CASE_STUDY_N=$N"
  echo "[eval_table3] OUR npz  $DATA  N(CASE_STUDY_N)=$N"
fi

CPFN2D_CKPT="${CPFN2D_CKPT:-$DEPLOY_ROOT/cpfn2d_j32_random_A1_h100/step_checkpoints/run/step_0050000.pt}"
CPFN1D_CKPT="${CPFN1D_CKPT:-$DEPLOY_ROOT/causalpfn_j1024_headrand_A1_h100/step_checkpoints/run/step_0050000.pt}"
GRAPH2D_CKPT="${GRAPH2D_CKPT:-$REPO/Required_checkpoints/graph2d_step_50000.pt}"
DOPFNBB_CKPT="${DOPFNBB_CKPT:-$REPO/Required_checkpoints/dopfn_bb_j10_step_150000.pt}"
UWYK_DIR="${UWYK_DIR:-$DEPLOY_ROOT/external/uwyk/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0}"
MODELS="${MODELS:-native bb cpfn2d cpfn1d graph2d uwyk}"

# shift already baked into the npz (0 for original) — never double-shift.
common="SCM_TRUE_ATE_SHIFT=0 SCM_N_QUERY=100 $DATA_ENV \
DOPFN_ROOT=$DOPFN_ROOT CAUSALPFN=$CAUSALPFN DEPLOY_ROOT=$DEPLOY_ROOT REPO=$REPO"
sub() { env $common "$@" sbatch --array=0-5 "$SB" >/dev/null; }
want() { case " $MODELS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
C="$ROOT/ctx${N}"

want native && sub MODEL=dopfn_native CKPT=none OUT_ROOT="$C/dopfn_native"

want bb && sub MODEL=dopfn_bb CKPT="$DOPFNBB_CKPT" BB_Y_SCALING=std BB_STD_TARGET=0.3 \
    OUT_ROOT="$C/dopfn_bb"

# cpfn2d: pooled and log (log1p(Y-min) then pooled std on log-Y).
want cpfn2d && sub MODEL=cpfn2d CKPT="$CPFN2D_CKPT" STD_MODE=pooled OUT_ROOT="$C/cpfn2d_pooled"
want cpfn2d && sub MODEL=cpfn2d CKPT="$CPFN2D_CKPT" STD_MODE=log    OUT_ROOT="$C/cpfn2d_log"

# cpfn1d: per_arm AND pooled.
want cpfn1d && sub MODEL=cpfn1d CKPT="$CPFN1D_CKPT" STD_MODE=per_arm OUT_ROOT="$C/cpfn1d_perarm"
want cpfn1d && sub MODEL=cpfn1d CKPT="$CPFN1D_CKPT" STD_MODE=pooled  OUT_ROOT="$C/cpfn1d_pooled"

want graph2d && sub MODEL=graph2d CKPT="$GRAPH2D_CKPT" GRAPH2D_ANC_MODE=case_family \
    OUT_ROOT="$C/graph2d"

if want uwyk; then
  UW="UWYK=$DEPLOY_ROOT/external/uwyk UWYK_SRC=$DEPLOY_ROOT/external/uwyk/src \
CKPT=$UWYK_DIR/best_model.pt UWYK_CONFIG=$UWYK_DIR/best_model_config.yaml"
  sub MODEL=uwyk $UW ANC_VARIANT=full      OUT_ROOT="$C/uwyk"
  sub MODEL=uwyk $UW ANC_VARIANT=paper_anc OUT_ROOT="$C/uwyk_v3a"
  sub MODEL=uwyk $UW ANC_VARIANT=noanc     OUT_ROOT="$C/uwyk_noanc"
fi

echo "[eval_table3] submitted. Report when done:"
echo "  python $REPO/case_study/cluster/orig_report.py --root $ROOT --n $N"
