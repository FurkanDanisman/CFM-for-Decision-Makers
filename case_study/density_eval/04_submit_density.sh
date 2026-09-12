#!/bin/bash
# Submit the DENSITY sweep (coverage / length / Winkler / CRPS) on case-study npz.
#
# Mirrors case_study/cluster/02_submit_eval.sh, but every model runs with its
# DENSITY path enabled and nothing outside case_study/ is modified:
#
#   cpfn1d   -> case_study/density_eval/eval_cpfn1d_perarm.py  (STD_MODE=per_arm)
#               Upstream's dump asserts pooled; the local copy saves per-arm
#               (shift, scale) so per_arm can be dumped. Point logic verbatim.
#   cpfn2d   -> the upstream 2D eval with DENSITY_DUMP=1 (needs non-per_arm)
#   graph2d / uwyk* / dopfn_* -> case_study/density_eval/eval_density_tauC.py,
#               whose sys.path puts THIS folder first, so density_truth.py here
#               (SCM-aware) shadows the IHDP/ACIC-only original.
#
# Usage:
#   export DEPLOY_ROOT=$SCRATCH/rpfn_bench_kit
#   bash $DEPLOY_ROOT/R-PFN/case_study/density_eval/04_submit_density.sh \
#        $DEPLOY_ROOT/case_study_data/shift+2 \
#        $DEPLOY_ROOT/results_density/shift+2
#
# Env:
#   CONTEXTS  default "200 500 1000"
#   MODELS    default "cpfn1d cpfn2d graph2d uwyk uwyk_v3a uwyk_noanc dopfn_native dopfn_bb"
#   CASES     default all six
#   DRY_RUN=1 print the sbatch lines instead of submitting
set -euo pipefail

DATA_ROOT="${1:?DATA_ROOT required (npz root, e.g. .../case_study_data/shift+2)}"
SWEEP="${2:?SWEEP_ROOT required}"
DEPLOY_ROOT="${DEPLOY_ROOT:?DEPLOY_ROOT required}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
HERE="$REPO/case_study/density_eval"
SB="$HERE/submit_density.sbatch"

for f in "$SB" "$HERE/run_density_scm.py" "$HERE/density_truth.py" \
         "$HERE/eval_cpfn1d_perarm.py"; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done
[ -d "$DATA_ROOT" ] || { echo "FATAL: DATA_ROOT not found: $DATA_ROOT" >&2; exit 1; }

CONTEXTS="${CONTEXTS:-200 500 1000}"
MODELS="${MODELS:-cpfn1d cpfn2d graph2d uwyk uwyk_v3a uwyk_noanc dopfn_native dopfn_bb}"
CASES="${CASES:-Observed_Confounder Backdoor_Criterion Observed_Mediator Observed_Mediator_and_Confounder Unobserved_Confounder Frontdoor_Criterion}"

export DOPFN_ROOT="${DOPFN_ROOT:-$DEPLOY_ROOT/external/dopfn}"
export CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
export UWYK="${UWYK:-$DEPLOY_ROOT/external/uwyk}"
CPFN2D_CKPT="${CPFN2D_CKPT:-$DEPLOY_ROOT/cpfn2d_j32_random_A1_h100/step_checkpoints/run/step_0050000.pt}"
CPFN1D_CKPT="${CPFN1D_CKPT:-$DEPLOY_ROOT/causalpfn_j1024_headrand_A1_h100/step_checkpoints/run/step_0050000.pt}"
GRAPH2D_CKPT="${GRAPH2D_CKPT:-$REPO/Required_checkpoints/graph2d_step_50000.pt}"
DOPFNBB_CKPT="${DOPFNBB_CKPT:-$REPO/Required_checkpoints/dopfn_bb_j10_step_150000.pt}"
UWYK_DIR="${UWYK_DIR:-$UWYK/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0}"

want() { case " $MODELS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
sub() {  # sub <name> <env assignments...>
    local name="$1"; shift
    if [ "${DRY_RUN:-0}" = 1 ]; then echo "  [dry] $name: $*"; return; fi
    env "$@" sbatch --job-name="dens-$name" "$SB" >/dev/null
    echo "  submitted $name"
}

echo "[density] data=$DATA_ROOT sweep=$SWEEP contexts=[$CONTEXTS] models=[$MODELS]"
mkdir -p "$SWEEP" logs_density

for CTX in $CONTEXTS; do
  COMMON="REPO=$REPO HERE=$HERE DATA_ROOT=$DATA_ROOT CASE_STUDY_DATA_ROOT=$DATA_ROOT \
CASE_STUDY_N=$CTX SCM_N_QUERY=${SCM_N_QUERY:-100} DEPLOY_ROOT=$DEPLOY_ROOT \
DOPFN_ROOT=$DOPFN_ROOT CAUSALPFN=$CAUSALPFN UWYK=$UWYK CASES=$(echo $CASES | tr ' ' ',') \
DENSITY_DUMP=1"

  # --- CausalPFN: histogram dumps, scored by run_density_cpfn.py -------------
  # per_arm as requested; the local eval copy is what makes that dumpable.
  want cpfn1d && sub "cpfn1d-c$CTX" $COMMON KIND=cpfn1d STD_MODE=per_arm \
      CKPT="$CPFN1D_CKPT" OUT="$SWEEP/ctx${CTX}/cpfn1d"
  # 2D dump refuses per_arm (one shared axis is required for the joint).
  want cpfn2d && sub "cpfn2d-c$CTX" $COMMON KIND=cpfn2d STD_MODE=pooled \
      CKPT="$CPFN2D_CKPT" OUT="$SWEEP/ctx${CTX}/cpfn2d"

  # --- tauC path: graph2d / uwyk variants / dopfn ---------------------------
  want graph2d && sub "graph2d-c$CTX" $COMMON KIND=tauC MODEL_FAMILY=uwyk \
      CKPT="$GRAPH2D_CKPT" ANC_VARIANT=full OUT="$SWEEP/ctx${CTX}/graph2d"
  UW="UWYK_CKPT=$UWYK_DIR/best_model.pt UWYK_CFG=$UWYK_DIR/best_model_config.yaml \
CKPT=$GRAPH2D_CKPT"
  want uwyk       && sub "uwyk-c$CTX"       $COMMON KIND=tauC MODEL_FAMILY=uwyk $UW \
      ANC_VARIANT=full      OUT="$SWEEP/ctx${CTX}/uwyk"
  want uwyk_v3a   && sub "uwykv3a-c$CTX"    $COMMON KIND=tauC MODEL_FAMILY=uwyk $UW \
      ANC_VARIANT=paper_anc OUT="$SWEEP/ctx${CTX}/uwyk_v3a"
  want uwyk_noanc && sub "uwyknoanc-c$CTX"  $COMMON KIND=tauC MODEL_FAMILY=uwyk $UW \
      ANC_VARIANT=noanc     OUT="$SWEEP/ctx${CTX}/uwyk_noanc"
  want dopfn_native && sub "dopfnnat-c$CTX" $COMMON KIND=tauC MODEL_FAMILY=dopfn \
      OUT="$SWEEP/ctx${CTX}/dopfn_native"
  want dopfn_bb && sub "dopfnbb-c$CTX" $COMMON KIND=tauC MODEL_FAMILY=dopfn \
      DOPFN_JOINT_CKPT="$DOPFNBB_CKPT" OUT="$SWEEP/ctx${CTX}/dopfn_bb"
done
echo "[density] done. Aggregate with 05_aggregate_density.py once jobs finish."
