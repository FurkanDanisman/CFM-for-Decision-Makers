#!/bin/bash
#SBATCH --job-name=ate-bary
#SBATCH --output=logs_ate/%x_%j.out
#SBATCH --error=logs_ate/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
# In-prior calibration of the OT ATE density (E2 "new" data, shards 0..ATE_SHARDS-1, 100 datasets each; every
# model on its own prior with the settings of E2_in_prior/eval_e2.sbatch). One job per model: one full-context
# pass per shard, reduced to each dataset's barycenter for every variant (reduce_bary.py): UWYK's No-Anc and Anc
# densities (tags noanc, v3a), and for single-outcome models the paper's assumed rho in {-0.5, 0, 0.5, 1}
# (Gaussian copula; rho = 0 = independence). Then the split-conformal scale per variant (conformal_bary.py).
# Result directories: $SCRATCH/ate_bary/<model>[-<tag>][-rho<r>]/. Finished shards are skipped; a job stops
# itself before the time limit, so a timed-out model is resubmitted (collect_rho.sbatch does this).
#   ATE_MODEL=<model> sbatch --account=aip-rgrosse --gres=gpu:l40s:1 R-PFN/revision/experiments/ate_subsample/run_ate_bary.sbatch
set -uo pipefail
mkdir -p logs_ate
KIT="${KIT:-$PWD}"; REPO="$KIT/R-PFN"; CK="$REPO/Required_checkpoints"; FC="$KIT/final_checkpoints_paper"
HERE="$REPO/revision/experiments/ate_subsample"
export CAUSALPFN="$KIT/external/causalpfn" DOPFN_ROOT="$KIT/external/dopfn"
UWYK="$KIT/external/uwyk_reproduce"
UWYK_CKPT_DIR="$UWYK/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0"
source "$KIT/venv/bin/activate"
export PYTHONUNBUFFERED=1 PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
export DATASET=CMECH_n5 DENSITY_DUMP=1
MODEL="${ATE_MODEL:?ATE_MODEL required}"
NSH="${ATE_SHARDS:-5}"; T0="${ATE_T0:-$(date +%s)}"
RHO1D="indep,gauss:-0.5,gauss:0.5,gauss:1"; RLAB=(0 m0.5 0.5 1)
TAGS=""; COUP="$RHO1D"
case "$MODEL" in
  dopfn_native|dopfn_joint2d_262k|dopfn1d_mr|dopfn1d_ml) PRIOR=dopfn ;;
  cpfn1d) PRIOR=cpfn_1d ;;
  cpfn2d|cpfn1d_mr|cpfn1d_ml) PRIOR=cpfn_2d ;;
  uwyk1d) PRIOR=uwyk_1d; TAGS="noanc,v3a" ;;
  graph2d|uwyk1d_mr|uwyk1d_ml) PRIOR=uwyk_2d; TAGS="noanc,v3a" ;;
  *) echo "unknown model $MODEL"; exit 1 ;;
esac
case "$MODEL" in dopfn_joint2d_262k|cpfn2d|graph2d) COUP="" ;; esac     # 2D: the learned joint
BASE="${ATE_RESULTS:-$SCRATCH/ate_bary}"
VARS=()                                                                    # result directory names
IFS=, read -r -a TL <<<"${TAGS:-}"; [ ${#TL[@]} -eq 0 ] && TL=("")
for t in "${TL[@]}"; do
  if [ -z "$COUP" ]; then VARS+=("$MODEL${t:+-$t}"); else for r in "${RLAB[@]}"; do VARS+=("$MODEL${t:+-$t}-rho$r"); done; fi
done
echo "variants: ${VARS[*]}"

for SHARD in $(seq 0 $(( NSH - 1 ))); do
  done_all=1; for v in "${VARS[@]}"; do [ -f "$BASE/$v/shard$SHARD.npz" ] || done_all=0; done
  [ "$done_all" = 1 ] && continue
  [ $(( $(date +%s) - T0 )) -gt 9000 ] && { echo "[$(date '+%F %T')] stopping before the time limit; resubmit to continue"; exit 0; }
  export UWYK_FIG34_DATA="${E2_DATA:-$SCRATCH/e2/new/data}/$PRIOR/shard$SHARD"
  [ -d "$UWYK_FIG34_DATA" ] || { echo "no data at $UWYK_FIG34_DATA"; exit 1; }
  export OUT="$BASE/_dumps/$MODEL/shard$SHARD"; rm -rf "${OUT:?}"; mkdir -p "$OUT"
  echo "[$(date '+%F %T')] model=$MODEL shard=$SHARD"
  case "$MODEL" in
    dopfn_native)
      python -u "$REPO/benchmarks/eval_scm_case_studies/eval_native_dopfn.py" --dataset "$DATASET" ;;
    dopfn_joint2d_262k|dopfn1d_mr|dopfn1d_ml)
      case "$MODEL" in dopfn_joint2d_262k) K="$CK/dopfn_repro_joint2d_step262144.pt" ;;
        dopfn1d_mr) K="$FC/dopfn1d_J10_step262144.pt" ;; dopfn1d_ml) K="$FC/dopfn1d_botharms_step262144.pt" ;; esac
      DOPFN_SOFTMAX_TEMP=0 DOPFN_CKPT="$K" \
        python -u "$REPO/benchmarks/eval_scm_case_studies/eval_native_dopfn.py" --dataset "$DATASET" ;;
    cpfn1d|cpfn1d_mr|cpfn1d_ml)
      case "$MODEL" in cpfn1d) K="$CK/cpfn1d_j1024_headrand_step_50000.pt" ;;
        cpfn1d_mr) K="$FC/cpfnC_mr_1d_j32_step50000.pt" ;; cpfn1d_ml) K="$FC/cpfnC_ml_1d_botharms_step50000.pt" ;; esac
      CKPT="$K" EVAL_MAX_CONTEXT= EVAL_CONTEXT_SEED=1 STD_MODE=per_arm COMPILE=0 SKIP_EM=1 \
        python -u "$REPO/case_study/density_eval/eval_cpfn1d_perarm.py" --dataset "$DATASET" ;;
    cpfn2d)
      python -u "$REPO/realcause_eval/Table1/cpfn2d.py" --dataset "$DATASET" --outdir "$OUT" \
        --density-dump 1 --ckpt "$CK/cpfn2d_j32_eta0_y01_step50000.pt" \
        --repo "$REPO" --causalpfn "$CAUSALPFN" --std-mode pooled ;;
    uwyk1d|uwyk1d_mr|uwyk1d_ml)
      case "$MODEL" in
        uwyk1d)    K="$UWYK_CKPT_DIR/best_model.pt"; C="$UWYK_CKPT_DIR/best_model_config.yaml"; E=target ;;
        uwyk1d_mr) K="$FC/uwyk_mr_D_j32_step50000.pt"; C="$FC/uwyk_mr_D_j32_config.yaml"; E=target ;;
        uwyk1d_ml) K="$FC/uwyk_ml_C_botharms_step50000.pt"; C="$FC/uwyk_ml_C_botharms_config.yaml"; E=binary ;;
      esac
      python -u "$REPO/realcause_eval/Table1/uwyk1d.py" --dataset "$DATASET" --outdir "$OUT" \
        --uwyk-repro "$UWYK" --repo "$REPO" --causalpfn "$CAUSALPFN" --ckpt "$K" --config "$C" \
        --anc-mode v3ab_only --t-encoding "$E" --eval-max-context 1000 --eval-context-seed 43 ;;
    graph2d)
      python -u "$REPO/realcause_eval/Table1/graph2d.py" --dataset "$DATASET" --outdir "$OUT" \
        --density-dump 1 --ckpt "$CK/graph2d_step_50000.pt" \
        --repo "$REPO" --causalpfn "$CAUSALPFN" --uwyk "$UWYK" \
        --anc-mode v3ab_only --density-primary-mode noanc --eval-max-context 1000 --eval-context-seed 43 ;;
  esac || { echo "harness failed at shard $SHARD"; exit 1; }
  python "$HERE/reduce_bary.py" "$OUT" --base "$BASE/$MODEL" --shard "$SHARD" --tags "$TAGS" --couplings "$COUP" \
    || { echo "reduce failed at shard $SHARD"; exit 1; }
  rm -rf "${OUT:?}"
done
for v in "${VARS[@]}"; do python "$HERE/conformal_bary.py" "$BASE/$v" | tee "$BASE/$v/coverage.txt"; done
