#!/bin/bash
#SBATCH --job-name=ate-bary
#SBATCH --output=logs_ate/%x_%j.out
#SBATCH --error=logs_ate/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# Scale-calibrated OT ATE density on in-prior data (E2 "new" data, shards 0..ATE_SHARDS-1, 100 datasets
# each, every model on its own prior; UWYK 1D on its binarized-T prior and UWYK 2D on its own, with the
# settings of eval_e2.sbatch, scored separately for the No-Anc (noanc) and Anc (v3a) densities). One job per model: one full-context pass per shard with the
# harness of E2_in_prior/eval_e2.sbatch (Do-PFN 2D with the inference temperature off), reduced to each
# dataset's barycenter quantiles (reduce_bary.py), dumps deleted; then the split-conformal scale
# (conformal_bary.py). Finished shards are skipped, so a timed-out model is simply resubmitted.
# From the kit root (no dependency jobs):
#   for M in dopfn_native dopfn_joint2d_262k cpfn1d cpfn2d uwyk1d graph2d; do ATE_MODEL=$M sbatch --account=aip-rgrosse \
#     --gres=gpu:l40s:1 R-PFN/revision/experiments/ate_subsample/run_ate_bary.sbatch; done
set -uo pipefail
mkdir -p logs_ate
KIT="${KIT:-$PWD}"; REPO="$KIT/R-PFN"; CK="$REPO/Required_checkpoints"
HERE="$REPO/revision/experiments/ate_subsample"
export CAUSALPFN="$KIT/external/causalpfn" DOPFN_ROOT="$KIT/external/dopfn"
source "$KIT/venv/bin/activate"
export PYTHONUNBUFFERED=1 PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
export DATASET=CMECH_n5 DENSITY_DUMP=1
MODEL="${ATE_MODEL:?ATE_MODEL = dopfn_native | dopfn_joint2d_262k | cpfn1d | cpfn2d | uwyk1d | graph2d}"
UWYK="$KIT/external/uwyk_reproduce"
UWYK_CKPT_DIR="$UWYK/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0"
TAGS=("")                                   # UWYK dumps carry both modes: one result directory per tag
NSH="${ATE_SHARDS:-5}"
case "$MODEL" in
  dopfn_native|dopfn_joint2d_262k) PRIOR=dopfn ;;
  cpfn1d) PRIOR=cpfn_1d ;;
  cpfn2d) PRIOR=cpfn_2d ;;
  uwyk1d) PRIOR=uwyk_1d; TAGS=(noanc v3a) ;;
  graph2d) PRIOR=uwyk_2d; TAGS=(noanc v3a) ;;
  *) echo "unknown model $MODEL"; exit 1 ;;
esac
BASE="${ATE_RESULTS:-$SCRATCH/ate_bary}"
res_of () { echo "$BASE/$MODEL${1:+-$1}"; }   # $1 = tag
for t in "${TAGS[@]}"; do mkdir -p "$(res_of "$t")"; done
RES="$(res_of "${TAGS[0]}")"

for SHARD in $(seq 0 $(( NSH - 1 ))); do
  F="$RES/shard$SHARD.npz"; done_all=1
  for t in "${TAGS[@]}"; do [ -f "$(res_of "$t")/shard$SHARD.npz" ] || done_all=0; done
  [ "$done_all" = 1 ] && continue
  [ "$SECONDS" -gt 9600 ] && { echo "[$(date '+%F %T')] stopping before the time limit; resubmit to continue"; exit 0; }
  export UWYK_FIG34_DATA="${E2_DATA:-$SCRATCH/e2/new/data}/$PRIOR/shard$SHARD"
  [ -d "$UWYK_FIG34_DATA" ] || { echo "no data at $UWYK_FIG34_DATA"; exit 1; }
  export OUT="$RES/dumps_$SHARD"; rm -rf "${OUT:?}"; mkdir -p "$OUT"
  echo "[$(date '+%F %T')] model=$MODEL shard=$SHARD"
  case "$MODEL" in
    dopfn_native)
      python -u "$REPO/benchmarks/eval_scm_case_studies/eval_native_dopfn.py" --dataset "$DATASET" ;;
    dopfn_joint2d_262k)
      DOPFN_SOFTMAX_TEMP=0 DOPFN_CKPT="$CK/dopfn_repro_joint2d_step262144.pt" \
        python -u "$REPO/benchmarks/eval_scm_case_studies/eval_native_dopfn.py" --dataset "$DATASET" ;;
    cpfn1d)
      CKPT="$CK/cpfn1d_j1024_headrand_step_50000.pt" EVAL_MAX_CONTEXT= EVAL_CONTEXT_SEED=1 \
      STD_MODE=per_arm COMPILE=0 SKIP_EM=1 \
        python -u "$REPO/case_study/density_eval/eval_cpfn1d_perarm.py" --dataset "$DATASET" ;;
    cpfn2d)
      python -u "$REPO/realcause_eval/Table1/cpfn2d.py" --dataset "$DATASET" --outdir "$OUT" \
        --density-dump 1 --ckpt "$CK/cpfn2d_j32_eta0_y01_step50000.pt" \
        --repo "$REPO" --causalpfn "$CAUSALPFN" --std-mode pooled ;;
    uwyk1d)
      python -u "$REPO/realcause_eval/Table1/uwyk1d.py" --dataset "$DATASET" --outdir "$OUT" \
        --uwyk-repro "$UWYK" --repo "$REPO" --causalpfn "$CAUSALPFN" \
        --ckpt "$UWYK_CKPT_DIR/best_model.pt" --config "$UWYK_CKPT_DIR/best_model_config.yaml" \
        --anc-mode v3ab_only --t-encoding target --eval-max-context 1000 --eval-context-seed 43 ;;
    graph2d)
      python -u "$REPO/realcause_eval/Table1/graph2d.py" --dataset "$DATASET" --outdir "$OUT" \
        --density-dump 1 --ckpt "$CK/graph2d_step_50000.pt" \
        --repo "$REPO" --causalpfn "$CAUSALPFN" --uwyk "$UWYK" \
        --anc-mode v3ab_only --density-primary-mode noanc --eval-max-context 1000 --eval-context-seed 43 ;;
  esac || { echo "harness failed at shard $SHARD"; exit 1; }
  for t in "${TAGS[@]}"; do
    python "$HERE/reduce_bary.py" "$OUT" "$(res_of "$t")/shard$SHARD.npz" $t || { echo "reduce failed at shard $SHARD"; exit 1; }
  done
  rm -rf "${OUT:?}"
done
for t in "${TAGS[@]}"; do python "$HERE/conformal_bary.py" "$(res_of "$t")" | tee "$(res_of "$t")/coverage.txt"; done
