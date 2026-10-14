#!/bin/bash
# Controlled-rho benchmarks, step 2: ONE GPU job per model, every benchmark and bin in turn, the paper's harness
# and settings for each benchmark (DENSITY_DUMP=1, dumps kept for step 3):
#   case studies : submit_cs_dvar_density.sbatch (npz backend, ANC_MODE=case_family, context 1000, 100 queries)
#   IHDP, ACIC   : submit_realcause_density_unified.sbatch settings, data in the E2 format (DATASET=CMECH_n5)
#   ComplexMech  : the main ComplexMech / E2 settings (UWYK models --eval-max-context 1000 --eval-context-seed 43)
# Checkpoints are the paper's: released Do-PFN; Do-PFN 2D 262k with temperature off; UWYK best_model;
# graph2d_step_50000; cpfn1d_j1024_headrand_step_50000; cpfn2d_j32_eta0_y01_step50000.
#   cd $KIT && git -C R-PFN pull --rebase && for m in dopfn1d dopfn2d uwyk1d graph2d cpfn1d cpfn2d; do
#     bash R-PFN/revision/experiments/rho_ctrl/2_eval.sh $m; done
# A cell (benchmark, bin, scenario) whose dump count equals its dataset count is skipped: resubmitting resumes.
set -uo pipefail
KIT="${KIT:-/scratch/furkanbd/rpfn_bench_kit}"
SELF="$KIT/R-PFN/revision/experiments/rho_ctrl/2_eval.sh"
MODEL="${1:?model: dopfn1d dopfn2d uwyk1d graph2d cpfn1d cpfn2d}"
case "$MODEL" in dopfn1d|dopfn2d|uwyk1d|graph2d|cpfn1d|cpfn2d) ;; *) echo "FATAL: unknown model $MODEL"; exit 1 ;; esac
if [ "${2:-}" != run ]; then
  cd "$KIT" && mkdir -p logs_rho_ctrl
  sbatch --account=aip-rgrosse --gres=gpu:l40s:1 --time="${RC_TIME:-3:00:00}" --cpus-per-task=4 --mem=32G \
    --job-name="rhoctrl-$MODEL" -o "logs_rho_ctrl/eval_${MODEL}_%j.out" --wrap "KIT=$KIT bash $SELF $MODEL run"
  exit $?
fi

REPO="$KIT/R-PFN"; CK="$REPO/Required_checkpoints"
D="$SCRATCH/rho_ctrl"; DU="$D/dumps/$MODEL"
export CAUSALPFN="$KIT/external/causalpfn" DOPFN_ROOT="$KIT/external/dopfn"
UWYK="$KIT/external/uwyk_reproduce"
UWYK_CKPT_DIR="$UWYK/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0"
cd "$KIT"; source venv/bin/activate
unset DOPFN_SOFTMAX_TEMP DOPFN_CKPT
export DENSITY_DUMP=1 SKIP_EM=1 COMPILE=0 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
nvidia-smi --query-gpu=name --format=csv,noheader || true
ndump() { ls "$1" 2>/dev/null | grep -c 'r[0-9][0-9]*\.npz$' || true; }

# run_model <dataset name> <outdir> <family: cs | rc | cm>
run_model() {
  local DS="$1" OUTD="$2" FAM="$3"
  export OUT="$OUTD" DATASET="$DS"
  local ANC=v3ab_only CAP=()
  [ "$FAM" = cs ] && ANC=case_family
  [ "$FAM" = cm ] && CAP=(--eval-max-context 1000 --eval-context-seed 43)
  case "$MODEL" in
    dopfn1d)
      python -u "$REPO/benchmarks/eval_scm_case_studies/eval_native_dopfn.py" --dataset "$DS" ;;
    dopfn2d)
      DOPFN_CKPT="$CK/dopfn_repro_joint2d_step262144.pt" DOPFN_SOFTMAX_TEMP=0 \
        python -u "$REPO/benchmarks/eval_scm_case_studies/eval_native_dopfn.py" --dataset "$DS" ;;
    uwyk1d)
      ANC_MODE=$ANC python -u "$REPO/realcause_eval/Table1/uwyk1d.py" --dataset "$DS" --outdir "$OUTD" \
        --uwyk-repro "$UWYK" --repo "$REPO" --causalpfn "$CAUSALPFN" \
        --ckpt "$UWYK_CKPT_DIR/best_model.pt" --config "$UWYK_CKPT_DIR/best_model_config.yaml" \
        --anc-mode "$ANC" --t-encoding target "${CAP[@]}" ;;
    graph2d)
      python -u "$REPO/realcause_eval/Table1/graph2d.py" --dataset "$DS" --outdir "$OUTD" \
        --density-dump 1 --ckpt "$CK/graph2d_step_50000.pt" --repo "$REPO" --causalpfn "$CAUSALPFN" --uwyk "$UWYK" \
        --anc-mode "$ANC" --density-primary-mode noanc "${CAP[@]}" ;;
    cpfn1d)
      python -u "$REPO/realcause_eval/Table1/cpfn1d.py" --dataset "$DS" --outdir "$OUTD" --density-dump 1 \
        --ckpt "$CK/cpfn1d_j1024_headrand_step_50000.pt" --repo "$REPO" --causalpfn "$CAUSALPFN" --std-mode pooled ;;
    cpfn2d)
      python -u "$REPO/realcause_eval/Table1/cpfn2d.py" --dataset "$DS" --outdir "$OUTD" --density-dump 1 \
        --ckpt "$CK/cpfn2d_j32_eta0_y01_step50000.pt" --repo "$REPO" --causalpfn "$CAUSALPFN" --std-mode pooled ;;
  esac || echo "WARN: $MODEL $DS $OUTD failed"
}

for b in 0 1 2; do
  # case studies: one harness call per (bin, scenario) covers every d and shift
  for CASE in Observed_Confounder Backdoor_Criterion Frontdoor_Criterion Unobserved_Confounder; do
    IN="$D/cs/bin$b/$CASE/N1000"; n_in=$(ls "$IN" 2>/dev/null | grep -c '\.npz$' || true)
    O="$DU/cs/bin$b/$CASE"; [ "$n_in" -gt 0 ] && [ "$(ndump "$O")" = "$n_in" ] && { echo "skip cs bin$b $CASE"; continue; }
    rm -rf "$O"; mkdir -p "$O"
    echo "[$(date '+%F %T')] $MODEL cs bin$b $CASE ($n_in datasets)"
    CASE_STUDY_DATA_ROOT="$D/cs/bin$b" CASE_STUDY_N=1000 SCM_N_QUERY=100 run_model "$CASE" "$O" cs
    echo "[$(date '+%F %T')] $MODEL cs bin$b $CASE: $(ndump "$O") dumps"
  done
  # IHDP, ACIC (E2 format), ComplexMech n = 5
  for SET in rc/bin$b/IHDP rc/bin$b/ACIC cm/bin$b; do
    fam=rc; [[ "$SET" == cm/* ]] && fam=cm
    export UWYK_FIG34_DATA="$D/$SET"; [ "$fam" = rc ] && export UWYK_FIG34_DATA="$D/$SET/shard0"
    n_in=$(ls "$UWYK_FIG34_DATA/complexmech/5node/path_TY/hide_0.0" 2>/dev/null | grep -c '\.npz$' || true)
    O="$DU/$SET"; [ "$n_in" -gt 0 ] && [ "$(ndump "$O")" = "$n_in" ] && { echo "skip $SET"; continue; }
    rm -rf "$O"; mkdir -p "$O"
    echo "[$(date '+%F %T')] $MODEL $SET ($n_in datasets)"
    run_model CMECH_n5 "$O" "$fam"
    echo "[$(date '+%F %T')] $MODEL $SET: $(ndump "$O") dumps"
  done
done
echo "[$(date '+%F %T')] done $MODEL"
