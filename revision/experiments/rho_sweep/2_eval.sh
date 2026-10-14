#!/bin/bash
# rho sweep, step 2: ONE GPU job per model. The job runs the paper's harness for that model (the same
# command and settings as revision/experiments/E2_in_prior/eval_e2.sbatch, DENSITY_DUMP=1) on the five
# rho cells in turn, 100 datasets each. Dumps are kept (small); step 3 scores them.
#   cd $KIT && git -C R-PFN pull --rebase && bash R-PFN/revision/experiments/rho_sweep/2_eval.sh <model>
#   <model> = dopfn1d | dopfn2d | uwyk1d | graph2d | cpfn1d | cpfn2d
#   (uwyk1d and graph2d run ANC_MODE=v3ab_only: No-Anc and Anc rows come from the same pass.)
# A rho cell with 100 dumps is skipped, so resubmitting a job resumes it.
# The same file is the job body: sbatch --wrap runs "bash <this file> <model> run" (never `source`).
set -uo pipefail
KIT="${KIT:-/scratch/furkanbd/rpfn_bench_kit}"
SELF="$KIT/R-PFN/revision/experiments/rho_sweep/2_eval.sh"
MODEL="${1:?model: dopfn1d dopfn2d uwyk1d graph2d cpfn1d cpfn2d}"
case "$MODEL" in dopfn1d|dopfn2d|uwyk1d|graph2d|cpfn1d|cpfn2d) ;; *) echo "FATAL: unknown model $MODEL"; exit 1 ;; esac

if [ "${2:-}" != run ]; then
  cd "$KIT" && mkdir -p logs_rho_sweep
  sbatch --account=aip-rgrosse --gres=gpu:l40s:1 --time="${RS_TIME:-3:00:00}" --cpus-per-task=4 --mem=32G \
    --job-name="rho-$MODEL" -o "logs_rho_sweep/eval_${MODEL}_%j.out" --wrap "KIT=$KIT bash $SELF $MODEL run"
  exit $?
fi

# ---- job body ----
set -e
REPO="$KIT/R-PFN"; CK="$REPO/Required_checkpoints"
DATA="${RS_DATA:-$SCRATCH/rho_sweep/data}"; DUMPS="${RS_DUMPS:-$SCRATCH/rho_sweep/dumps}"
export CAUSALPFN="$KIT/external/causalpfn" DOPFN_ROOT="$KIT/external/dopfn"
UWYK="$KIT/external/uwyk_reproduce"
UWYK_CKPT_DIR="$UWYK/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0"
case "$MODEL" in
  dopfn2d) need="$CK/dopfn_repro_joint2d_step262144.pt" ;;
  uwyk1d)  need="$UWYK_CKPT_DIR/best_model.pt" ;;
  graph2d) need="$CK/graph2d_step_50000.pt" ;;
  cpfn1d)  need="$CK/cpfn1d_j1024_headrand_step_50000.pt" ;;
  cpfn2d)  need="$CK/cpfn2d_j32_eta0_y01_step50000.pt" ;;
  *)       need="" ;;
esac
[ -z "$need" ] || [ -f "$need" ] || { echo "FATAL: no checkpoint $need"; exit 1; }
cd "$KIT"; source venv/bin/activate
unset DOPFN_SOFTMAX_TEMP DOPFN_CKPT                # set below for Do-PFN 2D only
export DATASET=CMECH_n5 DENSITY_DUMP=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
nvidia-smi --query-gpu=name --format=csv,noheader || true

for RHO in rho-0.50 rho+0.00 rho+0.50 rho+0.90 rho+1.00; do
  export UWYK_FIG34_DATA="$DATA/$RHO/shard0"
  n_in=$(ls "$UWYK_FIG34_DATA/complexmech/5node/path_TY/hide_0.0/" 2>/dev/null | grep -c '^r[0-9]*\.npz$' || true)
  [ "$n_in" = 100 ] || { echo "FATAL: $UWYK_FIG34_DATA has $n_in datasets, expected 100 (run 1_gen.sh)"; exit 1; }
  export OUT="$DUMPS/$MODEL/$RHO"
  n_out=$(ls "$OUT" 2>/dev/null | grep -c 'r[0-9][0-9][0-9]\.npz$' || true)
  [ "$n_out" = 100 ] && { echo "skip $MODEL $RHO (100 dumps)"; continue; }
  rm -rf "$OUT"; mkdir -p "$OUT"
  echo "[$(date '+%F %T')] model=$MODEL data=$UWYK_FIG34_DATA out=$OUT"
  case "$MODEL" in
    dopfn1d)                                         # published Do-PFN, its released inference config
      python -u "$REPO/benchmarks/eval_scm_case_studies/eval_native_dopfn.py" --dataset "$DATASET" ;;
    dopfn2d)                                         # our 2D at 262k steps, temperature off (D262k _t1)
      DOPFN_CKPT="$CK/dopfn_repro_joint2d_step262144.pt" DOPFN_SOFTMAX_TEMP=0 \
        python -u "$REPO/benchmarks/eval_scm_case_studies/eval_native_dopfn.py" --dataset "$DATASET" ;;
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
    cpfn1d)                                          # as the paper's ComplexMech runs (submit_cmech_pehe_1d_vs_2d.sbatch): pooled
      python -u "$REPO/realcause_eval/Table1/cpfn1d.py" --dataset "$DATASET" --outdir "$OUT" \
        --density-dump 1 --ckpt "$CK/cpfn1d_j1024_headrand_step_50000.pt" \
        --repo "$REPO" --causalpfn "$CAUSALPFN" --std-mode pooled ;;
    cpfn2d)
      python -u "$REPO/realcause_eval/Table1/cpfn2d.py" --dataset "$DATASET" --outdir "$OUT" \
        --density-dump 1 --ckpt "$CK/cpfn2d_j32_eta0_y01_step50000.pt" \
        --repo "$REPO" --causalpfn "$CAUSALPFN" --std-mode pooled ;;
  esac
  n_out=$(ls "$OUT" | grep -c 'r[0-9][0-9][0-9]\.npz$' || true)
  echo "[$(date '+%F %T')] $MODEL $RHO: $n_out dumps"
done
echo "[$(date '+%F %T')] done $MODEL"
