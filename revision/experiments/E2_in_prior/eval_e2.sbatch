#!/bin/bash
#SBATCH --job-name=e2-eval
#SBATCH --output=logs_e2/eval_%a.out
#SBATCH --error=logs_e2/eval_%a.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-6
# Do-PFN-bb (task 7) has 1000 worlds: run it on its own with E2_SHARDS=10 --array=7.
# Do-PFN 2D at the published 1D budget (task 8, 262,144 steps), added beside the 150k one:
#   E2_DATA=$SCRATCH/e2/new/data E2_RESULTS=$SCRATCH/e2/new/results E2_SHARDS=10 sbatch --array=8 ...
# Fixed-SCM experiment (tasks 9-12, Do-PFN 1D 10k: natural, setup1, setup2, setup3), same-SCM data only:
#   E2_DATA=$SCRATCH/e2/same/data E2_RESULTS=$SCRATCH/e2/same/results_fscm E2_SHARDS=10 sbatch --array=9-12 ...
# E2: the paper's per-model harnesses, unchanged, on the E2 datasets (DENSITY_DUMP=1).
# One task per model; it runs the shards (100 datasets each) in turn. After each shard the dumps
# are reduced to per-query 95% intervals (score_shard.py) and deleted. A shard with a result file
# is skipped, so resubmitting a task resumes it.
# Submit AFTER generation has finished (no dependency jobs). 7 tasks, one GPU each:
#   E2_DATA=$SCRATCH/e2/new/data  E2_RESULTS=$SCRATCH/e2/new/results  E2_SHARDS=100 sbatch R-PFN/revision/experiments/E2_in_prior/eval_e2.sbatch
#   E2_DATA=$SCRATCH/e2/same/data E2_RESULTS=$SCRATCH/e2/same/results E2_SHARDS=10  sbatch R-PFN/revision/experiments/E2_in_prior/eval_e2.sbatch
set -euo pipefail
mkdir -p logs_e2
DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
REPO="$DEPLOY_ROOT/R-PFN"
CK="$REPO/Required_checkpoints"
export CAUSALPFN="$DEPLOY_ROOT/external/causalpfn" DOPFN_ROOT="$DEPLOY_ROOT/external/dopfn"
UWYK="$DEPLOY_ROOT/external/uwyk_reproduce"
UWYK_CKPT_DIR="$UWYK/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0"

MODELS=(dopfn_native dopfn_joint2d uwyk1d graph2d cpfn1d cpfn2d uwyk1d_cont dopfn_bb dopfn_joint2d_262k
        fscm_natural fscm_setup1 fscm_setup2 fscm_setup3)
PRIORS=(dopfn        dopfn         uwyk_1d uwyk_2d cpfn_1d cpfn_2d uwyk_1dc    dopfn_bb dopfn
        dopfn        dopfn       dopfn       dopfn)
I=$SLURM_ARRAY_TASK_ID
MODEL=${MODELS[$I]}
export DATASET=CMECH_n5 DENSITY_DUMP=1        # all 100 queries of every world
source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1 PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

RES="${E2_RESULTS:?}/$MODEL"; mkdir -p "$RES"
# E2_SHARD_LIST (optional, space-separated) runs only those shards instead of 0..E2_SHARDS-1.
for SHARD in ${E2_SHARD_LIST:-$(seq 0 $(( ${E2_SHARDS:?} - 1 )))}; do
[ -f "$RES/shard$SHARD.npz" ] && { echo "skip shard $SHARD (done)"; continue; }
export UWYK_FIG34_DATA="${E2_DATA:?}/${PRIORS[$I]}/shard$SHARD"
export OUT="$RES/dumps_shard$SHARD"
rm -rf "$OUT"; mkdir -p "$OUT"
echo "[$(date '+%F %T')] model=$MODEL data=$UWYK_FIG34_DATA out=$OUT"

case "$MODEL" in
  dopfn_native)
    python -u "$REPO/benchmarks/eval_scm_case_studies/eval_native_dopfn.py" --dataset "$DATASET" ;;
  dopfn_bb)                                   # writes density_r###.npz next to --out
    python -u "$REPO/benchmarks/l2_ihdp/eval_dopfn_bb_raw.py" --dataset "$DATASET" --out "$OUT/summary.npz" \
      --repo "$REPO" --dopfn "$DOPFN_ROOT" --causalpfn "$CAUSALPFN" \
      --checkpoint "$CK/dopfn_bb_j10_step_150000.pt" --n-realizations 100 ;;
  dopfn_joint2d)
    DOPFN_CKPT="$CK/dopfn_repro_joint2d_step150000.pt" \
      python -u "$REPO/benchmarks/eval_scm_case_studies/eval_native_dopfn.py" --dataset "$DATASET" ;;
  dopfn_joint2d_262k)
    DOPFN_CKPT="$CK/dopfn_repro_joint2d_step262144.pt" \
      python -u "$REPO/benchmarks/eval_scm_case_studies/eval_native_dopfn.py" --dataset "$DATASET" ;;
  fscm_*)                                     # fixed-SCM experiment (revision/experiments/fixed_scm), temperature off
    DOPFN_SOFTMAX_TEMP=0 DOPFN_CKPT="$DEPLOY_ROOT/${FSCM_DIR:-checkpoints_fixed_scm}/${MODEL#fscm_}/dopfn_1d/${FSCM_CKPT:-step_10000_final.pt}" \
      python -u "$REPO/benchmarks/eval_scm_case_studies/eval_native_dopfn.py" --dataset "$DATASET" ;;
  uwyk1d)
    python -u "$REPO/realcause_eval/Table1/uwyk1d.py" --dataset "$DATASET" --outdir "$OUT" \
      --uwyk-repro "$UWYK" --repo "$REPO" --causalpfn "$CAUSALPFN" \
      --ckpt "$UWYK_CKPT_DIR/best_model.pt" --config "$UWYK_CKPT_DIR/best_model_config.yaml" \
      --anc-mode v3ab_only --t-encoding target --eval-max-context 1000 --eval-context-seed 43 ;;
  uwyk1d_cont)                                # UWYK 1D on its own continuous-T prior
    python -u "$REPO/realcause_eval/Table1/uwyk1d.py" --dataset "$DATASET" --outdir "$OUT" \
      --uwyk-repro "$UWYK" --repo "$REPO" --causalpfn "$CAUSALPFN" \
      --ckpt "$UWYK_CKPT_DIR/best_model.pt" --config "$UWYK_CKPT_DIR/best_model_config.yaml" \
      --anc-mode v3ab_only --t-encoding continuous --eval-max-context 1000 --eval-context-seed 43 ;;
  graph2d)
    python -u "$REPO/realcause_eval/Table1/graph2d.py" --dataset "$DATASET" --outdir "$OUT" \
      --density-dump 1 --ckpt "$CK/graph2d_step_50000.pt" \
      --repo "$REPO" --causalpfn "$CAUSALPFN" --uwyk "$UWYK" \
      --anc-mode v3ab_only --density-primary-mode noanc --eval-max-context 1000 --eval-context-seed 43 ;;
  cpfn1d)                                     # per-arm y scaling (its training convention)
    CKPT="$CK/cpfn1d_j1024_headrand_step_50000.pt" EVAL_MAX_CONTEXT= EVAL_CONTEXT_SEED=1 \
    STD_MODE=per_arm COMPILE=0 SKIP_EM=1 \
      python -u "$REPO/case_study/density_eval/eval_cpfn1d_perarm.py" --dataset "$DATASET" ;;
  cpfn2d)
    python -u "$REPO/realcause_eval/Table1/cpfn2d.py" --dataset "$DATASET" --outdir "$OUT" \
      --density-dump 1 --ckpt "$CK/cpfn2d_j32_eta0_y01_step50000.pt" \
      --repo "$REPO" --causalpfn "$CAUSALPFN" --std-mode pooled ;;
esac
python "$REPO/revision/experiments/E2_in_prior/score_shard.py" "$OUT" "$RES/shard$SHARD.npz"
# E2_TRUTH_Y (multi.sh eval_y): also the outcome-level coverage, while the dumps still exist
[ -n "${E2_TRUTH_Y:-}" ] && python "$REPO/revision/experiments/E2_in_prior/score_outcomes.py" "$OUT" "$E2_TRUTH_Y" "$SHARD" "$RES/outcomes_shard$SHARD.npz"
# E2_OUTCOMES_ALL=1 (newworld_y.sh): every model's outcome-level readouts (+ joint region and rho = 0 ITE for 2D)
[ -n "${E2_OUTCOMES_ALL:-}" ] && python "$REPO/revision/experiments/E2_in_prior/score_outcomes_all.py" "$OUT" "$RES/outall_shard$SHARD.npz" "${E2_DATA}/${PRIORS[$I]}/truths_y" "$SHARD"
# E2_MOMENTS=1 (fscm_moments.sh): also the per-query mean and sd of each predicted outcome
[ -n "${E2_MOMENTS:-}" ] && python "$REPO/revision/experiments/E2_in_prior/score_moments.py" "$OUT" "$RES/moments_shard$SHARD.npz"
rm -rf "$OUT"
done
echo "[$(date '+%F %T')] done"
