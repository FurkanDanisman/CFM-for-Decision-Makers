#!/bin/bash
#SBATCH --job-name=ate-sub
#SBATCH --output=logs_ate/%x_%j.out
#SBATCH --error=logs_ate/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# ATE interval by context subsampling, coverage on in-prior data (E2 "new" data, shard ATE_SHARD = 100
# datasets, 1000 context rows and 500 queries each, every model on its own prior as in eval_e2.sbatch).
# One job per model. Run b = full (whole context) and b = 0..ATE_B-1 (ATE_M of the 1000 context rows,
# without replacement, via the harnesses' own EVAL_MAX_CONTEXT / EVAL_CONTEXT_SEED), each reduced to
# per-dataset ATE estimates (reduce_ate.py) and the dumps deleted. Finished runs are skipped and the job
# stops itself before the 3 h limit, so a timed-out model is simply resubmitted. The model case blocks
# are those of E2_in_prior/eval_e2.sbatch (Do-PFN 2D with the inference temperature off, as D262k/14).
# From the kit root (no dependency jobs; resubmit until coverage.txt appears):
#   for M in dopfn_native dopfn_joint2d_262k cpfn1d cpfn2d; do ATE_MODEL=$M sbatch --account=aip-rgrosse \
#     --gres=gpu:l40s:1 R-PFN/revision/experiments/ate_subsample/run_ate_subsample.sbatch; done
set -uo pipefail
mkdir -p logs_ate
KIT="${KIT:-$PWD}"; REPO="$KIT/R-PFN"; CK="$REPO/Required_checkpoints"
HERE="$REPO/revision/experiments/ate_subsample"
export CAUSALPFN="$KIT/external/causalpfn" DOPFN_ROOT="$KIT/external/dopfn"
source "$KIT/venv/bin/activate"
export PYTHONUNBUFFERED=1 PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
export DATASET=CMECH_n5 DENSITY_DUMP=1
MODEL="${ATE_MODEL:?ATE_MODEL = dopfn_native | dopfn_joint2d_262k | cpfn1d | cpfn2d}"
B="${ATE_B:-100}"; M="${ATE_M:-500}"; SHARD="${ATE_SHARD:-0}"
case "$MODEL" in
  dopfn_native|dopfn_joint2d_262k) PRIOR=dopfn ;;
  cpfn1d) PRIOR=cpfn_1d ;;
  cpfn2d) PRIOR=cpfn_2d ;;
  *) echo "unknown model $MODEL"; exit 1 ;;
esac
export UWYK_FIG34_DATA="${E2_DATA:-$SCRATCH/e2/new/data}/$PRIOR/shard$SHARD"
[ -d "$UWYK_FIG34_DATA" ] || { echo "no data at $UWYK_FIG34_DATA"; exit 1; }
RES="${ATE_RESULTS:-$SCRATCH/ate_subsample}/$MODEL"; mkdir -p "$RES"

for b in full $(seq 0 $(( B - 1 ))); do
  F="$RES/b_$b.npz"; [ -f "$F" ] && continue
  [ "$SECONDS" -gt 9600 ] && { echo "[$(date '+%F %T')] stopping before the time limit; resubmit to continue"; exit 0; }
  if [ "$b" = full ]; then CAP=""; SEED=0; else CAP="$M"; SEED=$(( (b + 1) * 100000 )); fi
  export OUT="$RES/dumps_$b"; rm -rf "${OUT:?}"; mkdir -p "$OUT"
  echo "[$(date '+%F %T')] model=$MODEL b=$b cap=${CAP:-none} seed=$SEED"
  case "$MODEL" in
    dopfn_native)
      EVAL_MAX_CONTEXT="$CAP" EVAL_CONTEXT_SEED="$SEED" \
        python -u "$REPO/benchmarks/eval_scm_case_studies/eval_native_dopfn.py" --dataset "$DATASET" ;;
    dopfn_joint2d_262k)
      DOPFN_SOFTMAX_TEMP=0 DOPFN_CKPT="$CK/dopfn_repro_joint2d_step262144.pt" \
      EVAL_MAX_CONTEXT="$CAP" EVAL_CONTEXT_SEED="$SEED" \
        python -u "$REPO/benchmarks/eval_scm_case_studies/eval_native_dopfn.py" --dataset "$DATASET" ;;
    cpfn1d)
      CKPT="$CK/cpfn1d_j1024_headrand_step_50000.pt" EVAL_MAX_CONTEXT="$CAP" EVAL_CONTEXT_SEED="$SEED" \
      STD_MODE=per_arm COMPILE=0 SKIP_EM=1 \
        python -u "$REPO/case_study/density_eval/eval_cpfn1d_perarm.py" --dataset "$DATASET" ;;
    cpfn2d)
      python -u "$REPO/realcause_eval/Table1/cpfn2d.py" --dataset "$DATASET" --outdir "$OUT" \
        --density-dump 1 --ckpt "$CK/cpfn2d_j32_eta0_y01_step50000.pt" \
        --repo "$REPO" --causalpfn "$CAUSALPFN" --std-mode pooled \
        --eval-max-context "${CAP:-0}" --eval-context-seed "$SEED" ;;
  esac || { echo "harness failed at b=$b"; exit 1; }
  python "$HERE/reduce_ate.py" "$OUT" "$F" || { echo "reduce failed at b=$b"; exit 1; }
  rm -rf "${OUT:?}"
done
python "$HERE/ate_coverage.py" "$RES" --m "$M" --n 1000 | tee "$RES/coverage.txt"
