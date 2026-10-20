#!/bin/bash
# Bodik's IHDP-rho and Twins datasets (CLeaR 2026), scored with the paper's models. Inference only, no training.
#   step 1 (CPU, once):        bash run.sh gen
#   step 2 (GPU, ONE job per model, after step 1 has finished; parts ihdp_a, twins, ihdp_b in turn):
#        for m in dopfn1d dopfn2d uwyk1d graph2d cpfn1d cpfn2d; do bash run.sh eval $m; done
#      (a single part: bash run.sh eval <model> <ihdp_a | twins | ihdp_b>)
#   step 3 (CPU, after step 2):  bash run.sh score        -> pushes revision/results/d262t1/bodik.{csv,md}
# Every step resubmits safely: finished datasets / cells are skipped. Run from anywhere; KIT is the bench kit.
# The model calls and checkpoints are those of revision/experiments/rho_ctrl/2_eval.sh (the paper's models:
# released Do-PFN; Do-PFN 2D 262k, temperature off; UWYK best_model; graph2d_step_50000;
# cpfn1d_j1024_headrand_step_50000; cpfn2d_j32_eta0_y01_step50000), with the RealCause settings
# (UWYK ANC_MODE=v3ab_only, no context cap: the datasets are written with their final context).
set -uo pipefail
KIT="${KIT:-/scratch/furkanbd/rpfn_bench_kit}"
SELF="$KIT/R-PFN/revision/experiments/bodik/run.sh"
STEP="${1:?step: gen | eval <model> <part> | score}"
D="$SCRATCH/bodik"
if [ "${RUN:-}" != 1 ]; then
  cd "$KIT" && mkdir -p logs_bodik
  case "$STEP" in
    gen)   sbatch --account=aip-rgrosse --time=1:00:00 --cpus-per-task=2 --mem=16G --job-name=bodik-gen \
             -o logs_bodik/gen_%j.out --wrap "RUN=1 KIT=$KIT bash $SELF gen" ;;
    eval)  M="${2:?model}" P="${3:-all}"
           sbatch --account=aip-rgrosse --gres=gpu:l40s:1 --time=3:00:00 --cpus-per-task=4 --mem=48G \
             --job-name="bodik-$M-$P" -o "logs_bodik/eval_${M}_${P}_%j.out" --wrap "RUN=1 KIT=$KIT bash $SELF eval $M $P" ;;
    score) sbatch --account=aip-rgrosse --time=2:00:00 --cpus-per-task=4 --mem=64G --job-name=bodik-score \
             -o logs_bodik/score_%j.out --wrap "RUN=1 KIT=$KIT bash $SELF score" ;;
    *) echo "FATAL: unknown step $STEP"; exit 1 ;;
  esac
  exit $?
fi

REPO="$KIT/R-PFN"; CK="$REPO/Required_checkpoints"; HERE="$REPO/revision/experiments/bodik"
cd "$KIT"; source venv/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" PYTHONUNBUFFERED=1

if [ "$STEP" = gen ]; then
  python -u "$HERE/make_worlds.py" "$D" && echo "[$(date '+%F %T')] gen done"
  exit $?
fi

if [ "$STEP" = score ]; then
  export OMP_NUM_THREADS=1
  export PYTHONPATH="$REPO/benchmarks:$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
  O="$REPO/revision/results/d262t1"; mkdir -p "$O"
  python -u "$HERE/score_bodik.py" "$D" "$O/bodik" || { echo "score_bodik.py failed; nothing pushed"; exit 1; }
  cd "$REPO" && git add revision/results/d262t1/bodik.csv revision/results/d262t1/bodik.md \
    && git commit -q -m "d262t1: Bodik IHDP-rho and Twins results" && git pull -q --rebase && git push -q && echo pushed
  exit $?
fi

# ---- eval
MODEL="${2:?model}"; PARTS="${3:-all}"; [ "$PARTS" = all ] && PARTS="ihdp_a twins ihdp_b"
case "$MODEL" in dopfn1d|dopfn2d|uwyk1d|graph2d|cpfn1d|cpfn2d) ;; *) echo "FATAL: unknown model $MODEL"; exit 1 ;; esac
export CAUSALPFN="$KIT/external/causalpfn" DOPFN_ROOT="$KIT/external/dopfn"
UWYK="$KIT/external/uwyk_reproduce"
UWYK_CKPT_DIR="$UWYK/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0"
unset DOPFN_SOFTMAX_TEMP DOPFN_CKPT
export DENSITY_DUMP=1 SKIP_EM=1 COMPILE=0
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
nvidia-smi --query-gpu=name --format=csv,noheader || true
ndump() { ls "$1" 2>/dev/null | grep -c 'r[0-9][0-9]*\.npz$' || true; }

run_model() {   # run_model <outdir>   (dataset CMECH_n5 read from $UWYK_FIG34_DATA)
  local OUTD="$1"; export OUT="$OUTD" DATASET=CMECH_n5
  case "$MODEL" in
    dopfn1d)
      python -u "$REPO/benchmarks/eval_scm_case_studies/eval_native_dopfn.py" --dataset CMECH_n5 ;;
    dopfn2d)
      DOPFN_CKPT="$CK/dopfn_repro_joint2d_step262144.pt" DOPFN_SOFTMAX_TEMP=0 \
        python -u "$REPO/benchmarks/eval_scm_case_studies/eval_native_dopfn.py" --dataset CMECH_n5 ;;
    uwyk1d)
      ANC_MODE=v3ab_only python -u "$REPO/realcause_eval/Table1/uwyk1d.py" --dataset CMECH_n5 --outdir "$OUTD" \
        --uwyk-repro "$UWYK" --repo "$REPO" --causalpfn "$CAUSALPFN" \
        --ckpt "$UWYK_CKPT_DIR/best_model.pt" --config "$UWYK_CKPT_DIR/best_model_config.yaml" \
        --anc-mode v3ab_only --t-encoding target ;;
    graph2d)
      python -u "$REPO/realcause_eval/Table1/graph2d.py" --dataset CMECH_n5 --outdir "$OUTD" \
        --density-dump 1 --ckpt "$CK/graph2d_step_50000.pt" --repo "$REPO" --causalpfn "$CAUSALPFN" --uwyk "$UWYK" \
        --anc-mode v3ab_only --density-primary-mode noanc ;;
    cpfn1d)
      python -u "$REPO/realcause_eval/Table1/cpfn1d.py" --dataset CMECH_n5 --outdir "$OUTD" --density-dump 1 \
        --ckpt "$CK/cpfn1d_j1024_headrand_step_50000.pt" --repo "$REPO" --causalpfn "$CAUSALPFN" --std-mode pooled ;;
    cpfn2d)
      python -u "$REPO/realcause_eval/Table1/cpfn2d.py" --dataset CMECH_n5 --outdir "$OUTD" --density-dump 1 \
        --ckpt "$CK/cpfn2d_j32_eta0_y01_step50000.pt" --repo "$REPO" --causalpfn "$CAUSALPFN" --std-mode pooled ;;
  esac || echo "WARN: $MODEL $OUTD failed"
}

# cells of this part: <data dir relative to $D> <dump dir relative to $D/dumps/$MODEL>
cells() {
  case "$PART" in
    ihdp_a|ihdp_b)
      for d in 10 1; do for r in 1 0.75 0.5 0.25 0; do echo "$PART/d$d/rho$r $PART/d$d/rho$r"; done; done ;;
    twins)
      for d in 71 10 1; do
        if [ "$d" = 71 ] && { [ "$MODEL" = uwyk1d ] || [ "$MODEL" = graph2d ]; }; then
          echo "twins/d71_u50 twins/d71"          # UWYK takes at most 50 covariates: its first 50
        else
          echo "twins/d$d twins/d$d"
        fi
      done ;;
    *) echo "FATAL: unknown part $PART" >&2; exit 1 ;;
  esac
}

for PART in $PARTS; do
while read -r SRC DST; do
  IN="$D/$SRC/shard0/complexmech/5node/path_TY/hide_0.0"
  n_in=$(ls "$IN" 2>/dev/null | grep -c '\.npz$' || true)
  [ "$n_in" -gt 0 ] || { echo "MISSING $IN (run step 1 first)"; continue; }
  O="$D/dumps/$MODEL/$DST"
  [ "$(ndump "$O")" = "$n_in" ] && { echo "skip $DST"; continue; }
  rm -rf "$O"; mkdir -p "$O"
  export UWYK_FIG34_DATA="$D/$SRC/shard0"
  echo "[$(date '+%F %T')] $MODEL $DST ($n_in datasets)"
  run_model "$O"
  echo "[$(date '+%F %T')] $MODEL $DST: $(ndump "$O") dumps"
done < <(cells)
echo "[$(date '+%F %T')] done $MODEL $PART"
done
