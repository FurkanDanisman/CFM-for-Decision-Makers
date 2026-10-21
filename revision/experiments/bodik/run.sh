#!/bin/bash
# Bodik's IHDP-rho and Twins datasets (CLeaR 2026), scored with the paper's models. Inference only, no training.
#   step 1 (CPU, once):        bash run.sh gen
#   step 2 (GPU, ONE job per model, after step 1 has finished; parts ihdp_a, twins, ihdp_b in turn):
#        for m in dopfn1d dopfn2d uwyk1d graph2d cpfn1d cpfn2d; do bash run.sh eval $m; done
#      the appendix models (ablations, published CausalPFN, UWYK-B), the same way:
#        for m in dopfn1d_mr dopfn1d_ml uwyk1d_mr uwyk1d_ml cpfn1d_mr cpfn1d_ml cpfn_v0 uwyk_bin; do bash run.sh eval $m; done
#      (a single part: bash run.sh eval <model> <ihdp_a | twins | ihdp_b>)
#   step 3 (CPU, after step 2, four jobs in parallel):
#        for c in none comonotonic gauss:-0.5 gauss:0.5; do bash run.sh score $c; done
#   step 4 (after all four):     bash run.sh score merge  -> pushes revision/results/d262t1/bodik{,_comonotonic,
#        _gauss_-0.5,_gauss_0.5}.{csv,md} (the last three: the 1D models with their marginals under that coupling)
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
    twins) sbatch --account=aip-rgrosse --time=1:00:00 --cpus-per-task=16 --mem=64G --job-name=bodik-twins \
             -o logs_bodik/twins_%j.out --wrap "RUN=1 KIT=$KIT bash $SELF twins" ;;
    score) C="${2:-none}"                          # none | comonotonic | gauss:-0.5 | gauss:0.5 | merge
           if [ "$C" = merge ]; then
             sbatch --account=aip-rgrosse --time=0:30:00 --cpus-per-task=2 --mem=16G --job-name=bodik-merge \
               -o logs_bodik/merge_%j.out --wrap "RUN=1 KIT=$KIT bash $SELF score merge"
           else
             sbatch --account=aip-rgrosse --time=3:00:00 --cpus-per-task=16 --mem=96G --job-name="bodik-score-$C" \
               -o "logs_bodik/score_$(echo "$C" | tr ':' '_')_%j.out" --wrap "RUN=1 KIT=$KIT bash $SELF score $C"
           fi ;;
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

if [ "$STEP" = twins ]; then
  # Twins only, every (coupling, model) part in parallel in ONE job, then merged and pushed as bodik_twins*.
  export OMP_NUM_THREADS=1 TWINS_ONLY=1
  export PYTHONPATH="$REPO/benchmarks:$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
  O="$REPO/revision/results/d262t1"; P="$D/score_parts_twins"; rm -rf "$P"; mkdir -p "$O" "$P" "$KIT/logs_bodik/twins"
  DIRS1="dopfn1d uwyk1d cpfn1d dopfn1d_mr dopfn1d_ml uwyk1d_mr uwyk1d_ml cpfn1d_mr cpfn1d_ml cpfn_v0 uwyk_bin"
  DIRS2="dopfn2d graph2d cpfn2d"
  tpart() {
    local c="$1" m="$2" n; n="$(echo "$c" | tr ':' '_')__$m"
    if [ "$c" = none ]; then ONLY_DIR="$m" python -u "$HERE/score_bodik.py" "$D" "$P/$n" > "$KIT/logs_bodik/twins/$n.log" 2>&1
    else COUPLING="$c" ONLY_DIR="$m" python -u "$HERE/score_bodik.py" "$D" "$P/$n" > "$KIT/logs_bodik/twins/$n.log" 2>&1
    fi || echo "FAILED $n (logs_bodik/twins/$n.log)"
  }
  export -f tpart; export D P KIT HERE
  { echo "none observed"; for m in $DIRS1 $DIRS2; do echo "none $m"; done
    for c in comonotonic gauss:-0.5 gauss:0.5; do for m in $DIRS1; do echo "$c $m"; done; done; } \
    | xargs -P 16 -L 1 bash -c 'tpart "$0" "$1"'
  python -u "$HERE/score_bodik.py" merge "$O/bodik_twins" "$P"/none__*.csv
  for c in comonotonic gauss_-0.5 gauss_0.5; do python -u "$HERE/score_bodik.py" merge "$O/bodik_twins_$c" "$P"/${c}__*.csv; done
  cd "$REPO" && git add revision/results/d262t1/bodik_twins*.csv revision/results/d262t1/bodik_twins*.md \
    && git commit -q -m "d262t1: Bodik Twins, pair score split into factual and counterfactual-given-factual" \
    && git pull -q --rebase && git push -q && echo pushed
  exit $?
fi

if [ "$STEP" = score ]; then
  # ONE job per coupling ("none" = the main results: all models, 2D with their learned joint, 1D independent;
  # the others rescore the 1D models with their marginals under that coupling). Inside a job every model
  # directory is its own process (at most 15, all at once on 16 CPUs). Finished parts are skipped on resubmit.
  # When all four jobs are done: bash run.sh score merge  (merges, pushes revision/results/d262t1/bodik*).
  C="${2:-none}"
  export OMP_NUM_THREADS=1
  export PYTHONPATH="$REPO/benchmarks:$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
  O="$REPO/revision/results/d262t1"; P="$D/score_parts"; mkdir -p "$O" "$P" "$KIT/logs_bodik/score"
  DIRS1="dopfn1d uwyk1d cpfn1d dopfn1d_mr dopfn1d_ml uwyk1d_mr uwyk1d_ml cpfn1d_mr cpfn1d_ml cpfn_v0 uwyk_bin"
  DIRS2="dopfn2d graph2d cpfn2d"
  if [ "$C" = merge ]; then
    want=$(( 1 + $(echo $DIRS1 $DIRS2 | wc -w) + 3 * $(echo $DIRS1 | wc -w) ))
    have=$(ls "$P"/*.csv 2>/dev/null | wc -l)
    [ "$have" = "$want" ] || { echo "FATAL: $have of $want parts finished; resubmit the missing coupling jobs first"; exit 1; }
    python -u "$HERE/score_bodik.py" merge "$O/bodik" "$P"/none__*.csv
    for c in comonotonic gauss_-0.5 gauss_0.5; do python -u "$HERE/score_bodik.py" merge "$O/bodik_$c" "$P"/${c}__*.csv; done
    cd "$REPO" && git add revision/results/d262t1/bodik*.csv revision/results/d262t1/bodik*.md \
      && git commit -q -m "d262t1: Bodik IHDP-rho and Twins results (all models, 2x2 Twins tables)" && git pull -q --rebase && git push -q && echo pushed
    exit $?
  fi
  part() {   # part <coupling or "none"> <model dir or observed>
    local c="$1" m="$2" n; n="$(echo "$c" | tr ':' '_')__$m"
    [ -s "$P/$n.csv" ] && return 0
    if [ "$c" = none ]; then
      ONLY_DIR="$m" python -u "$HERE/score_bodik.py" "$D" "$P/$n" > "$KIT/logs_bodik/score/$n.log" 2>&1
    else
      COUPLING="$c" ONLY_DIR="$m" python -u "$HERE/score_bodik.py" "$D" "$P/$n" > "$KIT/logs_bodik/score/$n.log" 2>&1
    fi || echo "FAILED $n (logs_bodik/score/$n.log)"
  }
  export -f part; export D P KIT HERE
  if [ "$C" = none ]; then
    { echo "none observed"; for m in $DIRS1 $DIRS2; do echo "none $m"; done; }
  else
    for m in $DIRS1; do echo "$C $m"; done
  fi | xargs -P 16 -L 1 bash -c 'part "$0" "$1"'
  echo "[$(date '+%F %T')] coupling $C done: $(ls "$P"/$(echo "$C" | tr ':' '_')__*.csv 2>/dev/null | wc -l) parts"
  exit 0
fi

# ---- eval
MODEL="${2:?model}"; PARTS="${3:-all}"; [ "$PARTS" = all ] && PARTS="ihdp_a twins ihdp_b"
case "$MODEL" in dopfn1d|dopfn2d|uwyk1d|graph2d|cpfn1d|cpfn2d|dopfn1d_mr|dopfn1d_ml|uwyk1d_mr|uwyk1d_ml|cpfn1d_mr|cpfn1d_ml|cpfn_v0|uwyk_bin) ;;
  *) echo "FATAL: unknown model $MODEL"; exit 1 ;; esac
AUR="$KIT/from_aurora"; D262="$KIT/checkpoints_dopfn_repro_262k"
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
    dopfn1d_mr|dopfn1d_ml)                         # appendix ablations: 262k Do-PFN 1D, temperature off
      local V=dopfn_1d; [ "$MODEL" = dopfn1d_ml ] && V=dopfn_1d_botharms
      DOPFN_CKPT="$D262/$V/step_262144_final.pt" DOPFN_SOFTMAX_TEMP=0 \
        python -u "$REPO/benchmarks/eval_scm_case_studies/eval_native_dopfn.py" --dataset CMECH_n5 ;;
    uwyk1d_mr|uwyk1d_ml|uwyk_bin)                  # UWYK matched resolution (D) / matched label (C) / binary (UWYK-B)
      local C K E A
      case "$MODEL" in
        uwyk1d_mr) K="$AUR/uwyk_D_j32_nobin_step50000.pt"; C="$AUR/uwyk_D_j32_config.yaml"; E=target; A=v3a_only ;;
        uwyk1d_ml) K="$AUR/uwyk_C_botharms_step50000.pt"; C="$AUR/best_model_config.yaml"; E=binary; A=v3a_only ;;
        uwyk_bin)  K="$CK/uwyk_bin_step50000.pt"; C="$CK/uwyk_USED_IN_RESULTS_best_model_config.yaml"; E=binary; A=v3ab_only ;;
      esac
      ANC_MODE=$A python -u "$REPO/realcause_eval/Table1/uwyk1d.py" --dataset CMECH_n5 --outdir "$OUTD" \
        --uwyk-repro "$UWYK" --repo "$REPO" --causalpfn "$CAUSALPFN" --ckpt "$K" --config "$C" \
        --anc-mode "$A" --t-encoding "$E" ;;
    cpfn1d_mr|cpfn1d_ml|cpfn_v0)                   # CausalPFN-C matched resolution / matched label / published two-stage
      local K
      case "$MODEL" in
        cpfn1d_mr) K="$CK/cpfn1d_j32_step50000.pt" ;;
        cpfn1d_ml) K="$CK/cpfn1d_botharms_step50000.pt" ;;
        cpfn_v0)   K="$CK/cpfn_v0_original.pt" ;;
      esac
      python -u "$REPO/realcause_eval/Table1/cpfn1d.py" --dataset CMECH_n5 --outdir "$OUTD" --density-dump 1 \
        --ckpt "$K" --repo "$REPO" --causalpfn "$CAUSALPFN" --std-mode pooled ;;
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
    twins)                                          # his d = 10 and d = 1 (d = 71 is not reported)
      for d in 10 1; do echo "twins/d$d twins/d$d"; done ;;
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
