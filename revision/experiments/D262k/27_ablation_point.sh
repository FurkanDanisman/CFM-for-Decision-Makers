#!/bin/bash
# Stage 27: point estimates (sqrt PEHE, relative eps_ATE, mean +- SE; Delta of each 2D model vs the ablation) of the
# matched-label and matched-resolution ablation models and their 2D counterparts, on the benchmarks of
# tab:rc-ablation-cal, tab:cs-abl-cal and tab:cm-abl-r99-cal -> revision/results/d262t1/ablation_point.{csv,md}.
# ONE CPU job over existing dumps (body: 27_ablation_point.py, which calls point_raw_em.py, cs_point_total.py and
# cmech_point_total.py exactly as the paper's main point tables do), then the results are pushed.
#   cd $KIT && git -C R-PFN pull --rebase && bash R-PFN/revision/experiments/D262k/27_ablation_point.sh
# The same file is the job body: sbatch --wrap runs "bash <this file> run" (never `source` under sh).
set -uo pipefail
KIT="${KIT:-/scratch/furkanbd/rpfn_bench_kit}"
SELF="$KIT/R-PFN/revision/experiments/D262k/27_ablation_point.sh"

if [ "${1:-}" != run ]; then
  cd "$KIT" && mkdir -p logs_hist
  sbatch --account=aip-rgrosse --time="${AP_TIME:-3:00:00}" --cpus-per-task=16 --mem=64G \
    --job-name=abl-point -o logs_hist/abl_point_%j.out --wrap "KIT=$KIT bash $SELF run"
  exit $?
fi

# ---- job body ----
REPO="$KIT/R-PFN"; cd "$KIT"; source venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks:$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
O="$REPO/revision/results/d262t1"; mkdir -p "$O"
python -u "$REPO/revision/experiments/D262k/27_ablation_point.py" --repo "$REPO" --scratch "$SCRATCH" \
  --work "$SCRATCH/ablation_point" --out-dir "$O" --cs-workers 10 || echo "27_ablation_point.py exited non-zero (see log)"
[ -s "$O/ablation_point.csv" ] || { echo "no ablation_point.csv"; exit 1; }
cd "$REPO" && git add revision/results/d262t1/ablation_point.csv revision/results/d262t1/ablation_point.md \
  && git commit -q -m "d262t1: point estimates of the matched-label / matched-resolution ablation models" \
  && git pull -q --rebase && git push -q && echo pushed
