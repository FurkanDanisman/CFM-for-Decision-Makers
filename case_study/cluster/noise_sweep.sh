#!/bin/bash
# σ_ε resolution sweep at N=500: regenerate the case studies at a range of
# noise scales (σ_ε = k · 0.3·Beta(1,5)), evaluate all models on each, so we can
# see the coarse-bin 2D methods catch up to 1D as the CID widens past bin width.
#
# noise_scale k grid (default "1 3 10 30 100") → mean σ_ε ≈ 0.06, 0.18, 0.6, 1.8, 6.
# k=1 is exactly the current data. True CATE is unchanged by k (μ are noiseless).
#
# Usage:
#   DEPLOY_ROOT=$SCRATCH/rpfn_bench_kit \
#     bash case_study/cluster/noise_sweep.sh <ROOT>
#   <ROOT> : parent for both data and results (defaults under $DEPLOY_ROOT).
#
# Env: KS (default "1 3 10 30 100"), CATE_SHIFT (default 2), MODELS (default all).
set -euo pipefail

DEPLOY_ROOT="${DEPLOY_ROOT:?DEPLOY_ROOT required}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
ROOT="${1:-$DEPLOY_ROOT/results_case_study/noise_sweep}"
KS="${KS:-1 3 10 30 100}"
CATE_SHIFT="${CATE_SHIFT:-2}"
N=500

cd "$DEPLOY_ROOT"
echo "[noise_sweep] KS=[$KS]  N=$N  shift=$CATE_SHIFT  root=$ROOT"

for K in $KS; do
  DATA="$ROOT/data/k${K}"
  SWEEP="$ROOT/results/k${K}"
  echo "── k=$K ──  data=$DATA  sweep=$SWEEP"

  GEN_JID=$(DATA_ROOT="$DATA" CATE_SHIFT="$CATE_SHIFT" NOISE_SCALE="$K" CONTEXTS="$N" \
      sbatch --parsable "$REPO/case_study/cluster/01_generate.sbatch")
  echo "   generate job $GEN_JID"

  DEP="$GEN_JID" CONTEXTS="$N" MODELS="${MODELS:-}" \
      bash "$REPO/case_study/cluster/02_submit_eval.sh" "$DATA" "$SWEEP"
done

echo
echo "[noise_sweep] when done, report with:"
echo "  python $REPO/case_study/cluster/noise_report.py --root $ROOT/results --ks $KS"
