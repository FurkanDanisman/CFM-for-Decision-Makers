#!/bin/bash
# Tiny end-to-end run of each FOR_LUKE task's code path. Seconds, not hours —
# N and MALC_B are cut to the bone. This checks that imports, external repos,
# checkpoints and data all resolve; it says nothing about the numbers.
#
# Usage:
#   DEPLOY_ROOT=$SCRATCH/rpfn_bench_kit bash <repo>/benchmarks/cluster/smoke_test.sh

set -uo pipefail

DEPLOY_ROOT="${DEPLOY_ROOT:-$SCRATCH/rpfn_bench_kit}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
[ ! -d "$REPO/benchmarks" ] && REPO="$DEPLOY_ROOT"
OUT="${OUT:-$DEPLOY_ROOT/smoke}"
UWYK_CKPT_DIR="$DEPLOY_ROOT/external/uwyk/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0"

mkdir -p "$OUT"
source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1

fail=0
run() { echo; echo "── $1 ──"; shift; if "$@"; then echo "  PASS"; else echo "  FAIL"; fail=1; fi; }

# Task 1 — context sweep (both backbones)
run "task 1a: context_sweep backbone=ipfn" \
    python "$REPO/benchmarks/context_sweep/run_one.py" \
        --source prior --seed 0 --n-context 50 --n-test 3 --outdir "$OUT/sweep" \
        --repo "$REPO" --uwyk-src "$DEPLOY_ROOT/external/uwyk/src" \
        --causalpfn "$DEPLOY_ROOT/external/causalpfn" \
        --checkpoint "$REPO/checkpoints/step_50000_final.pt" \
        --uwyk-ckpt-dir "$UWYK_CKPT_DIR" --dopfn "$DEPLOY_ROOT/external/dopfn" \
        --workers 4 --backbone ipfn --malc-B 5 --malc-max-K 2 --n-eval 3

run "task 1b: context_sweep backbone=dopfn_bb" \
    python "$REPO/benchmarks/context_sweep/run_one.py" \
        --source prior --seed 1 --n-context 50 --n-test 3 --outdir "$OUT/sweep_bb" \
        --repo "$REPO" --uwyk-src "$DEPLOY_ROOT/external/uwyk/src" \
        --causalpfn "$DEPLOY_ROOT/external/causalpfn" \
        --checkpoint "$DEPLOY_ROOT/checkpoints_dopfn_backbone_j10/step_200000.pt" \
        --uwyk-ckpt-dir "$UWYK_CKPT_DIR" --dopfn "$DEPLOY_ROOT/external/dopfn" \
        --workers 4 --backbone dopfn_bb --malc-B 5 --malc-max-K 2 --n-eval 3

# Tasks 2 + 3 — IHDP density L2 (uwyk_anc, ours_dopfn_bb)
run "tasks 2+3: l2_ihdp uwyk_anc + ours_dopfn_bb" \
    python "$REPO/benchmarks/l2_ihdp/eval_realization.py" \
        --realization 0 --out "$OUT/ihdp" --repo "$REPO" \
        --dopfn "$DEPLOY_ROOT/external/dopfn" --causalpfn "$DEPLOY_ROOT/external/causalpfn" \
        --uwyk-src "$DEPLOY_ROOT/external/uwyk/src" --uwyk-ckpt-dir "$UWYK_CKPT_DIR" \
        --checkpoint50 "$REPO/checkpoints/step_50000_final.pt" \
        --checkpoint10 "$REPO/checkpoints_dopfn/step_50000_final.pt" \
        --checkpoint-dopfn-bb "$DEPLOY_ROOT/checkpoints_dopfn_backbone_j10/step_200000.pt" \
        --n-context 100 --n-eval 3 --malc-B 5 --malc-max-K 2 --uwyk-n-samples 64 \
        --methods uwyk_anc,ours_dopfn_bb

# Task 4 — linear-Gaussian synthetic
run "task 4: l2_syn dopfn + ours_dopfn_bb" \
    python "$REPO/benchmarks/l2_syn/eval_realization.py" \
        --seed 0 --out "$OUT/syn" --repo "$REPO" \
        --dopfn "$DEPLOY_ROOT/external/dopfn" --causalpfn "$DEPLOY_ROOT/external/causalpfn" \
        --checkpoint50 "$REPO/checkpoints/step_50000_final.pt" \
        --checkpoint-dopfn-bb "$DEPLOY_ROOT/checkpoints_dopfn_backbone_j10/step_200000.pt" \
        --n-train 100 --n-test 3 --syn-d 5 --malc-B 5 --malc-max-K 2 \
        --methods dopfn,ours_dopfn_bb

echo
if [ "$fail" = 0 ]; then echo "ALL SMOKE TESTS PASSED"; else echo "SOME SMOKE TESTS FAILED"; fi
exit "$fail"
