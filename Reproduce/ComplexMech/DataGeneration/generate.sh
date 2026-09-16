#!/bin/bash
# Generate the PEHE-able ComplexMech benchmark. CPU only, ~3 min.
#
# Needs UWYK_SRC / UWYK_ROOT pointing at the UWYK checkout, because the
# generator drives UWYK's OWN samplers and YAML configs rather than
# reimplementing the prior.
#
# OMP_NUM_THREADS=1 is set before torch is imported: torch and XGBoost each
# ship an OpenMP runtime and segfault on ComplexMech otherwise.
set -euo pipefail
DEPLOY_ROOT="${DEPLOY_ROOT:-$SCRATCH/rpfn_bench_kit}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
export UWYK_SRC="${UWYK_SRC:-$DEPLOY_ROOT/external/uwyk_reproduce/src}"
export UWYK_ROOT="${UWYK_ROOT:-$DEPLOY_ROOT/external/uwyk_reproduce}"
source "$DEPLOY_ROOT/venv/bin/activate"
python -u "$REPO/UWYK_Fig3_4/generate_pehe_benchmark.py" \
    --prior complexmech --nodes 5 10 20 30 40 50 --regimes path_TY \
    --hide-fractions 0.0 --n-realizations 100 \
    --out-dir "${UWYK_FIG34_DATA:-$REPO/UWYK_Fig3_4/data}"
echo
echo "self-test (null-regime zero CATE, T in {0,1}, bitwise regeneration):"
python -u "$REPO/UWYK_Fig3_4/generate_pehe_benchmark.py" --self-test
