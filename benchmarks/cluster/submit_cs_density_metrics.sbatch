#!/bin/bash
#SBATCH --job-name=cs-dens-metrics
#SBATCH --output=logs_cs_density/metrics_%j.out
#SBATCH --error=logs_cs_density/metrics_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#
# CPU-only scorer for the case-study density dumps produced by
# submit_cs_density_1d_vs_2d.sbatch.
#
# Emits coverage95 / length / IS_0.05 per method, for both estimands:
#   --target cate : each query's tau density vs its true tau
#   --target ate  : the Wasserstein barycenter of a realization's query
#                   densities vs that realization's realized ATE
#
# Run this AFTER the 42-task array. Typical chain:
#   AID=$(sbatch --parsable .../submit_cs_density_1d_vs_2d.sbatch)
#   sbatch --dependency=afterany:$AID .../submit_cs_density_metrics.sbatch
#
# Required env:
#   OUT_ROOT     same value given to the density array
# Optional env:
#   COUPLING     indep | comonotonic  (default indep)
#                how 1D heads turn two marginals into p(tau). 2D heads ignore
#                it. `indep` scores the independence assumption as much as the
#                head, so on arms with correlated noise it reads as over-wide;
#                run both if that matters.
#   METHODS      space-separated subset of method labels (default: all)
#   MIN_DENSITY_R2  consistency gate (default 0.99) — realizations whose dumped
#                density does not reproduce their own point CATE are dropped.

set -euo pipefail
mkdir -p logs_cs_density

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT required}"
COUPLING="${COUPLING:-indep}"
export MIN_DENSITY_R2="${MIN_DENSITY_R2:-0.99}"

CASES=(Observed_Confounder
       Observed_Mediator
       Observed_Mediator_and_Confounder
       Unobserved_Confounder
       Frontdoor_Criterion
       Backdoor_Criterion)

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

METHOD_ARG=()
[ -n "${METHODS:-}" ] && METHOD_ARG=(--methods ${METHODS})

for TARGET in cate ate; do
    echo "================ target=$TARGET  coupling=$COUPLING ================"
    python -u "$REPO/UWYK_Fig3_4/cate_density_metrics.py" \
        --root "$OUT_ROOT" \
        --dataset "${CASES[@]}" \
        --target "$TARGET" \
        --coupling "$COUPLING" \
        "${METHOD_ARG[@]}"

    # The multi-dataset writer hardcodes `realcause` in the filename (it was
    # written for that benchmark, and --dataset mode is shared). Rename so the
    # case-study tables don't masquerade as RealCause ones.
    SRC="$OUT_ROOT/${TARGET}_density_realcause_${COUPLING}.md"
    DST="$OUT_ROOT/${TARGET}_density_case_studies_${COUPLING}.md"
    [ -f "$SRC" ] && mv "$SRC" "$DST" && echo "[written] $DST"
done

echo "[$(date '+%F %T')] done"
