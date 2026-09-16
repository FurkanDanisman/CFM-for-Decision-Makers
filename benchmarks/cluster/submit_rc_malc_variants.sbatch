#!/bin/bash
#SBATCH --job-name=rc-malc-variants
#SBATCH --output=logs_rc_malc_variants/run_%j.out
#SBATCH --error=logs_rc_malc_variants/run_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#
# CPU ONLY. The RealCause MALC-variant smoke test, all in one job:
#
#   POINT   PEHE + eps_ATE from the density dumps under raw-mean and EM-mean.
#           No MALC. Seconds.
#   RAW     coverage / length / IS_0.05 from the unsmoothed tau density.
#           The baseline every other column is read against. Seconds.
#   T       the same metrics after MALC-1D smoothing of the tau density
#           (variant T: project/convolve FIRST, smooth second). SLOW --
#           ~65 ms per query at B=100, so ~70 min for IHDP's 100 realizations
#           on one core. Use MAX_REAL to cut it while iterating.
#
# Variant J -- smooth the JOINT, then diagonal-integrate -- is NOT here. It is
# realcause_eval/compute_malc_ci_cell{,_1d}.py, which write per-realization
# files that a separate aggregator reads. Run those with --malc-K 1
# --malc-B 100 --n-eval 2001 (2*2001-1 = 4001 anti-diagonals, matching N_TAU
# exactly, with no resampling and a quarter the cost of n_eval=4000).
#
# 96G is not padding: the ATE target barycenters every query of a realization,
# and dopfn_native's rebinned tau support reaches 2*ATOM_GRID_MAX-1 = 8191
# atoms per query.
#
# Env:
#   ROOT        density-dump root        (default $SCRATCH/rc_dens_uni)
#   DATASET     (default IHDP)
#   STAGES      subset of "point raw t"  (default all three)
#   TARGETS     "cate ate"               (default both, calibration stages only)
#   METHODS     optional subset of the METHODS labels
#   MAX_REAL    cap realizations per cell -- the knob for a fast smoke
#   MALC_B      synthetic points per fit (default 100; cost grows ~B^3.7)
#   MALC_K      mixture components (default 1 = tau forced unimodal)
#   N_TAU       smoothed tau grid points (default 4001)
#   ATOM_GRID_MAX  dopfn_native cap (default 4096 -- do NOT vary across
#                  benchmarks, it sets the tau resolution)
#
#   sbatch benchmarks/cluster/submit_rc_malc_variants.sbatch

set -uo pipefail
mkdir -p logs_rc_malc_variants

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
ROOT="${ROOT:-$SCRATCH/rc_dens_uni}"
DATASET="${DATASET:-IHDP}"
STAGES="${STAGES:-point raw t}"
MALC_B="${MALC_B:-100}"
MALC_K="${MALC_K:-1}"
N_TAU="${N_TAU:-4001}"
export ATOM_GRID_MAX="${ATOM_GRID_MAX:-4096}"

[ -d "$ROOT" ] || { echo "FATAL: dump root not found: $ROOT" >&2; exit 1; }

source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH="$REPO/benchmarks:$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

MD=(); [ -n "${METHODS:-}"  ] && MD=(--methods ${METHODS})
MR=(); [ -n "${MAX_REAL:-}" ] && MR=(--max-real "$MAX_REAL")

echo "[rc-malc] root=$ROOT  dataset=$DATASET  stages=[$STAGES]"
echo "          B=$MALC_B  K=$MALC_K  n_tau=$N_TAU  ATOM_GRID_MAX=$ATOM_GRID_MAX"
echo "          max_real=${MAX_REAL:-<all>}  methods=${METHODS:-<all>}"

for S in $STAGES; do
  case "$S" in
    point)
      echo "=== POINT (raw + EM, no MALC) ==="
      python -u "$REPO/realcause_eval/point_raw_em.py" \
          --root "$ROOT" --dataset "$DATASET" --modes raw em \
          "${MD[@]}" "${MR[@]}" \
          --out-md "$ROOT/point_raw_em_${DATASET}.md" \
        || echo "WARN: point stage exited nonzero"
      ;;
    raw)
      for T in ${TARGETS:-cate ate}; do
        echo "=== CALIB raw  target=$T ==="
        python -u "$REPO/UWYK_Fig3_4/cate_density_metrics.py" \
            --root "$ROOT" --dataset "$DATASET" --target "$T" \
            --tau-smoother none "${MD[@]}" "${MR[@]}" \
            --out-md "$ROOT/calib_${DATASET}_raw_${T}.md" \
          || echo "WARN: raw/$T exited nonzero"
      done
      ;;
    t)
      for T in ${TARGETS:-cate ate}; do
        echo "=== CALIB variant T  target=$T  (slow) ==="
        python -u "$REPO/UWYK_Fig3_4/cate_density_metrics.py" \
            --root "$ROOT" --dataset "$DATASET" --target "$T" \
            --tau-smoother malc --malc-B "$MALC_B" --malc-K "$MALC_K" \
            --n-tau "$N_TAU" "${MD[@]}" "${MR[@]}" \
            --out-md "$ROOT/calib_${DATASET}_T_B${MALC_B}_K${MALC_K}_${T}.md" \
          || echo "WARN: T/$T exited nonzero"
      done
      ;;
    *) echo "WARN: unknown stage '$S'" ;;
  esac
done

echo "[$(date '+%F %T')] done. Tables under $ROOT:"
ls -1 "$ROOT"/point_raw_em_"$DATASET".md "$ROOT"/calib_"$DATASET"_*.md 2>/dev/null || true
