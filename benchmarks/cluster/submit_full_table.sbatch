#!/bin/bash
#SBATCH --job-name=full-table
#SBATCH --output=logs_full_table/ft_%j.out
#SBATCH --error=logs_full_table/ft_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
# No --gres: this job is numpy only and never touches a GPU. Left on a GPU
# partition it queues behind real GPU work for nothing -- which is what starved
# the joint2d A/B. Pass --partition on the sbatch line to place it on CPU nodes.
#
# CPU ONLY. The full reporting matrix for RealCause and the case studies:
#
#   PEHE / ATEerr        point estimates, raw mean (no MALC)
#   Cov / Len / IS       raw tau density
#   Cov / Len / IS MALC  variant T
#   + RealCause ONLY, 2D heads: the same with the coupling FORCED INDEPENDENT
#
# The forced-independent pass is the controlled test of the joint head: same
# model, same marginals, same data -- only the coupling differs. So any score
# difference is attributable to the learned dependency alone, which no
# leaderboard column can isolate (rho affects only tau's spread, never its mean,
# so every accuracy-dominated metric is nearly blind to it).
#
# Scoring only. Every model listed in cate_density_metrics.METHODS with dumps
# under the given roots is picked up; entries without dumps print (no dumps) and
# are skipped, so this is safe to run before the remaining checkpoints are
# dumped.
#
# Env:
#   RC_ROOT / CS_ROOT   dump roots (defaults below)
#   ETA0_RC / ETA0_CS   the eta0 model's own roots (scored separately -- it
#                       lives under its own tree, not beside the others)
#   MALC_B / MALC_K / N_TAU / MALC_WORKERS
#   CS_CASES            (default all six)
#   STAGES              subset of "point raw malc indep"  (default all)
#
#   sbatch benchmarks/cluster/submit_full_table.sbatch

set -uo pipefail
mkdir -p logs_full_table

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
RC_ROOT="${RC_ROOT:-$SCRATCH/rc_dens_uni}"
CS_ROOT="${CS_ROOT:-$SCRATCH/cs_dvar_dens}"
ETA0_RC="${ETA0_RC:-$SCRATCH/rc_dens_eta0}"
ETA0_CS="${ETA0_CS:-$SCRATCH/cs_dvar_eta0}"
MALC_B="${MALC_B:-1000}"; MALC_K="${MALC_K:-1}"; N_TAU="${N_TAU:-4001}"
NW="${MALC_WORKERS:-${SLURM_CPUS_PER_TASK:-64}}"
STAGES="${STAGES:-point raw malc indep}"
CS_CASES="${CS_CASES:-Observed_Confounder Observed_Mediator Observed_Mediator_and_Confounder Unobserved_Confounder Frontdoor_Criterion Backdoor_Criterion}"
RC_DATASETS="${RC_DATASETS:-IHDP ACIC CPS PSID PSID_bal}"
# ONE_RC_ROOT / ONE_CS_ROOT: score exactly one root, so the driver can put one
# (root, slice) in each job. MALC at B=1000 is the whole cost here, and a single
# job cannot cover 13 models x 2 benchmarks inside a wall clock.
[ -n "${ONE_RC_ROOT:-}" ] && RC_ROOT="$ONE_RC_ROOT"
[ -n "${ONE_CS_ROOT:-}" ] && CS_ROOT="$ONE_CS_ROOT"
[ -n "${ONE_RC_ROOT:-}${ONE_CS_ROOT:-}" ] && { ETA0_RC=/nonexistent; ETA0_CS=/nonexistent; }
[ "${SKIP_RC:-0}" = 1 ] && RC_ROOT=/nonexistent
[ "${SKIP_CS:-0}" = 1 ] && CS_ROOT=/nonexistent
# CS_SHIFT restricts the case-study walk to one shift, matching one dump job.
CS_SHIFT="${CS_SHIFT:-}"
# CS_D restricts the walk to one d value. Needed because MALC at B=1000 on the
# multi-method root is ~3.4M fits per shift (8 d x 6 cases x 100 real x 100
# queries x ~7 methods), roughly 18h over 64 workers -- far past any 3h tier.
# Per (shift, d) it is ~2.5h, which fits.
CS_D="${CS_D:-}"
# Per-realization dumps: one file per (stage, reporting-group, pooled-slice), so
# collect_table.py can pool exactly and can build a table from a partial run.
PERREAL="${PERREAL:-$SCRATCH/perreal}"
LABEL="${LABEL:-$(basename "$(dirname "${ONE_RC_ROOT:-${ONE_CS_ROOT:-orig}}")")}"
mkdir -p "$PERREAL/$LABEL"
TAG="B${MALC_B}_K${MALC_K}"

source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ATOM_GRID_MAX="${ATOM_GRID_MAX:-4096}"
PT="$REPO/realcause_eval/point_raw_em.py"
CD="$REPO/UWYK_Fig3_4/cate_density_metrics.py"
CBR="$REPO/benchmarks/coverage_by_realization.py"
FAILED=0
run() { python -u "$@" >/dev/null 2>&1 || { echo "  ERROR: $*" >&2; FAILED=1; }; }

# ── RealCause ────────────────────────────────────────────────────────────────
for R in "$RC_ROOT" "$ETA0_RC"; do
  [ -d "$R" ] || continue
  echo "==== RealCause  root=$R ===="
  for ds in $RC_DATASETS; do
    for S in $STAGES; do case "$S" in
      point) echo "  point $ds"
        run "$PT" --root "$R" --dataset "$ds" --modes raw \
            --out-md "$R/point_raw_em_${ds}.md" ;;
      raw)   echo "  raw $ds/cate (per-realization)"
        # CATE coverage is the MEAN OF PER-REALIZATION COVERAGES +- sd, not the
        # pooled mean over all (realization, query) pairs. Pooling hid spreads
        # larger than the means themselves, so every CATE Cov/Len/IS number comes
        # from coverage_by_realization.
        run "$CBR" --root "$R" --dataset "$ds" --tau-smoother none \
            --dump-per-real "$PERREAL/$LABEL/raw__${ds}__-.npz" \
            > "$R/cbr_${ds}_raw_cate.md"
        echo "  raw $ds/ate"
        # ATE needs no per-realization treatment: one interval and one truth per
        # realization already, so the pooled mean IS the per-realization mean.
        run "$CD" --root "$R" --dataset "$ds" --target ate --tau-smoother none \
            --out "$R/calib_${ds}_raw_ate.md" ;;
      malc)  echo "  MALC $ds/cate (per-realization, B=$MALC_B K=$MALC_K)"
        run "$CBR" --root "$R" --dataset "$ds" --tau-smoother malc \
            --malc-B "$MALC_B" --malc-K "$MALC_K" --n-tau "$N_TAU" \
            --malc-workers "$NW" \
            --dump-per-real "$PERREAL/$LABEL/malc__${ds}__-.npz" \
            > "$R/cbr_${ds}_T_${TAG}_cate.md"
        echo "  MALC $ds/ate"
        run "$CD" --root "$R" --dataset "$ds" --target ate --tau-smoother malc \
            --malc-B "$MALC_B" --malc-K "$MALC_K" --n-tau "$N_TAU" \
            --malc-workers "$NW" --out "$R/calib_${ds}_T_${TAG}_ate.md" ;;
      indep) echo "  indep $ds/cate raw + MALC (RealCause only, 2D heads)"
        run "$CBR" --root "$R" --dataset "$ds" --tau-smoother none \
            --joint-coupling indep \
            --dump-per-real "$PERREAL/$LABEL/indep_raw__${ds}__-.npz" \
            > "$R/cbr_${ds}_indep_raw_cate.md"
        run "$CBR" --root "$R" --dataset "$ds" --tau-smoother malc \
            --malc-B "$MALC_B" --malc-K "$MALC_K" --n-tau "$N_TAU" \
            --malc-workers "$NW" --joint-coupling indep \
            --dump-per-real "$PERREAL/$LABEL/indep_malc__${ds}__-.npz" \
            > "$R/cbr_${ds}_indep_T_${TAG}_cate.md" ;;
    esac; done
  done
done

# ── Case studies: one cell x one case per invocation ─────────────────────────
for R in "$CS_ROOT" "$ETA0_CS"; do
  [ -d "$R" ] || continue
  echo "==== CaseStudy  root=$R ===="
  for cell in "$R"/shift${CS_SHIFT:-*}/d${CS_D:-*}/ctx1000; do
    [ -d "$cell" ] || continue
    echo "  cell ${cell#$R/}"
    # shift and d out of <root>/shift<S>/d<D>/ctx1000
    _rel="${cell#$R/}"; _sh="${_rel%%/*}"; _d="${_rel#*/}"; _d="${_d%%/*}"
    for c in $CS_CASES; do
      for S in $STAGES; do case "$S" in
        point) run "$PT" --root "$cell" --dataset "$c" --modes raw \
                   --out-md "$cell/point_raw_em_${c}.md" ;;
        raw)   run "$CBR" --root "$cell" --dataset "$c" --tau-smoother none \
                   --dump-per-real "$PERREAL/$LABEL/raw__${_d}_${c}__${_sh}.npz" \
                   > "$cell/cbr_raw_cate_${c}.md"
               run "$CD" --root "$cell" --dataset "$c" --target ate \
                   --tau-smoother none --out "$cell/calib_raw_ate_${c}.md" ;;
        malc)  run "$CBR" --root "$cell" --dataset "$c" --tau-smoother malc \
                   --malc-B "$MALC_B" --malc-K "$MALC_K" --n-tau "$N_TAU" \
                   --malc-workers "$NW" \
                   --dump-per-real "$PERREAL/$LABEL/malc__${_d}_${c}__${_sh}.npz" \
                   > "$cell/cbr_T_${TAG}_cate_${c}.md"
               run "$CD" --root "$cell" --dataset "$c" --target ate \
                   --tau-smoother malc --malc-B "$MALC_B" --malc-K "$MALC_K" \
                   --n-tau "$N_TAU" --malc-workers "$NW" \
                   --out "$cell/calib_T_${TAG}_ate_${c}.md" ;;
        # No indep stage here: forced independence is a RealCause-only ablation.
      # On the case studies the arms are near-comonotonic (rho = 1), so imposing
      # rho = 0 scores the imposed assumption rather than the head.
      indep) ;;
      esac; done
    done
  done
done

echo "[$(date '+%F %T')] done (FAILED=$FAILED)"
