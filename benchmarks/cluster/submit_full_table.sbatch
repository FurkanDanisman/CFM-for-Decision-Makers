#!/bin/bash
#SBATCH --job-name=full-table
#SBATCH --output=logs_full_table/ft_%j.out
#SBATCH --error=logs_full_table/ft_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
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
TAG="B${MALC_B}_K${MALC_K}"

source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ATOM_GRID_MAX="${ATOM_GRID_MAX:-4096}"
PT="$REPO/realcause_eval/point_raw_em.py"
CD="$REPO/UWYK_Fig3_4/cate_density_metrics.py"
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
      raw)   for t in cate ate; do echo "  raw $ds/$t"
        run "$CD" --root "$R" --dataset "$ds" --target "$t" --tau-smoother none \
            --out "$R/calib_${ds}_raw_${t}.md"; done ;;
      malc)  for t in cate ate; do echo "  MALC $ds/$t"
        run "$CD" --root "$R" --dataset "$ds" --target "$t" --tau-smoother malc \
            --malc-B "$MALC_B" --malc-K "$MALC_K" --n-tau "$N_TAU" \
            --malc-workers "$NW" --out "$R/calib_${ds}_T_${TAG}_${t}.md"; done ;;
      indep) for t in cate ate; do echo "  indep $ds/$t"
        run "$CD" --root "$R" --dataset "$ds" --target "$t" --tau-smoother none \
            --joint-coupling indep --out "$R/calib_${ds}_indep_${t}.md"; done ;;
    esac; done
  done
done

# ── Case studies: one cell x one case per invocation ─────────────────────────
for R in "$CS_ROOT" "$ETA0_CS"; do
  [ -d "$R" ] || continue
  echo "==== CaseStudy  root=$R ===="
  for cell in "$R"/shift*/d*/ctx1000; do
    [ -d "$cell" ] || continue
    echo "  cell ${cell#$R/}"
    for c in $CS_CASES; do
      for S in $STAGES; do case "$S" in
        point) run "$PT" --root "$cell" --dataset "$c" --modes raw \
                   --out-md "$cell/point_raw_em_${c}.md" ;;
        raw)   for t in cate ate; do
                 run "$CD" --root "$cell" --dataset "$c" --target "$t" \
                     --tau-smoother none --out "$cell/calib_raw_${t}_${c}.md"; done ;;
        malc)  for t in cate ate; do
                 run "$CD" --root "$cell" --dataset "$c" --target "$t" \
                     --tau-smoother malc --malc-B "$MALC_B" --malc-K "$MALC_K" \
                     --n-tau "$N_TAU" --malc-workers "$NW" \
                     --out "$cell/calib_T_${TAG}_${t}_${c}.md"; done ;;
        # No indep stage here: forced independence is a RealCause-only ablation.
      # On the case studies the arms are near-comonotonic (rho = 1), so imposing
      # rho = 0 scores the imposed assumption rather than the head.
      indep) ;;
      esac; done
    done
  done
done

echo "[$(date '+%F %T')] done (FAILED=$FAILED)"
