#!/bin/bash
#SBATCH --job-name=malc-all
#SBATCH --output=logs_malc_all/run_%A_%a.out
#SBATCH --error=logs_malc_all/run_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --array=0-5
#
# CPU ONLY. Full-scale MALC variant-T calibration across all three benchmarks,
# plus the raw baseline and the raw/EM point estimates. K=1, B configurable
# (default 1000).
#
#   task 0-4 : RealCause  IHDP / ACIC / CPS / PSID / PSID_bal
#   task 5   : ComplexMech
#   task 6   : REMOVED -- the case-study dumps form a (shift x d) grid of
#              separate scorer roots, so they get their own 24-task array:
#              benchmarks/cluster/submit_malc_case_studies.sbatch
#
# Each task runs, for its benchmark:
#   POINT   PEHE + eps_ATE (RealCause) or L1_ATE (the other two), under raw-mean
#           and EM-mean. No MALC. Seconds.
#   RAW     coverage / length / IS_0.05 from the unsmoothed tau density.
#   T       the same after MALC-1D smoothing of the tau density.
#
# WHY B=1000 IS AFFORDABLE NOW. The 1D MLE was a dense QP costing ~B^3.7; at
# B=1000 that was ~150 s per fit. It is now bound-constrained via the slope-gap
# substitution and runs in 1.6 s, and the per-query fits go through a process
# pool. Budget ~1.2 s / n_workers per query. IHDP (~67k queries) lands near 40
# minutes on 32 workers; the wider datasets scale with their query counts.
#
# Variant J is NOT here -- submit_rc_malc_J.sbatch, once n_eval=2001 is shown
# to be tractable.
#
# Env:
#   RC_ROOT      RealCause dumps      (default $SCRATCH/rc_dens_uni)
#   CMECH_ROOT   ComplexMech dumps    (default $SCRATCH/cmech_1d2d)
#   CS_ROOT      case-study dumps     (default $SCRATCH/cs_dvar)
#   MALC_B       (default 100)
#   MALC_K       (default 1)
#   N_TAU        (default 4001)
#   NW           MALC workers         (default = cpus-per-task)
#   STAGES       subset of "point raw t"  (default all)
#   TARGETS      "cate ate"           (default both)
#   MAX_REAL     cap realizations -- use for a smoke pass
#   CTX / NODES / SUBSET   ComplexMech selectors (1000 / "5 10 20 30 40 50" / total)
#
#   sbatch benchmarks/cluster/submit_malc_all_benchmarks.sbatch
#   sbatch --array=0 ... --export=ALL,MAX_REAL=3   # single-cell smoke

set -uo pipefail
mkdir -p logs_malc_all

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
CMECH_ROOT="${CMECH_ROOT:-$SCRATCH/cmech_1d2d}"
CS_ROOT="${CS_ROOT:-$SCRATCH/cs_dvar}"
MALC_B="${MALC_B:-100}"
MALC_K="${MALC_K:-1}"
N_TAU="${N_TAU:-4001}"
NW="${NW:-${SLURM_CPUS_PER_TASK:-32}}"
STAGES="${STAGES:-point raw t}"
CTX="${CTX:-1000}"
NODES="${NODES:-5 10 20 30 40 50}"
SUBSET="${SUBSET:-total}"
# Fixed across ALL benchmarks. Varying it would put dopfn_native's numbers on
# different tau resolutions and make the three tables incomparable.
export ATOM_GRID_MAX="${ATOM_GRID_MAX:-4096}"

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) BENCH=realcause; DATASET=IHDP     ;;
  1) BENCH=realcause; DATASET=ACIC     ;;
  2) BENCH=realcause; DATASET=CPS      ;;
  3) BENCH=realcause; DATASET=PSID     ;;
  4) BENCH=realcause; DATASET=PSID_bal ;;
  5) BENCH=cmech;     DATASET=CMECH    ;;
  6) echo "Case studies are NOT one root: the dumps form a (shift x d) grid," >&2
     echo "each cell its own scorer root at" >&2
     echo "  <CS_ROOT>/shift{0,+2,-2}/d{2..50}/ctx<CTX>/<method>/<CASE>/" >&2
     echo "Use benchmarks/cluster/submit_malc_case_studies.sbatch (24 tasks)." >&2
     exit 2 ;;
  *) echo "invalid SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

case "$BENCH" in
  realcause) ROOT="$RC_ROOT";    SEL=(--dataset "$DATASET") ;;
  cmech)     ROOT="$CMECH_ROOT"; SEL=(--context "$CTX" --nodes $NODES --subset "$SUBSET"
                                      --data-root "$REPO/UWYK_Fig3_4/data") ;;
esac
[ -d "$ROOT" ] || { echo "FATAL: dump root not found: $ROOT" >&2; exit 1; }

source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1      # the pool provides the parallelism; BLAS threads
export MKL_NUM_THREADS=1      # on top of 32 processes only oversubscribe
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$REPO/benchmarks:$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

MD=(); [ -n "${METHODS:-}"  ] && MD=(--methods ${METHODS})
MR=(); [ -n "${MAX_REAL:-}" ] && MR=(--max-real "$MAX_REAL")
TAG="B${MALC_B}_K${MALC_K}"
FAILED=0

echo "[malc-all] task=${SLURM_ARRAY_TASK_ID:-0} bench=$BENCH dataset=$DATASET"
echo "           root=$ROOT  B=$MALC_B K=$MALC_K n_tau=$N_TAU workers=$NW"
echo "           stages=[$STAGES]  max_real=${MAX_REAL:-<all>}  ATOM_GRID_MAX=$ATOM_GRID_MAX"

for S in $STAGES; do
  case "$S" in
    point)
      # point_raw_em resolves cells by --dataset, so it covers RealCause and
      # the case studies. ComplexMech cells are keyed by context/nodes/subset
      # instead, and its point estimates come from its own aggregator.
      if [ "$BENCH" = "cmech" ]; then
        echo "=== POINT skipped for ComplexMech (different cell layout) ==="
      else
        for DS in $( [ "$BENCH" = "realcause" ] && echo "$DATASET" || echo "${SEL[@]:1}" ); do
          echo "=== POINT $DS ==="
          python -u "$REPO/realcause_eval/point_raw_em.py" \
              --root "$ROOT" --dataset "$DS" --modes raw em "${MD[@]}" "${MR[@]}" \
              --out-md "$ROOT/point_raw_em_${DS}.md" \
            || { echo "ERROR: point/$DS FAILED" >&2; FAILED=1; }
        done
      fi
      ;;
    raw)
      for T in ${TARGETS:-cate ate}; do
        echo "=== RAW  target=$T ==="
        python -u "$REPO/UWYK_Fig3_4/cate_density_metrics.py" \
            --root "$ROOT" "${SEL[@]}" --target "$T" --tau-smoother none \
            "${MD[@]}" "${MR[@]}" \
            --out "$ROOT/calib_${DATASET}_raw_${T}.md" \
          || { echo "ERROR: raw/$T FAILED" >&2; FAILED=1; }
      done
      ;;
    t)
      for T in ${TARGETS:-cate ate}; do
        echo "=== T  target=$T  B=$MALC_B K=$MALC_K workers=$NW ==="
        python -u "$REPO/UWYK_Fig3_4/cate_density_metrics.py" \
            --root "$ROOT" "${SEL[@]}" --target "$T" \
            --tau-smoother malc --malc-B "$MALC_B" --malc-K "$MALC_K" \
            --n-tau "$N_TAU" --malc-workers "$NW" "${MD[@]}" "${MR[@]}" \
            --out "$ROOT/calib_${DATASET}_T_${TAG}_${T}.md" \
          || { echo "ERROR: T/$T FAILED" >&2; FAILED=1; }
      done
      ;;
    *) echo "WARN: unknown stage '$S'" ;;
  esac
done

echo "[$(date '+%F %T')] done (FAILED=$FAILED)  bench=$BENCH dataset=$DATASET"
