#!/bin/bash
#SBATCH --job-name=malc-cs
#SBATCH --output=logs_malc_cs/run_%A_%a.out
#SBATCH --error=logs_malc_cs/run_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --array=0-23
#
# CPU ONLY. The case-study half of the MALC study. Split out from
# submit_malc_all_benchmarks.sbatch because the case-study dumps are not one
# root -- they are a (shift x d) grid, each cell a separate scorer root:
#
#   <CS_ROOT>/shift{0,+2,-2}/d{2,3,5,10,20,30,40,50}/ctx<CTX>/<method>/<CASE>/
#
# 3 shifts x 8 d values = 24 tasks, one cell each. Every task scores all six
# cases in that cell in a single invocation (the scorer takes several
# --dataset names), for both the raw and variant-T constructions.
#
#   task = shift_idx * 8 + d_idx
#     shift_idx 0,1,2 -> 0, +2, -2
#     d_idx     0..7  -> 2 3 5 10 20 30 40 50
#
# The shift is a DGP-level constant treatment effect beta, present so the true
# ATE is never near zero and L1_ATE has a well-defined denominator.
#
# Env:
#   CS_ROOT   (default $SCRATCH/cs_dvar_dens)
#   CTX       context size, the directory level under d<N>  (default 1000)
#   SHIFTS    (default "0 +2 -2")     D_LIST  (default "2 3 5 10 20 30 40 50")
#   CASES     (default all six)
#   MALC_B / MALC_K / N_TAU / NW / STAGES / TARGETS / MAX_REAL / METHODS
#             as in submit_malc_all_benchmarks.sbatch
#
#   sbatch benchmarks/cluster/submit_malc_case_studies.sbatch
#   sbatch --array=0 --time=00:30:00 --export=ALL,MAX_REAL=3 ...   # smoke

set -uo pipefail
mkdir -p logs_malc_cs

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
CS_ROOT="${CS_ROOT:-$SCRATCH/cs_dvar_dens}"
CTX="${CTX:-1000}"
MALC_B="${MALC_B:-100}"
MALC_K="${MALC_K:-1}"
N_TAU="${N_TAU:-4001}"
NW="${NW:-${SLURM_CPUS_PER_TASK:-32}}"
STAGES="${STAGES:-point raw t}"
export ATOM_GRID_MAX="${ATOM_GRID_MAX:-4096}"

read -r -a SH <<< "${SHIFTS:-0 +2 -2}"
read -r -a DD <<< "${D_LIST:-2 3 5 10 20 30 40 50}"
CASES="${CASES:-Observed_Confounder Observed_Mediator Observed_Mediator_and_Confounder Unobserved_Confounder Frontdoor_Criterion Backdoor_Criterion}"

TID="${SLURM_ARRAY_TASK_ID:-0}"
NS=${#SH[@]}; ND=${#DD[@]}
(( TID < NS * ND )) || { echo "FATAL: task $TID exceeds ${NS}x${ND} grid" >&2; exit 2; }
SHIFT="${SH[$((TID / ND))]}"
D="${DD[$((TID % ND))]}"

ROOT="$CS_ROOT/shift${SHIFT}/d${D}/ctx${CTX}"
[ -d "$ROOT" ] || { echo "FATAL: cell not found: $ROOT" >&2; exit 1; }

source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$REPO/benchmarks:$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

MD=(); [ -n "${METHODS:-}"  ] && MD=(--methods ${METHODS})
MR=(); [ -n "${MAX_REAL:-}" ] && MR=(--max-real "$MAX_REAL")
TAG="B${MALC_B}_K${MALC_K}"
SUF="shift${SHIFT}_d${D}"
FAILED=0

echo "[malc-cs] task=$TID  shift=$SHIFT  d=$D  ctx=$CTX"
echo "          root=$ROOT"
echo "          B=$MALC_B K=$MALC_K n_tau=$N_TAU workers=$NW stages=[$STAGES]"
echo "          cases: $CASES"

for S in $STAGES; do
  case "$S" in
    point)
      for C in $CASES; do
        python -u "$REPO/realcause_eval/point_raw_em.py" \
            --root "$ROOT" --dataset "$C" --modes raw em "${MD[@]}" "${MR[@]}" \
            --out-md "$ROOT/point_raw_em_${C}.md" \
          || { echo "ERROR: point/$C FAILED" >&2; FAILED=1; }
      done
      ;;
    raw)
      for T in ${TARGETS:-cate ate}; do
        echo "=== RAW  $SUF  target=$T ==="
        python -u "$REPO/UWYK_Fig3_4/cate_density_metrics.py" \
            --root "$ROOT" --dataset $CASES --target "$T" --tau-smoother none \
            "${MD[@]}" "${MR[@]}" --out "$ROOT/calib_raw_${T}.md" \
          || { echo "ERROR: raw/$T FAILED" >&2; FAILED=1; }
      done
      ;;
    t)
      for T in ${TARGETS:-cate ate}; do
        echo "=== T  $SUF  target=$T  B=$MALC_B ==="
        python -u "$REPO/UWYK_Fig3_4/cate_density_metrics.py" \
            --root "$ROOT" --dataset $CASES --target "$T" \
            --tau-smoother malc --malc-B "$MALC_B" --malc-K "$MALC_K" \
            --n-tau "$N_TAU" --malc-workers "$NW" "${MD[@]}" "${MR[@]}" \
            --out "$ROOT/calib_T_${TAG}_${T}.md" \
          || { echo "ERROR: T/$T FAILED" >&2; FAILED=1; }
      done
      ;;
    *) echo "WARN: unknown stage '$S'" ;;
  esac
done

echo "[$(date '+%F %T')] done (FAILED=$FAILED)  shift=$SHIFT d=$D"
