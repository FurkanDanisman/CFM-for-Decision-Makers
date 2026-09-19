#!/bin/bash
#SBATCH --job-name=cov-by-real
#SBATCH --output=logs_cov_by_real/cov_%j.out
#SBATCH --error=logs_cov_by_real/cov_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#
# CPU ONLY. Per-realization coverage / length / IS / CRPS -- raw AND variant T --
# for all three benchmarks, in one table per selector.
#
# Aggregation is one number per realization, then mean +- sd across
# realizations. The main scorer pools every (realization, query) pair into a
# single mean, which is the right point estimate but an error bar ~10x too
# small: queries inside a realization share a context and an SCM, so they are
# clustered, not independent.
#
# The T pass RE-FITS MALC -- the earlier runs persisted aggregated tables only,
# so a per-realization MALC number cannot be read off disk. Budget
# ~0.065 s/query at B=100 and ~1.2 s at B=1000, divided by MALC_WORKERS.
#
# Env:
#   RC_ROOT / CM_ROOT / CS_ROOT   dump roots (defaults below)
#   CS_CELL       case-study cell under CS_ROOT (default shift0/d5/ctx1000)
#   CS_CASE       (default Observed_Confounder)
#   RC_DATASETS   (default "IHDP ACIC CPS PSID PSID_bal")
#   CM_NODES      (default "5 10 20 30 40 50")
#   MALC_B / MALC_K / N_TAU / MALC_WORKERS
#   VARIANTS      subset of "none malc"  (default both)
#   OUT           text file for the tables (default $SCRATCH/cov_by_real.txt)
#
#   sbatch benchmarks/cluster/submit_cov_by_real.sbatch

set -uo pipefail
mkdir -p logs_cov_by_real

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
CM_ROOT="${CM_ROOT:-$SCRATCH/cmech_dens}"
CS_ROOT="${CS_ROOT:-$SCRATCH/cs_dvar_dens}"
CS_CELL="${CS_CELL:-shift0/d5/ctx1000}"
CS_CASE="${CS_CASE:-Observed_Confounder}"
RC_DATASETS="${RC_DATASETS:-IHDP ACIC CPS PSID PSID_bal}"
CM_NODES="${CM_NODES:-5 10 20 30 40 50}"
MALC_B="${MALC_B:-100}"; MALC_K="${MALC_K:-1}"; N_TAU="${N_TAU:-4001}"
MALC_WORKERS="${MALC_WORKERS:-${SLURM_CPUS_PER_TASK:-64}}"
VARIANTS="${VARIANTS:-none malc}"
OUT="${OUT:-$SCRATCH/cov_by_real.txt}"

source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ATOM_GRID_MAX="${ATOM_GRID_MAX:-4096}"
S="$REPO/benchmarks/coverage_by_realization.py"
M=(--malc-B "$MALC_B" --malc-K "$MALC_K" --n-tau "$N_TAU" --malc-workers "$MALC_WORKERS")

: > "$OUT"
for V in $VARIANTS; do
  echo "################ variant=$V ################" | tee -a "$OUT"
  for ds in $RC_DATASETS; do
    echo "---- RealCause $ds ----" | tee -a "$OUT"
    python -u "$S" --root "$RC_ROOT" --dataset "$ds" \
        --tau-smoother "$V" "${M[@]}" 2>&1 | tee -a "$OUT"
  done
  for nd in $CM_NODES; do
    echo "---- ComplexMech d=$nd ----" | tee -a "$OUT"
    python -u "$S" --root "$CM_ROOT" --context 1000 --nodes "$nd" --subset total \
        --tau-smoother "$V" "${M[@]}" 2>&1 | tee -a "$OUT"
  done
  echo "---- CaseStudy $CS_CELL / $CS_CASE ----" | tee -a "$OUT"
  python -u "$S" --root "$CS_ROOT/$CS_CELL" --dataset "$CS_CASE" \
      --tau-smoother "$V" "${M[@]}" 2>&1 | tee -a "$OUT"
done
echo "[$(date '+%F %T')] done -> $OUT"
