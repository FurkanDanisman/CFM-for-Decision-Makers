#!/bin/bash
#SBATCH --job-name=figures
#SBATCH --output=logs_figures/fig_%j.out
#SBATCH --error=logs_figures/fig_%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#
# The by-d figures for both benchmarks, PEHE and relative ATE error:
#   cmech_by_d_pehe.png      cmech_by_d_eps_ate.png        (nodes x context)
#   fig3_cen3_by_d_pehe.png  fig3_cen3_by_d_eps_ate.png    (case x d)
#
# Rescores existing dumps -- no inference, no GPU. Never run cs_point_total on a
# login node: --workers defaults to $SLURM_CPUS_PER_TASK, which is unset there,
# so it silently drops to ONE worker for 972 (cell, case) units.
#
#   sbatch --account=def-zhijing R-PFN/benchmarks/cluster/submit_figures.sbatch
#
# Optional: CMECH_DUMPS, CMECH_DATA, CM_NODES, CM_CONTEXTS, FIGDIR,
#           SKIP_CASE=1, SKIP_CMECH=1, FORCE=1

set -uo pipefail
mkdir -p logs_figures
DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
SCRATCH="${SCRATCH:-/scratch/$USER}"
B="$REPO/benchmarks"
DUMPS="${CMECH_DUMPS:-$SCRATCH/cmech_dumps}"
DATA="${CMECH_DATA:-$SCRATCH/cmech_data_v2}"
NODES="${CM_NODES:-5 10 20 30 40 50}"
CONTEXTS="${CM_CONTEXTS:-50 100 250 500 1000}"
FIGDIR="${FIGDIR:-$SCRATCH/figs}"
NW="${SLURM_CPUS_PER_TASK:-32}"

source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$B:$B/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$FIGDIR"

done_already() { [ "${FORCE:-0}" = 1 ] && return 1; [ -s "$1" ]; }
FAILED=0

if [ "${SKIP_CASE:-0}" != 1 ]; then
  if done_already "$SCRATCH/cs_by_d.csv"; then
      echo "==== skip (done): case-study per-d scoring"
  else
      echo "==== [$(date '+%F %T')] case-study per-d scoring  (workers=$NW)"
      python -u "$B/cs_point_total.py" --by-d --workers "$NW" \
          --csv "$SCRATCH/cs_by_d.csv" --out "$SCRATCH/cs_by_d.md" >/dev/null \
          || { echo "    FAILED"; FAILED=$((FAILED+1)); }
  fi
  echo "==== [$(date '+%F %T')] case-study figures"
  python -u "$B/plot_pehe_grid.py" --csv "$SCRATCH/cs_by_d.csv" --kind case \
      --metric pehe eps_ate --out "$FIGDIR/fig3_cen3_by_d" \
      || FAILED=$((FAILED+1))
fi

if [ "${SKIP_CMECH:-0}" != 1 ]; then
  if done_already "$SCRATCH/cmech_by_d.csv"; then
      echo "==== skip (done): ComplexMech per-context scoring"
  else
      echo "==== [$(date '+%F %T')] ComplexMech per-context scoring  [$CONTEXTS]"
      python -u "$B/cmech_point_total.py" --dumps "$DUMPS" --data "$DATA" \
          --nodes $NODES --contexts $CONTEXTS \
          --csv "$SCRATCH/cmech_by_d.csv" --out "$SCRATCH/cmech_by_d.md" \
          >/dev/null 2>"$SCRATCH/.cmech_by_d.err" \
          || { echo "    FAILED (see $SCRATCH/.cmech_by_d.err)"; FAILED=$((FAILED+1)); }
  fi
  echo "==== [$(date '+%F %T')] ComplexMech figures"
  python -u "$B/plot_pehe_grid.py" --csv "$SCRATCH/cmech_by_d.csv" --kind cmech \
      --metric pehe eps_ate --out "$FIGDIR/cmech_by_d" \
      || FAILED=$((FAILED+1))
fi

echo
echo "[$(date '+%F %T')] done  failures=$FAILED"
ls -la "$FIGDIR"/*.png 2>/dev/null
exit $FAILED
