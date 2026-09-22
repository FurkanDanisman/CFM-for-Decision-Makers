#!/bin/bash
#SBATCH --job-name=full-report
#SBATCH --output=logs_full_report/fr_%j.out
#SBATCH --error=logs_full_report/fr_%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#
# The WHOLE report chain in one job: ComplexMech point (both versions),
# case-study point, the two-version ComplexMech table, and the assembled report.
# Rescores existing dumps -- no inference, no GPU.
#
#   sbatch --account=def-zhijing R-PFN/benchmarks/cluster/submit_full_report.sbatch
#
# Each stage is skipped if its output already exists and is non-empty, so a
# walltime kill costs only the stage in flight. FORCE=1 redoes everything.
#
# 2h is measured, not padded: cs_point_total ran in 20:47 as job 22444356 on a
# compute node (the ~2h figure was a contended LOGIN node), and the two
# ComplexMech point stages are minutes each. Resume makes a short walltime the
# right trade -- a timeout costs one stage, a long request costs queue priority.

set -uo pipefail
mkdir -p logs_full_report
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

source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$B:$B/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

# A markdown file with no data row is not done -- an empty table would otherwise
# let a failed stage masquerade as a cached one.
done_already() {
    [ "${FORCE:-0}" = 1 ] && return 1
    [ -s "$1" ] || return 1
    grep -qE '^\| *[A-Za-z0-9_-]+ *\|' "$1"
}
run() {  # run <output> <label> <cmd...>
    local out="$1" label="$2"; shift 2
    if done_already "$out"; then echo "==== skip (done): $label"; return 0; fi
    echo "==== [$(date '+%F %T')] $label"
    "$@" || { echo "    FAILED: $label" >&2; return 1; }
}

FAILED=0
run "$SCRATCH/cmech_point_total_main.md" "ComplexMech point (main)" \
    python -u "$B/cmech_point_total.py" --dumps "$SCRATCH/cmech_dumps" \
    --data "$SCRATCH/cmech_data_v2" --out "$SCRATCH/cmech_point_total_main.md" \
    || FAILED=$((FAILED+1))
run "$SCRATCH/cmech_point_total_rho99.md" "ComplexMech point (rho>0.99)" \
    python -u "$B/cmech_point_total.py" --dumps "$SCRATCH/cmech_dumps_rho99" \
    --data "$SCRATCH/cmech_data_rho99" --out "$SCRATCH/cmech_point_total_rho99.md" \
    || FAILED=$((FAILED+1))
run "$SCRATCH/cs_point_total.md" "case-study point (slow stage)" \
    python -u "$B/cs_point_total.py" --out "$SCRATCH/cs_point_total.md" \
    || FAILED=$((FAILED+1))

# The last two are seconds and depend on the three above, so they always re-run.
echo "==== [$(date '+%F %T')] ComplexMech both versions"
python -u "$B/cmech_combined_table.py" \
    --point-main  "$SCRATCH/cmech_point_total_main.md" \
    --point-rho99 "$SCRATCH/cmech_point_total_rho99.md" \
    --table-main  "$SCRATCH/final_table.md" \
    --table-rho99 "$SCRATCH/final_table_rho99.md" \
    --out "$SCRATCH/cmech_both.md" >/dev/null || FAILED=$((FAILED+1))

echo "==== [$(date '+%F %T')] assembling FULL_REPORT.md"
python -u "$B/master_report.py" \
    --final-table "$SCRATCH/final_table.md" \
    --cs-point    "$SCRATCH/cs_point_total.md" \
    --cmech-both  "$SCRATCH/cmech_both.md" \
    --out         "$SCRATCH/FULL_REPORT.md" || FAILED=$((FAILED+1))

echo
echo "[$(date '+%F %T')] done  failed_stages=$FAILED"
echo "  $SCRATCH/FULL_REPORT.md"
grep -c '^|' "$SCRATCH/FULL_REPORT.md" 2>/dev/null | xargs -I{} echo "  table rows: {}"
exit $FAILED
