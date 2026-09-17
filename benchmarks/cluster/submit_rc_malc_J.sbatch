#!/bin/bash
#SBATCH --job-name=rc-malc-J
#SBATCH --output=logs_rc_malc_J/run_%j.out
#SBATCH --error=logs_rc_malc_J/run_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#
# CPU ONLY. Variant J of the MALC calibration study: smooth the JOINT
# p(y0, y1), THEN diagonal-integrate to tau. The complement of variant T
# (submit_rc_malc_variants.sbatch), which forms tau first and smooths that.
#
# The two are not two approximations of one thing. J regularises the
# dependence structure as well as the shape, because a log-concave fit on the
# plane constrains how the arms co-vary; T leaves the joint alone and only
# regularises the shape of the tau density it produced. Where they disagree,
# the disagreement is attributable to the joint.
#
# Runs the EXISTING per-cell scripts unchanged:
#   2D methods -> realcause_eval/compute_malc_ci_cell.py
#   1D methods -> realcause_eval/compute_malc_ci_cell_1d.py
#                 (builds the independence joint p_y0 (x) p_y1 first, then
#                  hands off to the same MALC path)
# then aggregates with cate_ci_from_density_malc.py.
#
# N_EVAL=2001 IS DELIBERATE. The fitted joint is rendered on an
# n_eval x n_eval grid and diagonal-integrated, giving 2*n_eval-1 tau atoms.
# 2001 -> exactly 4001, matching variant T's tau grid with no resampling.
# n_eval=4000 would give 7999 atoms, thrown away down to 4001, at 4x the cost
# (the grid is quadratic in n_eval: 4.0M points per query at 2001, 16.0M at
# 4000).
#
# START WITH MAX_REAL SET. At n_eval=2001 each query renders a 4-million-point
# grid; this is far more expensive than the n_eval=50 these scripts default to.
# Confirm the wall-clock on 3 realizations before committing to 100.
#
# Env:
#   ROOT       density-dump root        (default $SCRATCH/rc_dens_uni)
#   DATASET    (default IHDP)
#   N_EVAL     (default 2001)
#   MALC_B     (default 100)
#   MALC_K     (default 1)
#   MAX_REAL   cap realizations per cell -- set this for the smoke
#   OUT_TAG    suffix on the per-realization files (default B<B>_K<K>_E<n_eval>)
#   METHODS_2D / METHODS_1D   override the cell lists
#   NW         workers (default = cpus-per-task)
#
#   sbatch benchmarks/cluster/submit_rc_malc_J.sbatch

set -uo pipefail
mkdir -p logs_rc_malc_J

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
N_EVAL="${N_EVAL:-2001}"
MALC_B="${MALC_B:-100}"
MALC_K="${MALC_K:-1}"
NW="${NW:-${SLURM_CPUS_PER_TASK:-32}}"
OUT_TAG="${OUT_TAG:-B${MALC_B}_K${MALC_K}_E${N_EVAL}}"

# dopfn_bb writes density_r<###>.npz + summary.npz (split layout) rather than
# one inline npz per realization; the 2D script needs --is-split told to it.
METHODS_2D="${METHODS_2D:-cpfn2d graph2d dopfn_bb}"
METHODS_1D="${METHODS_1D:-cpfn1d uwyk1d dopfn_native}"

[ -d "$ROOT" ] || { echo "FATAL: dump root not found: $ROOT" >&2; exit 1; }

source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/realcause_eval:$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

MR=(); [ -n "${MAX_REAL:-}" ] && MR=(--max-realizations "$MAX_REAL")
FAILED=0

echo "[rc-malc-J] root=$ROOT dataset=$DATASET n_eval=$N_EVAL B=$MALC_B K=$MALC_K"
echo "            tau atoms from diagonals = $((2*N_EVAL-1))   workers=$NW"
echo "            out_tag=$OUT_TAG  max_real=${MAX_REAL:-<all>}"

_resolve() {   # method name -> existing cell dir, honouring the cpfn2d alias
    for cand in "$1" "${1}_pooled"; do
        [ -d "$ROOT/$cand/$DATASET" ] && { echo "$ROOT/$cand"; return 0; }
    done
    return 1
}

for M in $METHODS_2D; do
    D="$(_resolve "$M")" || { echo "[skip] $M: no $ROOT/$M/$DATASET"; continue; }
    SPLIT=(); [ "$M" = "dopfn_bb" ] && SPLIT=(--is-split)
    echo "=== J / 2D  $M  ($D) ==="
    python -u "$REPO/realcause_eval/compute_malc_ci_cell.py" \
        --method-dir "$D" --dataset "$DATASET" --repo "$REPO" \
        --n-workers "$NW" --malc-K "$MALC_K" --malc-B "$MALC_B" \
        --n-eval "$N_EVAL" --out-tag "$OUT_TAG" "${SPLIT[@]}" "${MR[@]}" \
      || { echo "ERROR: J/2D $M FAILED" >&2; FAILED=1; }
done

for M in $METHODS_1D; do
    D="$(_resolve "$M")" || { echo "[skip] $M: no $ROOT/$M/$DATASET"; continue; }
    echo "=== J / 1D  $M  ($D) ==="
    python -u "$REPO/realcause_eval/compute_malc_ci_cell_1d.py" \
        --method-dir "$D" --dataset "$DATASET" --repo "$REPO" \
        --n-workers "$NW" --malc-K "$MALC_K" --malc-B "$MALC_B" \
        --n-eval "$N_EVAL" --out-tag "$OUT_TAG" "${MR[@]}" \
      || { echo "ERROR: J/1D $M FAILED" >&2; FAILED=1; }
done

echo "=== aggregate ==="
python -u "$REPO/realcause_eval/cate_ci_from_density_malc.py" \
    --out-root "$ROOT" --datasets "$DATASET" --in-tag "$OUT_TAG" \
    --out-md "$ROOT/calib_${DATASET}_J_${OUT_TAG}.md" \
  || { echo "ERROR: aggregation FAILED" >&2; FAILED=1; }

echo "[$(date '+%F %T')] done (FAILED=$FAILED)"
ls -1 "$ROOT"/calib_"$DATASET"_J_*.md 2>/dev/null || true
