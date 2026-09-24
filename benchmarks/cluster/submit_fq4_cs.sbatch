#!/bin/bash
#SBATCH --job-name=fq4-cs-gen
#SBATCH --output=logs_fq/fq4_cs_%j.out
#SBATCH --error=logs_fq/fq4_cs_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#
# Four-column fixed-query coverage, ONE case-study cell: generate the resampled
# datasets, then submit the 13 model dumps that read them.
#
# THE DESIGN. Pick one source realization. Freeze QUERIES rows of it chosen at
# random, and resample the rest of the observational sample DRAWS times. Every
# frozen row keeps its identity -- same covariates, same treatment, same tau -- in
# all DRAWS datasets, so each is a fixed estimand and the fraction of intervals
# covering it is frequentist coverage, not an average over queries and DGPs.
#
# The rows are chosen at RANDOM rather than taken as rows 0..9: consecutive rows of
# a realization are not a sample of its queries, just whatever the generator
# emitted first, and their taus can be atypically clustered.
#
# All QUERIES share the DRAWS datasets -- freezing 10 rows in one replicate gives
# 10 fixed estimands at once -- so this costs DRAWS datasets per realization, not
# QUERIES x DRAWS.
#
#   bash R-PFN/benchmarks/cluster/fq_submit.sh fq4cs      # smoke: 2 draws
# NOT plain sbatch: logs_fq must exist before Slurm opens --output.
#
# Env: CASE, SRC_D, SHIFT, REAL, QUERIES, DRAWS, QSEED, FQ4, OUT4,
#      ACCOUNT, TIME_DUMP, GRES, CTX

set -uo pipefail
mkdir -p logs_fq

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }
# shellcheck disable=SC1091
source "$DEPLOY_ROOT/venv/bin/activate"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
SC="${SCRATCH:?SCRATCH must be set}"

CASE="${CASE:-Observed_Confounder}"
CTX="${CTX:-1000}"
SRC_D="${SRC_D:-5}"
SHIFT="${SHIFT:-0}"
REAL="${REAL:-0}"
QUERIES="${QUERIES:-10}"
DRAWS="${DRAWS:-100}"
QSEED="${QSEED:-12345}"
FQ4="${FQ4:-$SC/fq4}"
OUT4="${OUT4:-$SC/fq4_dumps}"
TIME_DUMP="${TIME_DUMP:-8:00:00}"
GRES="${GRES:-none}"
ACCT="${ACCOUNT:-}"

TAG="${CASE}_shift${SHIFT}_d${SRC_D}_r${REAL}"
SRC="$DEPLOY_ROOT/case_study_data/d_variation/shift$SHIFT/d$SRC_D/$CASE/N$CTX/${CASE}_${REAL}.npz"
GEN_OUT="$FQ4/$TAG"
ROOT="$FQ4/${TAG}_root"
CELL="$GEN_OUT/$CASE/N$CTX"
DUMPS="$OUT4/$TAG"

echo "[$(date '+%F %T')] fq4 case study: $TAG"
echo "  source : $SRC"
echo "  freeze : $QUERIES random rows (seed $QSEED), $DRAWS resampled datasets"
echo "  dumps  : $DUMPS"
[ -f "$SRC" ] || { echo "FATAL: source npz not found: $SRC" >&2; exit 1; }

# ---- 1. generate -------------------------------------------------------------
HAVE=$(find "$CELL" -name "${CASE}_*.npz" 2>/dev/null | wc -l)
if [ "$HAVE" -ge "$DRAWS" ]; then
    echo "  generation: SKIP ($HAVE >= $DRAWS already in $CELL)"
else
    python "$REPO/case_study/gen_fixed_query.py" \
        --from-npz "$SRC" --d "$SRC_D" \
        --random-queries "$QUERIES" --query-seed "$QSEED" \
        --n-draws "$DRAWS" --out "$GEN_OUT" || {
            echo "FATAL: generation failed (or the fixed-estimand assert tripped)" >&2
            exit 1; }
fi

# ---- 2. the shift0/d0 view the dump harness expects --------------------------
# The generator writes <out>/<case>/N<ctx>; the harness reads
# <root>/shift<S>/d<K>/<case>/N<ctx>. Always shift0/d0 here regardless of where the
# SOURCE came from -- this is a standalone cell, not a member of the d sweep.
mkdir -p "$ROOT/shift0/d0"
ln -sfn "$GEN_OUT/$CASE" "$ROOT/shift0/d0/$CASE"
[ -d "$ROOT/shift0/d0/$CASE/N$CTX" ] || {
    echo "FATAL: symlink did not resolve to $CELL" >&2; exit 1; }

# ---- 3. the estimand really is fixed ----------------------------------------
python - "$CELL" "$QUERIES" <<'PYVER' || exit 1
import glob, os, sys
import numpy as np
cell, K = sys.argv[1], int(sys.argv[2])
fs = sorted(f for f in glob.glob(os.path.join(cell, "*.npz"))
            if os.path.basename(f) != "summary.npz")
t = np.stack([np.asarray(np.load(f, allow_pickle=True)["cate"]).ravel()[:K]
              for f in fs])
spread = float(np.abs(t - t[0]).max())
print(f"  {t.shape[0]} replicates x {t.shape[1]} frozen queries")
print(f"  true tau: min {t[0].min():+.4f}  max {t[0].max():+.4f}  sd {t[0].std():.4f}")
print(f"  spread across replicates = {spread:.3e}")
if spread >= 1e-6:
    print("FATAL: tau moves across replicates -- the estimand is not fixed.",
          file=sys.stderr)
    raise SystemExit(1)
if t.shape[1] > 1 and float(t[0].std()) < 1e-9:
    print("FATAL: all frozen queries share one tau -- coverage would be 0 or 1 by\n"
          "       construction. Pick a different realization or --query-seed.",
          file=sys.stderr)
    raise SystemExit(1)
print("  ESTIMAND FIXED -- submitting dumps")
PYVER

# ---- 4. the 13 dumps ---------------------------------------------------------
# NQ=QUERIES: the generator moves the frozen rows to positions 0..K-1, and the
# loader takes X_test = X[:n_q], so this evaluates exactly them.
cd "$(dirname "$REPO")"
# ACCOUNT is exported rather than written as ${ACCT:+ACCOUNT=...} in the assignment
# prefix: bash only treats an assignment as an assignment when it is LITERAL at
# parse time, so from an expansion "ACCOUNT=def-rgrosse" becomes the command name
# and the submit dies with "command not found".
[ -n "$ACCT" ] && export ACCOUNT="$ACCT"
DATA="$ROOT" CASE="$CASE" CTX="$CTX" NQ="$QUERIES" OUT_PARENT="$DUMPS" \
    TIME="$TIME_DUMP" GRES="$GRES" \
    bash "$REPO/benchmarks/cluster/submit_cs_fixedq_all.sh" --submit
echo "[$(date '+%F %T')] $TAG submitted"
