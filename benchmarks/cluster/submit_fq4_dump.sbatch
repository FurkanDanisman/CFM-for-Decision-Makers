#!/bin/bash
#SBATCH --job-name=fq4-dump
#SBATCH --output=logs_fq/fq4dump_%A_%a.out
#SBATCH --error=logs_fq/fq4dump_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#
# Dump one model on one fq4 cell, as an array over d x realization x model for ONE
# (case, shift). 6 d x 10 reals x 13 models = 780 tasks.
#
# An array rather than 780 sbatch calls: the full sweep is 108 cells x 13 models =
# 14,040 units, and submitting those one at a time is scheduler abuse and unkillable
# in practice. One array per (case, shift) keeps every array under the 1000-task
# limit and lets the sweep be submitted, watched and cancelled in 18 pieces.
#
# Task id decodes as  d = DS[id / (REALS*NM)],  real = (id / NM) % REALS,
# model = id % NM -- model varies fastest, so a truncated array still covers whole
# cells rather than leaving every cell missing the same models.
#
# Env: CASE, SHIFT, DS, REALS, QUERIES, CTX, FQ4, OUT4
set -uo pipefail
mkdir -p logs_fq

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
SC="${SCRATCH:?SCRATCH must be set}"
CK="${CK:-$REPO/Required_checkpoints}"
UWYKD="${UWYK_CKPT_DIR:-$DEPLOY_ROOT/external/uwyk_reproduce/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0}"
# shellcheck disable=SC1091
source "$REPO/benchmarks/cluster/fq_models.sh"

CASE="${CASE:-Observed_Confounder}"
SHIFT="${SHIFT:-0}"
CTX="${CTX:-1000}"
read -r -a DSA <<<"${DS:-5 10 20 30 40 50}"
REALS="${REALS:-10}"
QUERIES="${QUERIES:-10}"
FQ4="${FQ4:-$SC/fq4}"
OUT4="${OUT4:-$SC/fq4_dumps}"
NM="${#ROWS[@]}"

ID="${SLURM_ARRAY_TASK_ID:-0}"
MI=$(( ID % NM ))
REAL=$(( (ID / NM) % REALS ))
D="${DSA[$(( ID / (REALS * NM) ))]}"
IFS='|' read -r name harness extra <<<"${ROWS[$MI]}"
TAG="${CASE}_shift${SHIFT}_d${D}_r${REAL}"
ROOT="$FQ4/${TAG}_root"

echo "[$(date '+%F %T')] task $ID -> $TAG model=$name harness=$harness"
[ -d "$ROOT/shift0/d0/$CASE/N$CTX" ] || {
    echo "FATAL: cell not generated: $ROOT" >&2; exit 1; }

# A git-lfs POINTER is a few hundred bytes and dies as 'invalid load key'.
for kv in $extra; do
    case "$kv" in *=*.pt)
        f="${kv#*=}"
        sz=$(stat -Lc%s "$f" 2>/dev/null || echo "")
        [ -n "$sz" ] || { echo "FATAL: missing $f" >&2; exit 1; }
        [ "$sz" -ge 1000000 ] || { echo "FATAL: $f is $sz bytes -- lfs pointer" >&2; exit 1; } ;;
    esac
done

cd "$DEPLOY_ROOT"
# shellcheck disable=SC2086
env $extra CUDA_VISIBLE_DEVICES= MODEL_OVERRIDE="$harness" D=0 SHIFT=0 \
    CTX="$CTX" NQ="$QUERIES" DATA="$ROOT" CASES_OVERRIDE="$CASE" \
    OUT_ROOT="$OUT4/$TAG/$name" SLURM_ARRAY_TASK_ID=0 \
    bash "$REPO/benchmarks/cluster/submit_cs_dvar_density.sbatch"
echo "[$(date '+%F %T')] done $TAG $name"
