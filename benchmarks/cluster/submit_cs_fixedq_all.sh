#!/bin/bash
# Fixed-query case-study dumps for ALL 13 checkpoints, one job each.
#
# submit_cs_dvar_density.sbatch dispatches on HARNESS (dopfn_native, dopfn_bb,
# uwyk1d, graph2d, cpfn1d, cpfn2d_pooled) -- six names, not thirteen models. The
# other seven are the same harnesses at different checkpoints, so each gets its
# own OUT_ROOT: they would otherwise collide, since the dump path is keyed by
# harness name (two cpfn1d checkpoints both write .../cpfn1d/<case>).
#
#   DATA=$SCRATCH/cs_fixedq_root CASE=Observed_Confounder \
#     bash R-PFN/benchmarks/cluster/submit_cs_fixedq_all.sh --submit
#
# Env: DATA (the shift0/d0 root), CASE, CTX, NQ, OUT_PARENT, ACCOUNT, ONLY
set -uo pipefail
SUBMIT=0; [ "${1:-}" = "--submit" ] && SUBMIT=1
KIT="${KIT:-$PWD}"; REPO="${REPO:-$KIT/R-PFN}"
SC="${SCRATCH:?SCRATCH must be set}"
CK="${CK:-$REPO/Required_checkpoints}"
UWYKD="${UWYK_CKPT_DIR:-$KIT/external/uwyk_reproduce/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0}"
DATA="${DATA:-$SC/cs_fixedq_root}"
CASE="${CASE:-Observed_Confounder}"
CTX="${CTX:-1000}"; NQ="${NQ:-1}"
OUT_PARENT="${OUT_PARENT:-$SC/cs_fixedq_all}"
ACCT="${ACCOUNT:-}"; ONLY="${ONLY:-}"
# CPU by default, but the CPU partition is often the slower queue. GRES=gpu:h100:1
# switches to a GPU (nibi demands a TYPE, never a bare gpu:N) and CUDA_VISIBLE_DEVICES
# is then left alone so the harness actually uses it.
GRES="${GRES:-none}"; CPUS="${CPUS:-8}"; MEM="${MEM:-32G}"
# The inner sbatch asks for 1h, which is fine at ~100 draws but not at 1000: the
# uwyk/graph2d harnesses ran ~100 realizations in most of an hour, so they need
# roughly 8-10x that. Overridable rather than raised for everyone, since the fast
# models finish in minutes and a long request only costs them queue priority.
TIME="${TIME:-}"
# DEP=<jobid> holds every dump until that job succeeds, so a fresh replicate root
# and the 13 dumps that read it can be submitted in one go rather than the user
# sitting at a terminal to fire the second half by hand.
DEP="${DEP:-}"
SB="$REPO/benchmarks/cluster/submit_cs_dvar_density.sbatch"

# The model table lives in fq_models.sh, shared with the array driver.
source "$REPO/benchmarks/cluster/fq_models.sh"

N=0
for row in "${ROWS[@]}"; do
    IFS='|' read -r name harness extra <<<"$row"
    [ -n "$ONLY" ] && [ "$name" != "$ONLY" ] && continue
    # A git-lfs POINTER is a few hundred bytes and fails as 'invalid load key';
    # stat -L follows symlinks, whose own size is the target path length.
    for kv in $extra; do
        case "$kv" in *=*.pt)
            f="${kv#*=}"
            sz=$(stat -Lc%s "$f" 2>/dev/null || stat -Lf%z "$f" 2>/dev/null || echo "")
            if [ -z "$sz" ]; then printf '%-22s SKIP: missing %s\n' "$name" "$f"; continue 2; fi
            if [ "$sz" -lt 1000000 ] 2>/dev/null; then
                printf '%-22s SKIP: %s bytes -- git-lfs pointer, not weights\n' "$name" "$sz"; continue 2
            fi ;;
        esac
    done
    N=$((N+1))
    if [ "$SUBMIT" = 1 ]; then
        printf '%-22s harness=%-14s -> ' "$name" "$harness"
        if [ "$GRES" = none ]; then _cvd=(CUDA_VISIBLE_DEVICES=); else _cvd=(); fi
        env $extra "${_cvd[@]}" MODEL_OVERRIDE="$harness" D=0 SHIFT=0 \
            CTX="$CTX" NQ="$NQ" DATA="$DATA" CASES_OVERRIDE="$CASE" \
            OUT_ROOT="$OUT_PARENT/$name" \
            sbatch --array=0 --gres="$GRES" --cpus-per-task="$CPUS" --mem="$MEM" \
                   ${TIME:+--time=$TIME} ${DEP:+--dependency=afterok:$DEP} \
                   ${ACCT:+--account=$ACCT} --job-name="fq-$name" "$SB"
    else
        printf '%-22s harness=%-14s %s\n' "$name" "$harness" "${extra:-<defaults>}"
    fi
done
echo
echo "jobs: $N   gres=$GRES cpus=$CPUS mem=$MEM time=${TIME:-<sbatch default>}${DEP:+ dep=afterok:$DEP}"
echo "out:  $OUT_PARENT/<model>/shift0/d0/ctx$CTX/<harness>/$CASE"
[ "$SUBMIT" = 1 ] || echo "dry run -- add --submit"
