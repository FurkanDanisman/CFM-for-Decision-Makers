#!/bin/bash
# Scale-up dumps: one job per model per benchmark. 5 models x 2 = 10 jobs.
#
# Replaces the array-per-dataset layout (which would be 5 + 8 = 13 queue entries
# per model, 65 total) with one entry per model that walks its datasets
# sequentially. Fewer, longer jobs -- and a model's output is complete or not,
# rather than partially present across array tasks that failed independently.
#
#   bash R-PFN/benchmarks/cluster/submit_scale_dumps.sh            # dry run
#   bash R-PFN/benchmarks/cluster/submit_scale_dumps.sh --submit
#   ONLY=cpfn1d_j32 BENCH=rc bash ... --submit                      # narrow
#
# Output lands in $SCRATCH/dumps_all/<model>/<rc|cs>, one root per model, which
# is what the scorer expects: model identity lives in the root, so one METHODS
# row is reused per harness.

set -uo pipefail
SUBMIT=0; [ "${1:-}" = "--submit" ] && SUBMIT=1
KIT="${KIT:-/scratch/furkanbd/rpfn_bench_kit}"
REPO="${REPO:-$KIT/R-PFN}"
CK="${CK:-$REPO/Required_checkpoints}"
SC="${SCRATCH:-/scratch/furkanbd}"
SB="$REPO/benchmarks/cluster/submit_dump_one_model.sbatch"
ONLY="${ONLY:-}"; WANT_BENCH="${BENCH:-both}"
RC_TIME="${RC_TIME:-12:00:00}"; CS_TIME="${CS_TIME:-12:00:00}"

# name | MODEL_IDX | CKPT_ENV | checkpoint
# MODEL_IDX indexes the inner MODELS array:
#   0 dopfn_native  1 dopfn_bb  2 uwyk1d  3 graph2d  4 cpfn1d  5 cpfn2d_pooled
ROWS=(
  "dopfn_repro_1d_J10|0|DOPFN_CKPT|$CK/dopfn_repro_1d_J10_step150000.pt"
  "dopfn_repro_1d_J100|0|DOPFN_CKPT|$CK/dopfn_repro_1d_J100_step150000.pt"
  "dopfn_repro_joint2d|1|CKPT_DOPFN_BB|$CK/dopfn_repro_joint2d_bb.pt"
  "cpfn1d_j32|4|CKPT_CPFN1D|$CK/cpfn1d_j32_step50000.pt"
  "cpfn1d_botharms|4|CKPT_CPFN1D|$CK/cpfn1d_botharms_step50000.pt"
  "cpfn_v0|4|CKPT_CPFN1D|${CKPT_CPFN_V0:-$CK/cpfn_v0_original.pt}"
  "uwyk_bin|2|CKPT|$CK/uwyk_bin_step50000.pt"
)

cd "$KIT" || exit 1
N=0
for r in "${ROWS[@]}"; do
    IFS='|' read -r name idx envv ck <<<"$r"
    [ -n "$ONLY" ] && [ "$ONLY" != "$name" ] && continue
    if [ ! -f "$ck" ]; then printf '%-22s SKIP: no %s\n' "$name" "$ck"; continue; fi
    sz=$(stat -Lc%s "$ck" 2>/dev/null || echo 0)
    if [ "$sz" -lt 1000000 ]; then
        printf '%-22s SKIP: %s bytes -- git-lfs pointer, not weights\n' "$name" "$sz"; continue
    fi
    for b in rc cs; do
        [ "$WANT_BENCH" = both ] || [ "$WANT_BENCH" = "$b" ] || continue
        [ "$b" = rc ] && T="$RC_TIME" || T="$CS_TIME"
        N=$((N+1))
        if [ "$SUBMIT" = 1 ]; then
            printf '%-22s %s  -> ' "$name" "$b"
            UWYK_EXTRA_CONFIG=""; UWYK_EXTRA_ENC=""
            if [ "$name" = uwyk_bin ]; then
                UWYK_EXTRA_CONFIG="${UWYK_BIN_CONFIG:-$CK/uwyk_USED_IN_RESULTS_best_model_config.yaml}"
                UWYK_EXTRA_ENC="${UWYK_T_ENCODING:-binary}"
            fi
            MODEL_NAME="$name" MODEL_IDX="$idx" CKPT_ENV="$envv" CKPT="$ck" \
            BENCH="$b" OUT_ROOT="$SC/dumps_all/$name/$b" SHIFT=0 \
            CONFIG="$UWYK_EXTRA_CONFIG" UWYK_T_ENCODING="$UWYK_EXTRA_ENC" \
                sbatch --time="$T" --job-name="dump-$b-$name" "$SB"
        else
            printf '%-22s %s  (idx %s, %s, %s)\n' "$name" "$b" "$idx" "$T" "$(basename "$ck")"
        fi
    done
done
echo
echo "jobs: $N"
[ "$SUBMIT" = 1 ] || echo "dry run -- add --submit to queue them"
