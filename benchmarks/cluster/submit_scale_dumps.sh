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
SKIP="${SKIP:-}"
# Case studies are the pooling of shifts 0, +2 and -2, and the sbatch handles ONE
# shift per job -- so a model needs three case-study jobs, not one. 24 cells in
# the reference tree is 3 shifts x 8 d values, which is what this reproduces.
SHIFTS="${SHIFTS:-0 +2 -2}"
# Sized from the smoke runs, not guessed: RealCause was 1m20s for one dataset
# (5 -> well under an hour) and a case-study cell of 6 cases was ~6 min (8 d ->
# ~1h). A 12h request queues far worse than a 3h one for the same work.
# Killarney has NO cpu partition -- every partition is GPU -- and they are TIERED
# BY WALLTIME, with shorter requests reaching far more nodes:
#   b1  3h   168 l40s      b2  12h  126      b3  1d  84      b4  3d  42
# So staying under 3h is the scheduling lever, not the partition name. Measured
# from the smoke runs: one RealCause dataset 1m20s (five ~ 10 min) and one
# case-study cell of 6 cases ~6 min (eight d ~ 50 min). Both fit 3h with room.
RC_TIME="${RC_TIME:-3:00:00}"; CS_TIME="${CS_TIME:-3:00:00}"

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
    case " $SKIP " in *" $name "*) printf '%-22s SKIPPED (SKIP=)\n' "$name"; continue ;; esac
    if [ ! -f "$ck" ]; then printf '%-22s SKIP: no %s\n' "$name" "$ck"; continue; fi
    sz=$(stat -Lc%s "$ck" 2>/dev/null || echo 0)
    if [ "$sz" -lt 1000000 ]; then
        printf '%-22s SKIP: %s bytes -- git-lfs pointer, not weights\n' "$name" "$sz"; continue
    fi
    UWYK_EXTRA_CONFIG=""; UWYK_EXTRA_ENC=""
    if [ "$name" = uwyk_bin ]; then
        UWYK_EXTRA_CONFIG="${UWYK_BIN_CONFIG:-$CK/uwyk_USED_IN_RESULTS_best_model_config.yaml}"
        UWYK_EXTRA_ENC="${UWYK_T_ENCODING:-binary}"
    fi
    for b in rc cs; do
        [ "$WANT_BENCH" = both ] || [ "$WANT_BENCH" = "$b" ] || continue
        if [ "$b" = rc ]; then T="$RC_TIME"; shift_list="-"; else T="$CS_TIME"; shift_list="$SHIFTS"; fi
        for sh in $shift_list; do
            N=$((N+1))
            tag="$b"; [ "$sh" != "-" ] && tag="$b$sh"
            if [ "$SUBMIT" = 1 ]; then
                printf '%-22s %-6s -> ' "$name" "$tag"
                MODEL_NAME="$name" MODEL_IDX="$idx" CKPT_ENV="$envv" CKPT="$ck" \
                BENCH="$b" OUT_ROOT="$SC/dumps_all/$name/$b" \
                SHIFT="$([ "$sh" = "-" ] && echo 0 || echo "$sh")" \
                CONFIG="$UWYK_EXTRA_CONFIG" UWYK_T_ENCODING="$UWYK_EXTRA_ENC" \
                    sbatch --time="$T" --job-name="dump-$tag-$name" "$SB"
            else
                printf '%-22s %-6s (idx %s, %s, %s)\n' "$name" "$tag" "$idx" "$T" "$(basename "$ck")"
            fi
        done
    done
done
echo
echo "jobs: $N"
[ "$SUBMIT" = 1 ] || echo "dry run -- add --submit to queue them"
