#!/bin/bash
# The three Aurora checkpoints (label-matched Do-PFN 1D, label-matched UWYK 1D,
# lower-resolution UWYK 1D) across all three benchmarks, point estimate AND
# coverage in one pass.
#
# Point estimate and coverage come from the SAME dump: DENSITY_DUMP=1 makes each
# harness write its predictive density beside its point estimate, so PEHE and the
# intervals always describe the same estimator. Scoring splits them afterwards.
#
#   bash R-PFN/benchmarks/cluster/submit_aurora3_all.sh            # dry run
#   bash R-PFN/benchmarks/cluster/submit_aurora3_all.sh --submit
#   ONLY=uwyk_D_j32_nobin BENCH=cs bash ... --submit               # one slice
#
# Env: AUR, OUT_PARENT, DATA_CS, CMECH_DATA, CTX, ACCOUNT, ONLY, BENCH
set -uo pipefail
SUBMIT=0; [ "${1:-}" = "--submit" ] && SUBMIT=1
KIT="${KIT:-$PWD}"; REPO="${REPO:-$KIT/R-PFN}"
SC="${SCRATCH:?SCRATCH must be set}"
AUR="${AUR:-$KIT/from_aurora}"
OUT_PARENT="${OUT_PARENT:-$SC/aurora3}"
CTX="${CTX:-1000}"
ACCT="${ACCOUNT:-}"; ONLY="${ONLY:-}"; BENCH="${BENCH:-all}"
# Case-study dvar root: $DATA_CS/shift<S>/d<D>/<case>/N<ctx>.
DATA_CS="${DATA_CS:-$SC/cs_dvar_data}"
# ComplexMech rho>0.99 root, as written by submit_cmech_rho99_gen.sbatch.
CMECH_DATA="${CMECH_DATA:-$SC/cmech_data_rho99}"

# uwyk1d.py:134 falls back to <ckpt_dir>/best_model_config.yaml and :135 checks
# the file EXISTS, so a config must sit beside the .pt. The J=1000 config is the
# only one we have; if uwyk_D (J=32) disagrees it fails on a state_dict shape
# mismatch, which is loud, not silent.
UWYK_CFG="$AUR/best_model_config.yaml"

# name | harness | harness index | extra env
ROWS=(
  "dopfn_1d_botharms|dopfn_native|0|DOPFN_CKPT=$AUR/dopfn_1d_botharms_step150000.pt"
  "uwyk_C_botharms|uwyk1d|2|CKPT=$AUR/uwyk_C_botharms_step50000.pt CONFIG=$UWYK_CFG UWYK_T_ENCODING=binary"
  "uwyk_D_j32_nobin|uwyk1d|2|CKPT=$AUR/uwyk_D_j32_nobin_step50000.pt CONFIG=$UWYK_CFG UWYK_T_ENCODING=target"
)

problems=0
say() { printf '%s\n' "$*"; }
chk() { [ -e "$1" ] || { say "  MISSING: $1"; problems=$((problems+1)); }; }

say "=== preflight ==="
for row in "${ROWS[@]}"; do
    IFS='|' read -r name harness hidx extra <<<"$row"
    for kv in $extra; do case "$kv" in *=*.pt|*=*.yaml) chk "${kv#*=}";; esac; done
done
chk "$DATA_CS"
chk "$CMECH_DATA"
if [ ! -e "$UWYK_CFG" ]; then
    say "  -> no config beside the .pt files. Copy the J=1000 one:"
    say "     cp \$REPO/Required_checkpoints/uwyk_USED_IN_RESULTS_best_model_config.yaml $UWYK_CFG"
fi
if [ "$problems" -gt 0 ]; then
    say ""
    say "$problems path(s) missing. Candidates on disk:"
    ls -d "$SC"/*cs_dvar* "$SC"/*shift* "$SC"/cmech_data* 2>/dev/null | sed 's/^/     /'
    say "Set DATA_CS=... CMECH_DATA=... and re-run."
fi
say ""

N_SB=0; N_TASK=0
go() {   # go <jobname> <array> <sbatch> <env...>
    local jn="$1" arr="$2" sb="$3"; shift 3
    local lo=${arr%-*} hi=${arr#*-}; local n=$(( hi - lo + 1 ))
    N_SB=$((N_SB+1)); N_TASK=$((N_TASK+n))
    if [ "$SUBMIT" = 1 ]; then
        printf '  %-42s array=%-8s tasks=%-3s -> ' "$jn" "$arr" "$n"
        env "$@" sbatch --array="$arr" ${ACCT:+--account=$ACCT} \
            --job-name="$jn" "$REPO/benchmarks/cluster/$sb"
    else
        printf '  %-42s array=%-8s tasks=%-3s %s\n' "$jn" "$arr" "$n" "$sb"
    fi
}

for row in "${ROWS[@]}"; do
    IFS='|' read -r name harness hidx extra <<<"$row"
    [ -n "$ONLY" ] && [ "$name" != "$ONLY" ] && continue
    say "--- $name  (harness=$harness) ---"

    # RealCause: MODELS[ID/5], SETS[ID%5]. No MODEL_OVERRIDE, so the harness is
    # selected by array range instead.
    if [ "$BENCH" = all ] || [ "$BENCH" = rc ]; then
        go "a3-rc-$name" "$(( hidx*5 ))-$(( hidx*5+4 ))" submit_realcause_density_unified.sbatch \
           $extra CUDA_VISIBLE_DEVICES= DENSITY_DUMP=1 \
           OUT_ROOT="$OUT_PARENT/rc/$name"
    fi

    # Case study: MODEL_OVERRIDE is supported; SHIFT is an env, not an array
    # dimension, so the three pooled shifts are three submissions.
    if [ "$BENCH" = all ] || [ "$BENCH" = cs ]; then
        for SH in 0 +2 -2; do
            go "a3-cs$SH-$name" "0-7" submit_cs_dvar_density.sbatch \
               $extra CUDA_VISIBLE_DEVICES= DENSITY_DUMP=1 \
               MODEL_OVERRIDE="$harness" SHIFT="$SH" CTX="$CTX" DATA="$DATA_CS" \
               ANC_MODE=case_family OUT_ROOT="$OUT_PARENT/cs/$name"
        done
    fi

    # ComplexMech rho>0.99: PER_MODEL=30 (NODES x CONTEXTS). --subset total at
    # scoring needs BOTH nonzero and zero dumped, so two submissions.
    if [ "$BENCH" = all ] || [ "$BENCH" = cm ]; then
        for SS in nonzero zero; do
            go "a3-cm-$SS-$name" "$(( hidx*30 ))-$(( hidx*30+29 ))" submit_cmech_pehe_1d_vs_2d.sbatch \
               $extra CUDA_VISIBLE_DEVICES= DENSITY_DUMP=1 \
               SUBSET="$SS" UWYK_FIG34_DATA="$CMECH_DATA" \
               OUT_ROOT="$OUT_PARENT/cm_rho99/$name"
        done
    fi
    say ""
done

say "=== $N_SB sbatch submissions, $N_TASK array tasks total (all CPU) ==="
say "out: $OUT_PARENT/{rc,cs,cm_rho99}/<model>/"
[ "$SUBMIT" = 1 ] || say "DRY RUN -- add --submit"
