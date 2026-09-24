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
ACCT="${ACCOUNT:-def-rgrosse}"; ONLY="${ONLY:-}"; BENCH="${BENCH:-all}"
# Case-study dvar root: $DATA_CS/shift<S>/d<D>/<case>/N<ctx>.
#
# Do NOT walk $SCRATCH looking for this. `find -maxdepth 7` over a scratch
# filesystem holding this many dump trees does not finish -- the script appears
# to hang before printing anything. Check known locations directly; if a search
# is needed, bound it to the repo and stop at the first hit with -print -quit.
if [ -z "${DATA_CS:-}" ]; then
    for _c in "$KIT/case_study_data/d_variation" "$SC/case_study_data/d_variation" \
              "$REPO/case_study/d_variation" "$KIT/case_study/d_variation" \
              "$SC/cs_dvar_data"; do
        if [ -d "$_c/shift0" ]; then DATA_CS="$_c"; break; fi
    done
fi
if [ -z "${DATA_CS:-}" ]; then
    _hit=$(find "$REPO" -maxdepth 5 -type d \
             -path '*/shift0/d*/Observed_Confounder' -print -quit 2>/dev/null)
    [ -n "$_hit" ] && DATA_CS=$(dirname "$(dirname "$(dirname "$_hit")")")
fi
if [ -z "${DATA_CS:-}" ]; then
    DATA_CS="$SC/cs_dvar_data"
    say_cs="  DATA_CS NOT FOUND -- pass DATA_CS=<root holding shift0/d2/...>"
else
    say_cs="  DATA_CS=$DATA_CS"
fi
# All three dump sbatches hardcode '#SBATCH --gres=gpu:1', which nibi rejects
# ("submitted a GPU job without specifying a GPU type"). CUDA_VISIBLE_DEVICES=
# does not help: the directive is inside the file, so the gres must be overridden
# on the sbatch COMMAND LINE.
#
# Default is CPU. The sbatch files ask for 3 h, which was sized for the GPU runs;
# on CPU uwyk1d measures ~320 s/dataset, so one RealCause task (100 realizations)
# is hours, not minutes. Hence TIME=24:00:00 and 16 cores by default -- a job
# that dies at the 3 h wall leaves a half-written dump tree behind.
#
# On killarney instead: GRES=gpu:l40s:1 CPUS=4 MEM=32G TIME=03:00:00.
# Slurm runs a whole array at once unless told otherwise. PAR caps how many
# tasks of EACH array are in flight via the %n suffix, so total concurrency is
# (number of submissions) x PAR. At the default PAR=1 that is 18 running tasks,
# with the remaining 249 queued behind them.
PAR="${PAR:-1}"
# MAXCONC caps (submissions x PAR). Slurm has no per-user running cap we can set
# from sbatch, so the only lever is how many arrays we submit and each array's %n.
# Refuse rather than silently exceed it.
MAXCONC="${MAXCONC:-0}"
GRES="${GRES:-none}"
CPUS="${CPUS:-16}"; MEM="${MEM:-64G}"; TIME="${TIME:-24:00:00}"
mkdir -p logs_a3
# ComplexMech rho>0.99 root, as written by submit_cmech_rho99_gen.sbatch.
CMECH_DATA="${CMECH_DATA:-$SC/cmech_data_rho99}"

# uwyk1d.py:134 falls back to <ckpt_dir>/best_model_config.yaml and :135 checks
# the file EXISTS, so a config must sit beside the .pt. The J=1000 config is the
# only one we have; if uwyk_D (J=32) disagrees it fails on a state_dict shape
# mismatch, which is loud, not silent.
UWYK_CFG="$AUR/best_model_config.yaml"
# uwyk_D is J=32, so it needs its OWN config: the loader reads model_cfg.num_bars
# and builds output_dim = num_bars + 4 (GraphConditionedInterventionalPFN_sklearn.py
# :278-279). Against the J=1000 config it built regression_head [1004, 256] while
# the checkpoint carries [36, 256] and load_state_dict(strict=True) aborted.
# Create it with:
#   sed 's/num_bars: *[0-9]*/num_bars: 32/' $AUR/best_model_config.yaml \
#     > $AUR/uwyk_D_j32_config.yaml
UWYK_CFG_J32="$AUR/uwyk_D_j32_config.yaml"

# UWYK_ANC_MODE=v3a_only emits exactly v3a + noanc. The sbatch default is
# v3ab_only, which adds v3b: a third forward pass per realization and a column
# that is not reported. Case study is untouched -- it must stay case_family,
# since v3a/v3b hardcode an adjacency that does not apply to the case families.
# name | harness | harness index | extra env
ROWS=(
  "dopfn_1d_botharms|dopfn_native|0|DOPFN_CKPT=$AUR/dopfn_1d_botharms_step150000.pt"
  "uwyk_C_botharms|uwyk1d|2|CKPT=$AUR/uwyk_C_botharms_step50000.pt CONFIG=$UWYK_CFG UWYK_T_ENCODING=binary UWYK_ANC_MODE=v3a_only"
  "uwyk_D_j32_nobin|uwyk1d|2|CKPT=$AUR/uwyk_D_j32_nobin_step50000.pt CONFIG=$UWYK_CFG_J32 UWYK_T_ENCODING=target UWYK_ANC_MODE=v3a_only"
)

if [ "$GRES" = none ]; then CVD="CUDA_VISIBLE_DEVICES="; else CVD="DUMMY_UNUSED=1"; fi

problems=0
say() { printf '%s\n' "$*"; }
chk() { [ -e "$1" ] || { say "  MISSING: $1"; problems=$((problems+1)); }; }

say "=== preflight ==="
[ -n "${say_cs:-}" ] && say "$say_cs"
for row in "${ROWS[@]}"; do
    IFS='|' read -r name harness hidx extra <<<"$row"
    for kv in $extra; do case "$kv" in *=*.pt|*=*.yaml) chk "${kv#*=}";; esac; done
done
chk "$DATA_CS"
chk "$CMECH_DATA"
if [ -e "$UWYK_CFG_J32" ]; then
    # PARSE it, do not grep it. A sed that leaves "num_bars: 32# comment" greps
    # fine but is invalid YAML: '#' only opens a comment after whitespace, so the
    # value swallows the comment and the file dies several lines later.
    _nb=$(python -c "
import sys,yaml
try: c=yaml.safe_load(open('$UWYK_CFG_J32'))
except Exception as e: print('PARSE_ERROR'); sys.exit()
def find(d):
    if isinstance(d,dict):
        for k,v in d.items():
            if k=='num_bars': return v
            r=find(v)
            if r is not None: return r
    return None
print(find(c))
" 2>/dev/null)
    case "$_nb" in
        32) ;;
        PARSE_ERROR) say "  BROKEN YAML: $UWYK_CFG_J32 does not parse"; problems=$((problems+1)) ;;
        *)  say "  WRONG: $UWYK_CFG_J32 has num_bars=$_nb, uwyk_D needs 32"; problems=$((problems+1)) ;;
    esac
fi
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
# Refuse to submit against paths that do not exist. FORCE=1 overrides, e.g. to
# queue behind a generation job that will create the root before these start.
if [ "$problems" -gt 0 ] && [ "$SUBMIT" = 1 ] && [ "${FORCE:-0}" != 1 ]; then
    say ""
    say "REFUSING TO SUBMIT: $problems path(s) missing. Every job touching them"
    say "would fail on arrival. Fix the paths, or set FORCE=1 if deliberate."
    exit 1
fi
say ""

N_SB=0; N_TASK=0
LAST_JID=""
# NO JOB DEPENDENCIES. One array per (model, benchmark); shift and subset ride
# the array index (SHIFTS_IN_ARRAY / SUBSETS_IN_ARRAY), so concurrency is
# submissions x PAR with nothing waiting on anything else.
go() {   # go <jobname> <lo-hi> <inner sbatch> <env...>
    local jn="$1" arr="$2" sb="$3"; shift 3
    local lo=${arr%-*} hi=${arr#*-}; local n=$(( hi - lo + 1 ))
    N_SB=$((N_SB+1)); N_TASK=$((N_TASK+n))
    if [ "$SUBMIT" = 1 ]; then
        printf '  %-42s cells=%-8s (%s in ONE job) -> ' "$jn" "$lo-$hi" "$n"
        env "$@" REPO="$REPO" INNER="$sb" IDX_LO="$lo" IDX_HI="$hi" \
            sbatch --gres="$GRES" --cpus-per-task="$CPUS" --mem="$MEM" \
            ${TIME:+--time=$TIME} ${ACCT:+--account=$ACCT} \
            --job-name="$jn" "$REPO/benchmarks/cluster/submit_aurora3_one.sbatch"
    else
        printf '  %-42s cells=%-8s (%s in ONE job) %s\n' "$jn" "$lo-$hi" "$n" "$sb"
    fi
}


for row in "${ROWS[@]}"; do
    IFS='|' read -r name harness hidx extra <<<"$row"
    # ONLY accepts a LIST (space- or comma-separated), so a subset of models can
    # be sent to one cluster in a single invocation and the MAXCONC arithmetic
    # stays correct. Two invocations would each see only their own array count.
    if [ -n "$ONLY" ]; then
        case " ${ONLY//,/ } " in *" $name "*) ;; *) continue ;; esac
    fi
    say "--- $name  (harness=$harness) ---"

    # RealCause: MODELS[ID/5], SETS[ID%5]. No MODEL_OVERRIDE, so the harness is
    # selected by array range instead.
    if [ "$BENCH" = all ] || [ "$BENCH" = rc ]; then
        go "a3-rc-$name" "$(( hidx*5 ))-$(( hidx*5+4 ))" submit_realcause_density_unified.sbatch BENCH=rc \
           $extra $CVD DENSITY_DUMP=1 \
           OUT_ROOT="$OUT_PARENT/rc/$name"
    fi

    # Case study: MODEL_OVERRIDE is supported; SHIFT is an env, not an array
    # dimension, so the three pooled shifts are three submissions.
    if [ "$BENCH" = all ] || [ "$BENCH" = cs ]; then
        go "a3-cs-$name" "0-23" submit_cs_dvar_density.sbatch BENCH=cs \
           $extra $CVD DENSITY_DUMP=1 SHIFTS_IN_ARRAY=1 \
           MODEL_OVERRIDE="$harness" CTX="$CTX" DATA="$DATA_CS" \
           ANC_MODE=case_family OUT_ROOT="$OUT_PARENT/cs/$name"
    fi

    # ComplexMech rho>0.99: PER_MODEL=30 (NODES x CONTEXTS). --subset total at
    # scoring needs BOTH nonzero and zero dumped, so two submissions.
    if [ "$BENCH" = all ] || [ "$BENCH" = cm ]; then
        go "a3-cm-$name" "0-59" submit_cmech_pehe_1d_vs_2d.sbatch BENCH=cm \
           $extra $CVD DENSITY_DUMP=1 SUBSETS_IN_ARRAY=1 \
           MODEL_OVERRIDE="$harness" UWYK_FIG34_DATA="$CMECH_DATA" \
           OUT_ROOT="$OUT_PARENT/cm_rho99/$name"
    fi
    say ""
done

say "=== $N_SB sbatch submissions, $N_TASK array tasks total ==="
say "    $N_SB JOB(S) TOTAL -- every cell runs inside its job, no arrays, no dependencies"
say "    (case-study shifts and ComplexMech subsets ride the array index)"
if [ "$MAXCONC" -gt 0 ] && [ $(( N_SB * PAR )) -gt "$MAXCONC" ]; then
    say ""
    say "OVER BUDGET: $(( N_SB * PAR )) concurrent > MAXCONC=$MAXCONC."
    say "  Lower PAR, or narrow BENCH/ONLY so fewer arrays are submitted."
    say "  Arrays per BENCH: rc=1, cs=3, cm=2 per model."
    [ "$SUBMIT" = 1 ] && exit 1
fi
say "    gres=$GRES cpus=$CPUS mem=$MEM time=${TIME:-<sbatch default>}"
say "out: $OUT_PARENT/{rc,cs,cm_rho99}/<model>/"
[ "$SUBMIT" = 1 ] || say "DRY RUN -- add --submit"
