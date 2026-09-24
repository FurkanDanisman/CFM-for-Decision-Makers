#!/bin/bash
# Submit exactly what is still missing from the tau-C MALC-T run.
#
#     bash benchmarks/cluster/submit_density_tauC_chain.sh
#
# STATE-DRIVEN, not phase-driven. Earlier versions of this script hardcoded
# "run the repair, then the MALC arms" and had to be rewritten every time the
# state moved. This one counts what is on disk and submits the gaps, so it is
# safe to run repeatedly and does nothing when everything is complete.
#
# For each shard it checks two things:
#   raw dumps   <shard>/<DS>/predictions/*.npz      100 IHDP, 10 ACIC
#   MALC-T      results_density_tauC_malcT/<shard>/<DS>/*.npz
# and submits the raw array tasks that are missing, then a MALC-T job gated on
# them (afterok) when it had to submit any, ungated when it did not.
#
# ── THE TASK-0 TRAP, WHICH HAS COST TEN REALIZATIONS TWICE ────────────────
# The raw array maps task t -> IHDP realizations [10t, 10t+10) for t < 10, and
# tasks 10-11 -> ACIC in blocks of 5. Task 0 ALSO runs the validation gates,
# and until this was fixed three of the four gates had no error handling, so
# under `set -euo pipefail` a gate failure killed task 0 silently -- no message,
# no eval, and exactly r000-r009 missing while the other 11 tasks succeeded.
# That is why the original CausalPFN shard and the first Do-PFN refresh both
# came back 90/100. submit_density_tauC.sbatch now names the failing gate and
# offers SKIP_GATES=1; this script maps missing realizations back to array task
# ids so the recovery is one submission rather than a guess.
#
# DRY RUN -- print what would be submitted, submit nothing:
#     CHAIN_DRY_RUN=1 bash benchmarks/cluster/submit_density_tauC_chain.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

# Point every job at THIS checkout. The sbatches each default REPO to a
# hardcoded cluster path, and the shards record three different roots (the
# families were dumped on different machines), so pinning it here is what keeps
# one submission internally consistent.
export REPO="${REPO:-$PWD}"

RAW="$REPO/benchmarks/cluster/submit_density_tauC.sbatch"
MALCT="$REPO/benchmarks/cluster/submit_density_tauC_malcT.sbatch"
RESULTS="${RESULTS:-$REPO/results_density_tauC}"
MALCT_OUT="${MALCT_OUT:-$REPO/results_density_tauC_malcT}"
DRY="${CHAIN_DRY_RUN:-0}"

# shard : model family for the raw job (or '-' when its raw dumps are final and
# must never be regenerated -- the UWYK and CausalPFN shards predate this
# script and their raw runs are not reproducible from here).
SHARDS=(
    "5312882:-"
    "5312884:-"
    "5571187:-"
    "dopfn_refresh:dopfn"
)

# Pure-bash count, no pipeline. `ls dir/*.npz | wc -l` looks equivalent and is
# not: under `set -o pipefail` a directory that does not exist yet makes ls exit
# non-zero, which takes the whole script down through `set -e` before it can
# report that the shard has no MALC-T output. Which is exactly the case this
# script exists to detect.
count() {
    local n=0 f
    for f in "$1"/*.npz; do [ -f "$f" ] && n=$((n+1)); done
    echo "$n"
}

# Realizations present -> the array task ids that would recompute the rest.
# Task t covers IHDP [10t, 10t+10) for t in 0..9; ACIC [5(t-10), 5(t-10)+5) for
# t in 10..11. A task is resubmitted if ANY of its realizations is absent; the
# eval skips the ones already there, so an over-broad task is free.
missing_tasks() {
    local shard="$1" out=""
    local t lo hi ds n r f
    for t in $(seq 0 11); do
        if [ "$t" -lt 10 ]; then ds=IHDP; lo=$((t*10)); hi=$((t*10+10))
        else ds=ACIC; lo=$(((t-10)*5)); hi=$(((t-10)*5+5)); fi
        for r in $(seq "$lo" $((hi-1))); do
            f=$(printf '%s/%s/%s/predictions/%s_r%03d.npz' "$RESULTS" "$shard" "$ds" "$ds" "$r")
            [ -f "$f" ] || { out="$out,$t"; break; }
        done
    done
    echo "${out#,}"
}

submit() {   # submit <description> <args...>
    local desc="$1"; shift
    if [ "$DRY" = "1" ]; then
        echo "[dry-run] $desc" >&2
        echo "          sbatch $*" >&2
        echo "DRYRUN$RANDOM"
    else
        sbatch --parsable "$@"
    fi
}

printf '%-16s %-14s %-14s %s\n' SHARD "RAW(IHDP/ACIC)" "MALCT(I/A)" ACTION
any=0
for entry in "${SHARDS[@]}"; do
    shard="${entry%%:*}"; family="${entry#*:}"
    ri=$(count "$RESULTS/$shard/IHDP/predictions")
    ra=$(count "$RESULTS/$shard/ACIC/predictions")
    mi=$(count "$MALCT_OUT/$shard/IHDP")
    ma=$(count "$MALCT_OUT/$shard/ACIC")

    raw_id=""; action=""
    if [ "$ri" -lt 100 ] || [ "$ra" -lt 10 ]; then
        if [ "$family" = "-" ]; then
            action="RAW INCOMPLETE - not regenerable from here, skipping"
        else
            tasks=$(missing_tasks "$shard")
            action="raw --array=$tasks"
        fi
    fi
    if [ "$mi" -lt 100 ] || [ "$ma" -lt 10 ]; then
        action="${action:+$action + }malcT"
    fi
    [ -n "$action" ] || action="complete, nothing to do"
    printf '%-16s %-14s %-14s %s\n' "$shard" "$ri/$ra" "$mi/$ma" "$action"

    case "$action" in complete*|RAW\ INCOMPLETE*) continue;; esac
    any=1

    if [ -n "${tasks:-}" ] && [ "$family" != "-" ] && { [ "$ri" -lt 100 ] || [ "$ra" -lt 10 ]; }; then
        raw_id=$(MODEL_FAMILY="$family" OUT_ROOT="$RESULTS/$shard" \
                 DOPFN_ROOT="${DOPFN_ROOT:-${DOPFN:-$REPO/Do-PFN}}" \
                 SKIP_GATES="${SKIP_GATES:-0}" \
                 submit "raw $shard tasks $tasks" --array="$tasks" "$RAW")
        echo "    raw   $shard  $raw_id  (array $tasks)"
    fi
    if [ "$mi" -lt 100 ] || [ "$ma" -lt 10 ]; then
        if [ -n "$raw_id" ]; then
            id=$(DUMPS_ROOT="$RESULTS/$shard" OUT_ROOT="$MALCT_OUT/$shard" \
                 submit "malcT $shard" --dependency=afterok:"$raw_id" "$MALCT")
            echo "    malcT $shard  $id  after raw $raw_id"
        else
            id=$(DUMPS_ROOT="$RESULTS/$shard" OUT_ROOT="$MALCT_OUT/$shard" \
                 submit "malcT $shard" "$MALCT")
            echo "    malcT $shard  $id  (no dependency)"
        fi
    fi
    unset tasks
done

if [ "$any" = "0" ]; then
    echo; echo "Everything is complete. Nothing submitted."
    exit 0
fi

if [ "$DRY" = "1" ]; then echo; echo "CHAIN_DRY_RUN=1 -- nothing submitted."; exit 0; fi

cat <<'EOF'

Watch with:  squeue -u $USER

IF A RAW TASK 0 FAILS, read its log before anything else -- it now names the
gate that failed. Re-run with SKIP_GATES=1 only once you have read that failure
and decided it does not invalidate the numbers. Tasks 1-11 never run the gates,
so skipping them on a task-0 re-run leaves the shard internally uniform.

Rsync the RAW job's logs too -- logs_density_tauC/ -- not just the MALC ones.
The last two investigations were blind because only logs_density_tauC_malcT/
and logs_density_tauC_repair/ came back.
EOF
