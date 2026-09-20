#!/bin/bash
# Wait out a slurm outage, then submit everything pending. Fire and forget.
#
# The controller is refusing work ("backup controller in standby mode", then
# "Socket timed out on send/recv"). Nothing script-side fixes that, so this polls
# until it answers and submits then, instead of you retrying by hand.
#
# Launch detached so it survives logout:
#   cd /scratch/furkanbd/rpfn_bench_kit
#   nohup bash R-PFN/benchmarks/cluster/submit_when_up.sh > sub.log 2>&1 &
#   tail -f sub.log
#
# Idempotent: every successful submission is recorded in a state file and never
# repeated, so re-running after a partial outage submits only what is still
# missing. Delete the state file to force a resubmit.
#
#   POLL=120  seconds between controller checks
#   DEADLINE  give up after this many seconds (default 24h)
#   STATE     state file (default $SMOKE/submitted.txt)

set -uo pipefail
# Paths are DERIVED, not hardcoded, so this runs on any cluster: the script lives
# at <kit>/R-PFN/benchmarks/cluster/, so the kit root is three levels up. SCRATCH
# must be set by the environment -- guessing a per-cluster scratch path is how a
# run silently writes to the wrong filesystem.
_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_KIT_DEFAULT="$(cd "$_SELF_DIR/../../.." && pwd)"
KIT="${KIT:-$_KIT_DEFAULT}"
REPO="${REPO:-$KIT/R-PFN}"
CK="${CK:-$REPO/Required_checkpoints}"
SC="${SCRATCH:?SCRATCH must be set}"
SMOKE="${SMOKE:-$SC/smoke_new}"
POLL="${POLL:-120}"
DEADLINE="${DEADLINE:-86400}"
STATE="${STATE:-$SMOKE/submitted.txt}"
mkdir -p "$SMOKE"; touch "$STATE"

RC_SB="$REPO/benchmarks/cluster/submit_realcause_density_unified.sbatch"
CS_SB="$REPO/benchmarks/cluster/submit_cs_dvar_density.sbatch"
FT_SB="$REPO/benchmarks/cluster/submit_full_table.sbatch"

# label | env assignments | sbatch args | script
#
# The scoring job goes FIRST: it needs no new dumps and produces the actual
# deliverable (all eight columns plus forced-independence) from the six models
# that already have data, so if the controller accepts only one thing, that is
# the one worth getting in.
#
# Array indices select the harness inside each sbatch's MODELS array:
#   RC: task = 5*model_idx + dataset_idx   (0 = IHDP)
#   CS: task = 8*model_idx + d_idx         (2 = d5)
# MODELS=(dopfn_native dopfn_bb uwyk1d graph2d cpfn1d cpfn2d_pooled).
JOBS=(
"full_table|OUT_ROOT=$SC/ft_unused|--job-name=full-table|$FT_SB"

"J10-rc|DOPFN_CKPT=$CK/dopfn_repro_1d_J10_step150000.pt OUT_ROOT=$SMOKE/dopfn_repro_1d_J10/rc DENSITY_DUMP=1|--array=0 --job-name=sm-rc-J10|$RC_SB"
"J10-cs|DOPFN_CKPT=$CK/dopfn_repro_1d_J10_step150000.pt OUT_ROOT=$SMOKE/dopfn_repro_1d_J10/cs DENSITY_DUMP=1 SHIFT=0|--array=2 --job-name=sm-cs-J10|$CS_SB"

"J100-rc|DOPFN_CKPT=$CK/dopfn_repro_1d_J100_step150000.pt OUT_ROOT=$SMOKE/dopfn_repro_1d_J100/rc DENSITY_DUMP=1|--array=0 --job-name=sm-rc-J100|$RC_SB"
"J100-cs|DOPFN_CKPT=$CK/dopfn_repro_1d_J100_step150000.pt OUT_ROOT=$SMOKE/dopfn_repro_1d_J100/cs DENSITY_DUMP=1 SHIFT=0|--array=2 --job-name=sm-cs-J100|$CS_SB"

"joint2d-rc|CKPT_DOPFN_BB=$CK/dopfn_repro_joint2d_bb.pt OUT_ROOT=$SMOKE/dopfn_repro_joint2d/rc DENSITY_DUMP=1|--array=5 --job-name=sm-rc-joint2d|$RC_SB"
"joint2d-cs|CKPT_DOPFN_BB=$CK/dopfn_repro_joint2d_bb.pt OUT_ROOT=$SMOKE/dopfn_repro_joint2d/cs DENSITY_DUMP=1 SHIFT=0|--array=10 --job-name=sm-cs-joint2d|$CS_SB"

"j32-rc|CKPT_CPFN1D=$CK/cpfn1d_j32_step50000.pt OUT_ROOT=$SMOKE/cpfn1d_j32/rc DENSITY_DUMP=1|--array=20 --job-name=sm-rc-j32|$RC_SB"
"j32-cs|CKPT_CPFN1D=$CK/cpfn1d_j32_step50000.pt OUT_ROOT=$SMOKE/cpfn1d_j32/cs DENSITY_DUMP=1 SHIFT=0|--array=34 --job-name=sm-cs-j32|$CS_SB"

"botharms-rc|CKPT_CPFN1D=$CK/cpfn1d_botharms_step50000.pt OUT_ROOT=$SMOKE/cpfn1d_botharms/rc DENSITY_DUMP=1|--array=20 --job-name=sm-rc-botharms|$RC_SB"
"botharms-cs|CKPT_CPFN1D=$CK/cpfn1d_botharms_step50000.pt OUT_ROOT=$SMOKE/cpfn1d_botharms/cs DENSITY_DUMP=1 SHIFT=0|--array=34 --job-name=sm-cs-botharms|$CS_SB"
)

log() { echo "[$(date '+%F %T')] $*"; }

controller_up() {
    # scontrol ping names the primary/backup explicitly; sinfo alone can answer
    # from cache. Require a real partition line back before trusting it.
    timeout 15 sinfo -h -o "%P" >/dev/null 2>&1 || return 1
    [ -n "$(timeout 15 sinfo -h -o '%P' 2>/dev/null | head -1)" ]
}

cd "$KIT" || exit 1
START=$(date +%s)
log "waiting for slurm; $(wc -l < "$STATE" | tr -d ' ') job(s) already submitted"

while :; do
    NOW=$(date +%s)
    if [ $(( NOW - START )) -gt "$DEADLINE" ]; then
        log "DEADLINE reached; giving up. Re-run to continue."; exit 1
    fi

    if ! controller_up; then
        log "controller not answering; retry in ${POLL}s"
        sleep "$POLL"; continue
    fi

    PENDING=0; DONE=0
    for j in "${JOBS[@]}"; do
        IFS='|' read -r label envs args script <<<"$j"
        if grep -qxF "$label" "$STATE"; then DONE=$((DONE+1)); continue; fi
        [ -f "$script" ] || { log "SKIP $label: no $script"; continue; }
        # Unquoted on purpose: envs and args are whitespace-separated lists and
        # none of these paths contain spaces.
        out=$(timeout 90 env $envs sbatch $args "$script" 2>&1); st=$?
        if [ "$st" = 0 ] && printf '%s' "$out" | grep -q 'Submitted batch job'; then
            log "OK   $label -> $out"
            echo "$label" >> "$STATE"
            DONE=$((DONE+1))
        else
            log "FAIL $label (exit $st): $out"
            PENDING=$((PENDING+1))
        fi
        sleep 2
    done

    log "submitted=$DONE  still-pending=$PENDING"
    [ "$PENDING" = 0 ] && { log "all jobs submitted"; break; }
    log "retry the remainder in ${POLL}s"
    sleep "$POLL"
done

log "done. watch:  squeue --me -o '%.10i %.18j %.8T %.10M %R'"
