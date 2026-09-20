#!/bin/bash
# Submit the 10 smoke cells, with nothing that can block.
#
# The inspection stage already verified every checkpoint, so this skips it and
# only submits. Differences from smoke_new_models.sh that matter when the slurm
# controller is sick:
#   * each sbatch runs under `timeout`, so a hung controller costs 60s, not the run
#   * no pipe into sed -- output goes straight to the terminal
#   * a failed submit is reported and skipped; the remaining jobs still go in
#
# Re-running is safe: each cell writes to its own OUT_ROOT, so a duplicate
# submission overwrites its own output rather than mixing with another model's.
#
#   bash R-PFN/benchmarks/cluster/smoke_submit_only.sh
#   ONLY=cpfn1d_j32 bash R-PFN/benchmarks/cluster/smoke_submit_only.sh   # one model
#   BENCH=rc        bash R-PFN/benchmarks/cluster/smoke_submit_only.sh   # one benchmark

set -uo pipefail
KIT="${KIT:-/scratch/furkanbd/rpfn_bench_kit}"
REPO="${REPO:-$KIT/R-PFN}"
CK="${CK:-$REPO/Required_checkpoints}"
SMOKE="${SMOKE:-${SCRATCH:-/scratch/furkanbd}/smoke_new}"
TMO="${TMO:-60}"
ONLY="${ONLY:-}"
BENCH="${BENCH:-both}"
RC_SB="$REPO/benchmarks/cluster/submit_realcause_density_unified.sbatch"
CS_SB="$REPO/benchmarks/cluster/submit_cs_dvar_density.sbatch"

# name | env var | checkpoint | RC array task | CS array task
# Array indices select the harness inside each sbatch's MODELS array:
#   RC: task = 5*model_idx + dataset_idx   (dataset 0 = IHDP)
#   CS: task = 8*model_idx + d_idx         (d_idx 2 = d5)
# with MODELS=(dopfn_native dopfn_bb uwyk1d graph2d cpfn1d cpfn2d_pooled).
ROWS=(
  "dopfn_repro_1d_J10|DOPFN_CKPT|$CK/dopfn_repro_1d_J10_step150000.pt|0|2"
  "dopfn_repro_1d_J100|DOPFN_CKPT|$CK/dopfn_repro_1d_J100_step150000.pt|0|2"
  # native harness (index 0): RC task 0, CS task 2
  "dopfn_repro_joint2d|DOPFN_CKPT|$CK/dopfn_repro_joint2d_step150000.pt|0|2"
  "cpfn1d_j32|CKPT_CPFN1D|$CK/cpfn1d_j32_step50000.pt|20|34"
  "cpfn1d_botharms|CKPT_CPFN1D|$CK/cpfn1d_botharms_step50000.pt|20|34"
  # cpfn_v0 is a symlink to warmstart/causalpfn_v0.pt (75.4 MB, J=1024).
  # It was absent from this list, so ONLY=cpfn_v0 silently matched nothing
  # and reported submitted=0.
  "cpfn_v0|CKPT_CPFN1D|${CKPT_CPFN_V0:-$CK/cpfn_v0_original.pt}|20|34"
  # Binarized UWYK. Harness index 2 (uwyk1d), so RC task 10 and CS task 18.
  # Needs CONFIG and UWYK_T_ENCODING=binary alongside CKPT -- see EXTRA below.
  "uwyk_bin|CKPT|$CK/uwyk_bin_step50000.pt|10|18"
)

cd "$KIT" || exit 1
echo "controller check:"
timeout 10 sinfo -h -o "%P %a" | head -3
echo "  sinfo exit=$?   (non-zero or empty => slurm is down, stop here)"
echo

OK=0; BAD=0
for r in "${ROWS[@]}"; do
    IFS='|' read -r name env ck rc cs <<<"$r"
    [ -n "$ONLY" ] && [ "$ONLY" != "$name" ] && continue
    [ -f "$ck" ] || { echo "SKIP $name: missing $ck"; BAD=$((BAD+1)); continue; }
    for b in rc cs; do
        [ "$BENCH" = both ] || [ "$BENCH" = "$b" ] || continue
        if [ "$b" = rc ]; then task="$rc"; sb="$RC_SB"; extra=""
        else                  task="$cs"; sb="$CS_SB"; extra="SHIFT=0"
        fi
        echo "--- $name/$b  array=$task"
        # The UWYK harness needs a config YAML and a treatment encoding beside
        # its weights; every other model needs only CKPT. Default the config to
        # the one shipped in Required_checkpoints unless UWYK_BIN_CONFIG is set.
        uwyk_extra=""
        if [ "$name" = uwyk_bin ]; then
            uwyk_extra="CONFIG=${UWYK_BIN_CONFIG:-$CK/uwyk_USED_IN_RESULTS_best_model_config.yaml} UWYK_T_ENCODING=${UWYK_T_ENCODING:-binary}"
        fi
        timeout "$TMO" env "$env=$ck" OUT_ROOT="$SMOKE/$name/$b" DENSITY_DUMP=1 $extra $uwyk_extra \
            sbatch --array="$task" --job-name="sm-$b-$name" "$sb"
        st=$?
        if [ "$st" = 0 ]; then OK=$((OK+1))
        elif [ "$st" = 124 ]; then echo "    TIMEOUT after ${TMO}s -- controller not answering"; BAD=$((BAD+1))
        else echo "    sbatch failed (exit $st)"; BAD=$((BAD+1)); fi
    done
done
echo
if [ "$OK" = 0 ] && [ "$BAD" = 0 ]; then
    echo "NOTHING MATCHED. ONLY='${ONLY}' selected no row; known rows are:" >&2
    for r in "${ROWS[@]}"; do echo "    ${r%%|*}" >&2; done
fi
echo "submitted=$OK  failed=$BAD"
echo "watch:  squeue --me -o '%.10i %.18j %.8T %.10M %R'"
