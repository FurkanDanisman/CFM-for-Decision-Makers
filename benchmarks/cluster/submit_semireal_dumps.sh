#!/bin/bash
# Dump the Do-PFN semi-real datasets (sales, law_race) for all 13 models.
#
# Reuses submit_realcause_density_unified.sbatch unchanged apart from
# DATASET_OVERRIDE: the semi-real sets resolve through the same _cmech_dataset hook
# every harness already consults, so the harness dispatch, the density dump and the
# per-model checkpoint handling all work as they do for RealCause.
#
# A REALIZATION IS A SPLIT here, and there are five -- Do-PFN's own
# generate_valid_split protocol. Five is a much smaller denominator than the 100 used
# elsewhere, so coverage from this benchmark carries wide error bars; report n.
#
#   bash R-PFN/benchmarks/cluster/submit_semireal_dumps.sh            # dry run
#   bash R-PFN/benchmarks/cluster/submit_semireal_dumps.sh --submit
#   ONLY="cpfn1d_j32" DATASETS="SEMIREAL_sales" bash ... --submit

set -uo pipefail
SUBMIT=0; [ "${1:-}" = "--submit" ] && SUBMIT=1
_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
KIT="${KIT:-$(cd "$_SELF_DIR/../../.." && pwd)}"
REPO="${REPO:-$KIT/R-PFN}"
CK="${CK:-$REPO/Required_checkpoints}"
SC="${SCRATCH:?SCRATCH must be set}"
SB="$REPO/benchmarks/cluster/submit_realcause_density_unified.sbatch"
DATASETS="${DATASETS:-SEMIREAL_sales SEMIREAL_law_race}"
ACCT="${ACCOUNT:-}"; PART="${PARTITION:-}"; GRES="${GRES:-}"
MEM="${MEM:-32G}"; CPUS="${CPUS:-8}"; TIME="${DUMP_TIME:-3:00:00}"
ONLY="${ONLY:-}"

# name | harness array index | ckpt env | checkpoint
#   0 dopfn_native  1 dopfn_bb  2 uwyk1d  3 graph2d  4 cpfn1d  5 cpfn2d_pooled
ROWS=(
  "dopfn_native|0|NONE|-"
  "dopfn_bb|1|CKPT_DOPFN_BB|$CK/dopfn_bb_j10_step_150000.pt"
  "dopfn_repro_1d_J10|0|DOPFN_CKPT|$CK/dopfn_repro_1d_J10_step150000.pt"
  "dopfn_repro_1d_J100|0|DOPFN_CKPT|$CK/dopfn_repro_1d_J100_step150000.pt"
  "dopfn_repro_joint2d|0|DOPFN_CKPT|$CK/dopfn_repro_joint2d_step150000.pt"
  "uwyk1d|2|NONE|-"
  "uwyk_bin|2|CKPT|$CK/uwyk_bin_step50000.pt"
  "graph2d|3|CKPT_GRAPH2D|$CK/graph2d_step_50000.pt"
  "cpfn1d_j1024|4|CKPT_CPFN1D|$CK/cpfn1d_j1024_headrand_step_50000.pt"
  "cpfn1d_j32|4|CKPT_CPFN1D|$CK/cpfn1d_j32_step50000.pt"
  "cpfn1d_botharms|4|CKPT_CPFN1D|$CK/cpfn1d_botharms_step50000.pt"
  "cpfn_v0|4|CKPT_CPFN1D|$CK/cpfn_v0_original.pt"
  "cpfn2d_eta0|5|CKPT_CPFN2D|$CK/cpfn2d_j32_eta0_y01_step50000.pt"
)

[ -d "${DOPFN_ROOT:-$KIT/external/dopfn}" ] || {
    echo "FATAL: no Do-PFN repo. These datasets are loaded by its own datasets" >&2
    echo "       module, which resolves artifacts by relative path." >&2; exit 1; }
export DOPFN_ROOT="${DOPFN_ROOT:-$KIT/external/dopfn}"
cd "$KIT" || exit 1

N=0
for r in "${ROWS[@]}"; do
    IFS='|' read -r name idx envv ck <<<"$r"
    [ -n "$ONLY" ] && { case " $ONLY " in *" $name "*) ;; *) continue ;; esac; }
    if [ "$envv" != NONE ]; then
        [ -f "$ck" ] || { printf '%-22s SKIP: no %s\n' "$name" "$ck"; continue; }
        sz=$(stat -Lc%s "$ck" 2>/dev/null || stat -Lf%z "$ck" 2>/dev/null || echo "")
        if [ -n "$sz" ] && [ "$sz" -lt 1000000 ] 2>/dev/null; then
            printf '%-22s SKIP: %s bytes (pointer)\n' "$name" "$sz"; continue; fi
    fi
    uwyk_extra=""
    [ "$name" = uwyk_bin ] && uwyk_extra="CONFIG=$CK/uwyk_USED_IN_RESULTS_best_model_config.yaml UWYK_T_ENCODING=binary"
    for ds in $DATASETS; do
        N=$((N+1))
        # array index selects the harness; %5 is unused once DATASET_OVERRIDE is set
        task=$(( idx * 5 ))
        if [ "$SUBMIT" = 1 ]; then
            printf '%-22s %-20s -> ' "$name" "$ds"
            # shellcheck disable=SC2086
            env ${envv:+$( [ "$envv" != NONE ] && echo "$envv=$ck" )} \
                DATASET_OVERRIDE="$ds" OUT_ROOT="$SC/semireal_dumps/$name" \
                DENSITY_DUMP=1 DOPFN_ROOT="$DOPFN_ROOT" $uwyk_extra \
                sbatch --array="$task" --time="$TIME" --mem="$MEM" \
                       --cpus-per-task="$CPUS" \
                       ${GRES:+--gres=$GRES} ${ACCT:+--account=$ACCT} \
                       ${PART:+--partition=$PART} \
                       --job-name="sr-$name-${ds#SEMIREAL_}" "$SB"
        else
            printf '%-22s %-20s idx=%s %s\n' "$name" "$ds" "$idx" "$(basename "$ck")"
        fi
    done
done
echo
echo "jobs: $N   (13 models x $(echo $DATASETS | wc -w) datasets, 5 splits each)"
[ "$SUBMIT" = 1 ] || echo "dry run -- add --submit"
