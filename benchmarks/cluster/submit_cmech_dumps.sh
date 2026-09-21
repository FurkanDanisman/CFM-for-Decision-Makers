#!/bin/bash
# ComplexMech dumps: one job per model, all 13 models, same pattern as RealCause
# and the case studies.
#
# No change to the dump dispatch: submit_cmech_pehe_1d_vs_2d.sbatch already selects
# a harness from SLURM_ARRAY_TASK_ID and takes weights from CKPT_* / DOPFN_CKPT, so
# each model is an existing harness at a different checkpoint and OUT_ROOT. Model
# identity lives in the root, which is what keeps two models that share a harness
# (and therefore a subdir name) from merging.
#
# Grid per job: 6 node counts x 1 context (N=1000) x 2 subsets = 12 cells, walked
# sequentially. The inner script's array index encodes model x node x context, so
# the wrapper sets it directly -- the .sbatch body is plain bash and #SBATCH lines
# are comments.
#
#   bash R-PFN/benchmarks/cluster/submit_cmech_dumps.sh            # dry run
#   bash R-PFN/benchmarks/cluster/submit_cmech_dumps.sh --submit
#   ONLY="cpfn1d_j32 uwyk_bin" bash ... --submit

set -uo pipefail
SUBMIT=0; [ "${1:-}" = "--submit" ] && SUBMIT=1
_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
KIT="${KIT:-$(cd "$_SELF_DIR/../../.." && pwd)}"
REPO="${REPO:-$KIT/R-PFN}"
CK="${CK:-$REPO/Required_checkpoints}"
SC="${SCRATCH:?SCRATCH must be set}"
DATA="${CMECH_DATA:-$SC/cmech_data_v2}"
ACCT="${ACCOUNT:-}"; PART="${PARTITION:-}"
TIME="${DUMP_TIME:-3:00:00}"
# nibi refuses a bare --gres=gpu:N and demands a type. h100 is the most plentiful
# (232), a5000 and t4 are smaller and often quicker to schedule for a single-GPU
# inference job like this.
GRES="${GRES:-gpu:h100:1}"
ONLY="${ONLY:-}"
SB="$REPO/benchmarks/cluster/submit_cmech_dump_one.sbatch"

# name | harness index | ckpt env | checkpoint | extra env
#   0 dopfn_native  1 dopfn_bb  2 uwyk1d  3 graph2d  4 cpfn1d  5 cpfn2d_pooled
#
# dopfn_repro_joint2d goes through harness 0, not 1: eval_native_dopfn handles the
# joint_2d variant and applies DoPFN's own y normalisation, which is what fixed the
# grid mismatch on RealCause. Routing it through dopfn_bb would reintroduce that.
ROWS=(
  "dopfn_native|0|NONE|-|"
  "dopfn_bb|1|CKPT_DOPFN_BB|$CK/dopfn_bb_j10_step_150000.pt|"
  "dopfn_repro_1d_J10|0|DOPFN_CKPT|$CK/dopfn_repro_1d_J10_step150000.pt|"
  "dopfn_repro_1d_J100|0|DOPFN_CKPT|$CK/dopfn_repro_1d_J100_step150000.pt|"
  "dopfn_repro_joint2d|0|DOPFN_CKPT|$CK/dopfn_repro_joint2d_step150000.pt|"
  "uwyk1d|2|NONE|-|"
  "uwyk_bin|2|CKPT|$CK/uwyk_bin_step50000.pt|CONFIG=$CK/uwyk_USED_IN_RESULTS_best_model_config.yaml UWYK_T_ENCODING=binary"
  "graph2d|3|CKPT_GRAPH2D|$CK/graph2d_step_50000.pt|"
  "cpfn1d_j1024|4|CKPT_CPFN1D|$CK/cpfn1d_j1024_headrand_step_50000.pt|"
  "cpfn1d_j32|4|CKPT_CPFN1D|$CK/cpfn1d_j32_step50000.pt|"
  "cpfn1d_botharms|4|CKPT_CPFN1D|$CK/cpfn1d_botharms_step50000.pt|"
  "cpfn_v0|4|CKPT_CPFN1D|$CK/cpfn_v0_original.pt|"
  "cpfn2d_eta0|5|CKPT_CPFN2D|$CK/cpfn2d_j32_eta0_y01_step50000.pt|"
)

[ -d "$DATA/complexmech" ] || { echo "FATAL: no data at $DATA/complexmech" >&2; exit 1; }
[ -f "$DATA/manifest_complexmech.json" ] || echo "WARN: no manifest at $DATA -- is this the filtered benchmark?" >&2
cd "$KIT" || exit 1

N=0
for r in "${ROWS[@]}"; do
    IFS='|' read -r name idx envv ck extra <<<"$r"
    [ -n "$ONLY" ] && { case " $ONLY " in *" $name "*) ;; *) continue ;; esac; }
    if [ "$envv" != NONE ]; then
        if [ ! -f "$ck" ]; then printf '%-22s SKIP: no %s\n' "$name" "$ck"; continue; fi
        # -L follows symlinks (two checkpoints are links whose own "size" is just
        # their target path length). stat -Lc is GNU-only, so fall back to BSD -f
        # and, if NEITHER works, proceed rather than skip: a size we could not read
        # is not evidence the file is bad, and failing closed here would silently
        # drop every model.
        sz=$(stat -Lc%s "$ck" 2>/dev/null || stat -Lf%z "$ck" 2>/dev/null || echo "")
        if [ -n "$sz" ] && [ "$sz" -lt 1000000 ] 2>/dev/null; then
            printf '%-22s SKIP: %s bytes -- git-lfs pointer, not weights\n' "$name" "$sz"
            continue
        fi
    fi
    N=$((N+1))
    if [ "$SUBMIT" = 1 ]; then
        printf '%-22s idx=%s -> ' "$name" "$idx"
        env MODEL_NAME="$name" MODEL_IDX="$idx" CKPT_ENV="$envv" CKPT_PATH="$ck" \
            EXTRA_ENV="$extra" OUT_ROOT="$SC/cmech_dumps/$name" \
            UWYK_FIG34_DATA="$DATA" \
            sbatch --time="$TIME" --gres="$GRES" \
                   ${ACCT:+--account=$ACCT} ${PART:+--partition=$PART} \
                   --job-name="cm-$name" "$SB"
    else
        printf '%-22s idx=%-2s %s\n' "$name" "$idx" "$(basename "$ck")"
    fi
done
echo
echo "jobs: $N   (each walks 6 node counts x 2 subsets at N=1000)"
[ "$SUBMIT" = 1 ] || echo "dry run -- add --submit"
