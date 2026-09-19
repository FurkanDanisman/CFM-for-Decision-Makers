#!/bin/bash
# Smoke test the six checkpoints that have no dumps yet, on ONE cell each.
#
# No harness code changes: the two dump sbatches already dispatch on $MODEL and
# take weights from CKPT_* / DOPFN_CKPT, so each new model is an existing
# harness pointed at a different checkpoint and a different OUT_ROOT -- the same
# arrangement the eta0 model already runs under. One METHODS row is therefore
# reused per harness, and the model identity lives in the root, not the subdir.
#
# Array indices are forced so exactly one cell runs per model:
#   RealCause    task = 5*model_idx + dataset_idx   datasets: IHDP ACIC CPS PSID PSID_bal
#   Case study   task = 8*model_idx + d_idx         d: 2 3 5 10 20 30 40 50
# with MODELS=(dopfn_native dopfn_bb uwyk1d graph2d cpfn1d cpfn2d_pooled), so
# dopfn_native=0, dopfn_bb=1, cpfn1d=4. We take IHDP (idx 0) and d=5 (idx 2).
#
#   bash R-PFN/benchmarks/cluster/smoke_new_models.sh            # inspect only
#   bash R-PFN/benchmarks/cluster/smoke_new_models.sh --submit    # inspect + submit
#
# Stage 1 (checkpoint inspection) is CPU-only and runs here on the login node:
# it only torch.loads each file to print its provenance. If a path is wrong you
# find out in seconds instead of after a GPU queue wait.

set -uo pipefail
SUBMIT=0; [ "${1:-}" = "--submit" ] && SUBMIT=1

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found from $PWD" >&2; exit 1; }
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
SMOKE="${SMOKE:-$SCRATCH/smoke_new}"

# Where the six .pt files landed. Tried in order; first hit wins.
# Require .pt files, not merely an existing directory: an EMPTY
# $SCRATCH/final_checkpoints shadowed the real Required_checkpoints and every
# model came back MISSING while the weights sat one directory away.
FINAL=""
for c in "${FINAL_CKPTS:-}" "$REPO/Required_checkpoints" "$REPO/final_checkpoints" \
         "$DEPLOY_ROOT/final_checkpoints" "$SCRATCH/final_checkpoints" \
         "$HOME/final_checkpoints"; do
    [ -n "$c" ] || continue
    [ -d "$c" ] || continue
    if [ "$(find "$c" -maxdepth 1 -name '*.pt' | head -1)" ]; then FINAL="$c"; break; fi
    echo "note: $c exists but holds no .pt files -- skipping"
done
[ -n "$FINAL" ] || { echo "FATAL: final_checkpoints not found. Set FINAL_CKPTS=<dir>" >&2; exit 1; }
echo "FINAL_CKPTS = $FINAL"
echo "SMOKE roots = $SMOKE/<model>"
echo

# model | harness-model | array-idx-base | ckpt-env | ckpt-path
# cpfn_v0 is deliberately unresolved: the two checkpoints in Required_checkpoints
# (step_50000_final.pt, cpfn2d_j32_random_step_50000.pt) are both on the
# do-not-run list, so guessing which file is "cpfn v0" would be exactly the kind
# of filename-trust that has burned this project before. Set CKPT_CPFN_V0 to
# include it.
# model | harness-model | array-idx-base | ckpt-env | filename glob
#
# Globs, not exact names: the copy on scratch does not necessarily carry the
# same filenames as the source tree, and a hardcoded name turns that into a
# MISSING that looks like an absent model. The glob must match exactly one file
# -- an ambiguous pattern is an error, not a coin flip, because picking the
# wrong .pt here is precisely the failure mode that costs days.
#
# cpfn_v0's file is named unambiguously (cpfn_v0_original.pt) so there is nothing
# left to guess -- but on this filesystem it is 58 bytes, i.e. a git-lfs pointer
# checked out as text, because killarney has no git-lfs. The size gate in
# resolve() catches that and reports it as a POINTER rather than letting torch
# fail on a text file. Copy the real weights across, or set CKPT_CPFN_V0.
ROWS=(
  "dopfn_repro_1d_J10|dopfn_native|0|DOPFN_CKPT|*repro*1d*J10_*.pt"
  "dopfn_repro_1d_J100|dopfn_native|0|DOPFN_CKPT|*repro*1d*J100*.pt"
  "dopfn_repro_joint2d|dopfn_bb|1|CKPT_DOPFN_BB|@REPO@/Required_checkpoints/dopfn_repro_joint2d_bb.pt"
  "cpfn1d_j32|cpfn1d|4|CKPT_CPFN1D|*cpfn1d*j32*.pt"
  "cpfn1d_botharms|cpfn1d|4|CKPT_CPFN1D|*botharms*.pt"
  "cpfn_v0|cpfn1d|4|CKPT_CPFN1D|${CKPT_CPFN_V0:+@ENV@CKPT_CPFN_V0}${CKPT_CPFN_V0:-*cpfn_v0*.pt}"
)

# Resolve one glob against $FINAL. Echoes the path, or an empty string plus a
# diagnostic on stderr when the pattern matches zero or several files.
resolve() {
    local pat="$1" name="$2"
    case "$pat" in
      @REPO@*) local f="${pat/@REPO@/$REPO}"
               [ -f "$f" ] && { echo "$f"; return 0; }
               echo "  $name: not at $f" >&2; return 1 ;;
      @ENV@*)  local v="${pat#@ENV@}"; local f="${!v:-}"
               [ -n "$f" ] && [ -f "$f" ] && { echo "$f"; return 0; }
               return 2 ;;
    esac
    local -a hits=()
    shopt -s nullglob nocaseglob
    for f in "$FINAL"/$pat; do hits+=("$f"); done
    shopt -u nullglob nocaseglob
    if [ "${#hits[@]}" -eq 1 ]; then
        local f="${hits[0]}"
        local sz; sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
        # A real checkpoint here is >=29 MB. Anything tiny is a git-lfs pointer
        # checked out as text (killarney has no git-lfs), which would load as a
        # parse error rather than a model.
        if [ "$sz" -lt 1000000 ]; then
            echo "  $name: $f is only ${sz} bytes -- a git-lfs POINTER, not weights:" >&2
            head -c 200 "$f" | sed 's/^/        /' >&2; echo >&2
            return 3
        fi
        echo "$f"; return 0
    fi
    if [ "${#hits[@]}" -eq 0 ]; then
        echo "  $name: no file in $FINAL matches '$pat'" >&2; return 1
    fi
    echo "  $name: '$pat' is AMBIGUOUS -- ${#hits[@]} matches:" >&2
    printf '      %s\n' "${hits[@]}" >&2
    return 1
}

# ── Stage 1: what are these files, really? ───────────────────────────────────
source "$DEPLOY_ROOT/venv/bin/activate"
PATHS=(); MISSING=0; POINTER=0; declare -A CK=()
echo "--- resolving checkpoints in $FINAL"
ls -1 "$FINAL" | sed 's/^/      /'
echo
for r in "${ROWS[@]}"; do
    IFS='|' read -r name hm idx env pat <<<"$r"
    ck="$(resolve "$pat" "$name")"; rc=$?
    if [ "$rc" = 2 ]; then
        printf '%-22s UNRESOLVED  (set CKPT_CPFN_V0=<path>)\n' "$name"; continue
    fi
    if [ "$rc" = 3 ]; then
        printf '%-22s LFS POINTER (needs the real file copied over)\n' "$name"
        POINTER=1; continue
    fi
    if [ "$rc" != 0 ] || [ -z "$ck" ]; then
        printf '%-22s MISSING\n' "$name"; MISSING=1; continue
    fi
    printf '%-22s %s\n' "$name" "$ck"
    CK[$name]="$ck"; PATHS+=("$ck")
done
echo
# Compare against the reference DoPFN-bb weights: if the repro joint2d head has a
# different bin count, the bb harness would decode its logits on the wrong grid.
REF="$REPO/Required_checkpoints/dopfn_bb_j10_step_150000.pt"
[ -f "$REF" ] && PATHS+=("$REF")
[ "${#PATHS[@]}" -gt 0 ] && python -u "$REPO/benchmarks/inspect_ckpt.py" "${PATHS[@]}"
echo
# Several of these files share a byte size exactly (same architecture, same
# step), so size is not evidence of identity either way. Hash them: two models
# that are supposed to differ and hash the same would mean a copy went wrong.
echo "--- sha256 (same-size files must still differ)"
if [ "${#PATHS[@]}" -gt 0 ]; then
    sha256sum "${PATHS[@]}" 2>/dev/null | sed 's/^/      /'
    dup=$(sha256sum "${PATHS[@]}" 2>/dev/null | awk '{print $1}' | sort | uniq -d | wc -l | tr -d ' ')
    [ "$dup" != 0 ] && echo "  WARNING: $dup hash(es) appear more than once -- two 'different' models are the same file" >&2
fi
echo
if [ "$MISSING" = 1 ]; then
    echo "FATAL: some checkpoints are missing -- fix the paths before submitting." >&2
    exit 1
fi
[ "$POINTER" = 1 ] && echo "NOTE: skipping the lfs-pointer model(s) above; the rest proceed."
[ "$SUBMIT" = 1 ] || { echo "Inspection only. Re-run with --submit to queue the dump cells."; exit 0; }

# ── Stage 2: one dump cell per model, per benchmark ──────────────────────────
cd "$DEPLOY_ROOT" || exit 1
for r in "${ROWS[@]}"; do
    IFS='|' read -r name hm idx env pat <<<"$r"
    ck="${CK[$name]:-}"; [ -n "$ck" ] || continue
    RC_TASK=$(( idx * 5 + 0 ))            # IHDP
    CS_TASK=$(( idx * 8 + 2 ))            # d = 5
    echo "== $name  (harness $hm, RC task $RC_TASK, CS task $CS_TASK)"
    env "$env=$ck" OUT_ROOT="$SMOKE/$name/rc" \
        sbatch --array="$RC_TASK" --job-name="sm-rc-$name" \
        "$REPO/benchmarks/cluster/submit_realcause_density_unified.sbatch" \
        | sed 's/^/   RC /'
    env "$env=$ck" OUT_ROOT="$SMOKE/$name/cs" SHIFT=0 DENSITY_DUMP=1 \
        sbatch --array="$CS_TASK" --job-name="sm-cs-$name" \
        "$REPO/benchmarks/cluster/submit_cs_dvar_density.sbatch" \
        | sed 's/^/   CS /'
done
echo
echo "Queued. When they finish:  bash $REPO/benchmarks/cluster/check_smoke.sh"
