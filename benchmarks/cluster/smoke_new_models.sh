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
FINAL=""
for c in "${FINAL_CKPTS:-}" "$REPO/final_checkpoints" "$DEPLOY_ROOT/final_checkpoints" \
         "$SCRATCH/final_checkpoints" "$HOME/final_checkpoints"; do
    [ -n "$c" ] && [ -d "$c" ] && { FINAL="$c"; break; }
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
ROWS=(
  "dopfn_repro_1d_J10|dopfn_native|0|DOPFN_CKPT|$FINAL/dopfn_repro_1d_J10_step150000.pt"
  "dopfn_repro_1d_J100|dopfn_native|0|DOPFN_CKPT|$FINAL/dopfn_repro_1d_J100_step150000.pt"
  "dopfn_repro_joint2d|dopfn_bb|1|CKPT_DOPFN_BB|$REPO/Required_checkpoints/dopfn_repro_joint2d_bb.pt"
  "cpfn1d_j32|cpfn1d|4|CKPT_CPFN1D|$FINAL/cpfn1d_j32_step50000.pt"
  "cpfn1d_botharms|cpfn1d|4|CKPT_CPFN1D|$FINAL/cpfn1d_botharms_step50000.pt"
  "cpfn_v0|cpfn1d|4|CKPT_CPFN1D|${CKPT_CPFN_V0:-}"
)

# ── Stage 1: what are these files, really? ───────────────────────────────────
source "$DEPLOY_ROOT/venv/bin/activate"
PATHS=(); MISSING=0
for r in "${ROWS[@]}"; do
    IFS='|' read -r name hm idx env ck <<<"$r"
    if [ -z "$ck" ]; then
        printf '%-22s UNRESOLVED  (set CKPT_%s)\n' "$name" "$(echo "$name" | tr '[:lower:]' '[:upper:]')"
        continue
    fi
    if [ -f "$ck" ]; then PATHS+=("$ck")
    else printf '%-22s MISSING     %s\n' "$name" "$ck"; MISSING=1; fi
done
echo
[ "${#PATHS[@]}" -gt 0 ] && python -u "$REPO/benchmarks/inspect_ckpt.py" "${PATHS[@]}"
echo
if [ "$MISSING" = 1 ]; then
    echo "FATAL: some checkpoints are missing -- fix the paths before submitting." >&2
    exit 1
fi
[ "$SUBMIT" = 1 ] || { echo "Inspection only. Re-run with --submit to queue the dump cells."; exit 0; }

# ── Stage 2: one dump cell per model, per benchmark ──────────────────────────
cd "$DEPLOY_ROOT" || exit 1
for r in "${ROWS[@]}"; do
    IFS='|' read -r name hm idx env ck <<<"$r"
    [ -n "$ck" ] || continue
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
