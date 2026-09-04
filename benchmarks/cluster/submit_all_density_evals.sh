#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────
# submit_all_density_evals.sh
# ────────────────────────────────────────────────────────────────────────────
#
# End-to-end launcher for the bivariate-Y-noise density-eval pipeline.
#
# Steps (each chained via afterok):
#   1. Regenerate DoPFN's 6 case-study pkls with rho=0.2 Y-noise
#      → $OUT_DIR/prior_sampling_rho02/<Case>/*.pkl
#   2. Rerun each of the 8 models on the new pkls via the existing
#      benchmarks/cluster/submit_eval_scm_case_studies.sbatch, pointing
#      DOPFN_DATA_ROOT at the new pkls.
#   3. Run the density-eval sbatch (one array per model × 6 cases) on the
#      rerun shard dirs.
#
# Requirements:
#   - CKPT_<MODELTAG> env vars pre-set (one per model) with the checkpoint
#     path; see the list under "MODELS" below. Missing ones are skipped
#     with a warning.
#   - The existing submit_eval_scm_case_studies.sbatch is NOT modified;
#     however for step-3 density metrics to work each per-model eval
#     script MUST emit per-realization NPZs containing the density-required
#     keys (edges, p_y0_scaled, p_y1_scaled, y_shift, y_scale, [p_joint_scaled]).
#     See TODO list emitted by density_eval.py — this is the single
#     outstanding pipeline change needed (add ~10 lines to each of the
#     6 eval scripts under benchmarks/eval_scm_case_studies/ + the two
#     under benchmarks/eval_causalpfn2d/ and benchmarks/eval_graph2d/).
#
# Env-var contract:
#   DEPLOY_ROOT     working root that has ./venv, ./R-PFN, ./external/...
#   OUT_DIR         where new pkls land (default: $DEPLOY_ROOT/external/dopfn/data)
#   Y_NOISE_CORR    default 0.2
#   RERUN_ROOT      per-model shard dir root (default: $DEPLOY_ROOT/results/scm_cs_rho02_reruns)
#   DENSITY_ROOT    density-eval results root (default: $DEPLOY_ROOT/results/scm_cs_density_eval_rho02)
#
# Model checkpoint env vars (skipped if unset):
#   CKPT_CPFN2D_POOLED   / CKPT_CPFN1D_PERARM  / CKPT_FN50
#   CKPT_GRAPH2D         / CKPT_UWYK
#   CKPT_DOPFN_BB        / CKPT_DOPFN_NATIVE  (dopfn_native uses artifacts,
#                                             set to "artifacts")
#
set -euo pipefail

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
DOPFN_ROOT="${DOPFN_ROOT:-$DEPLOY_ROOT/external/dopfn}"
OUT_DIR="${OUT_DIR:-$DOPFN_ROOT/data}"
Y_NOISE_CORR="${Y_NOISE_CORR:-0.2}"
RERUN_ROOT="${RERUN_ROOT:-$DEPLOY_ROOT/results/scm_cs_rho02_reruns}"
DENSITY_ROOT="${DENSITY_ROOT:-$DEPLOY_ROOT/results/scm_cs_density_eval_rho02}"
N_PER_CASE="${N_PER_CASE:-100}"
SEQ_LEN="${SEQ_LEN:-500}"

# Cases handled by the array job (kept in sync with array=0-5 in the sbatch)
CASES=(Observed_Confounder Observed_Mediator Observed_Mediator_and_Confounder \
       Unobserved_Confounder Frontdoor_Criterion Backdoor_Criterion)

# Model tags — one entry per invocation of submit_eval_scm_case_studies.sbatch.
# Each row: TAG  MODEL  EXTRA_EXPORTS
declare -a MODELS=(
    "cpfn2d_pooled     cpfn2d     STD_MODE=pooled"
    "cpfn1d_perarm     cpfn1d     STD_MODE=per_arm"
    "fn50              fn50       "
    "graph2d_noanc     graph2d    GRAPH2D_ANC_MODE=noanc"
    "graph2d_full      graph2d    GRAPH2D_ANC_MODE=anc"
    "uwyk_noanc        uwyk       UWYK_ANC_MODE=noanc"
    "uwyk_anc          uwyk       UWYK_ANC_MODE=anc"
    "dopfn_bb          dopfn_bb   "
    "dopfn_native      dopfn_native "
)

mkdir -p "$RERUN_ROOT" "$DENSITY_ROOT"

# ── 1. Regen job ────────────────────────────────────────────────────────────
regen_out=$(sbatch \
    --parsable \
    --export=ALL,DEPLOY_ROOT="$DEPLOY_ROOT",REPO="$REPO",DOPFN_ROOT="$DOPFN_ROOT",OUT_DIR="$OUT_DIR",Y_NOISE_CORR="$Y_NOISE_CORR",N_PER_CASE="$N_PER_CASE",SEQ_LEN="$SEQ_LEN" \
    "$REPO/benchmarks/cluster/submit_regen_case_studies.sbatch")
regen_jid="$regen_out"
echo "[submit] regen job id: $regen_jid"

DATA_ROOT_NEW="$OUT_DIR/prior_sampling_rho02"

# ── 2. Per-model reruns, all dependent on regen ─────────────────────────────
declare -a rerun_jids=()
for row in "${MODELS[@]}"; do
    IFS=' ' read -r TAG MODEL EXTRA <<< "$row"
    ckpt_var="CKPT_$(echo "$TAG" | tr '[:lower:]' '[:upper:]')"
    ckpt_val="${!ckpt_var:-}"
    if [ -z "$ckpt_val" ]; then
        echo "[submit] skipping $TAG (no $ckpt_var set)"
        continue
    fi
    out_root="$RERUN_ROOT/$TAG"
    export_str="ALL,DEPLOY_ROOT=$DEPLOY_ROOT,REPO=$REPO,DOPFN_ROOT=$DOPFN_ROOT,MODEL=$MODEL,CKPT=$ckpt_val,OUT_ROOT=$out_root,DOPFN_DATA_ROOT=$DATA_ROOT_NEW"
    if [ -n "$EXTRA" ]; then
        export_str="$export_str,$(echo "$EXTRA" | tr ' ' ',')"
    fi
    jid=$(sbatch \
        --parsable \
        --dependency="afterok:$regen_jid" \
        --export="$export_str" \
        "$REPO/benchmarks/cluster/submit_eval_scm_case_studies.sbatch")
    echo "[submit] rerun $TAG  job $jid  → $out_root"
    rerun_jids+=("$jid")
done

# ── 3. Density-eval per model, dependent on ALL reruns ──────────────────────
if [ ${#rerun_jids[@]} -eq 0 ]; then
    echo "[submit] no reruns launched; skipping density-eval chain"
    exit 0
fi
dep_str=$(IFS=:; echo "${rerun_jids[*]}")
for row in "${MODELS[@]}"; do
    IFS=' ' read -r TAG MODEL EXTRA <<< "$row"
    ckpt_var="CKPT_$(echo "$TAG" | tr '[:lower:]' '[:upper:]')"
    ckpt_val="${!ckpt_var:-}"
    if [ -z "$ckpt_val" ]; then continue; fi
    shard_root="$RERUN_ROOT/$TAG"
    out_root="$DENSITY_ROOT/$TAG"
    dens_jid=$(sbatch \
        --parsable \
        --dependency="afterok:$dep_str" \
        --export="ALL,DEPLOY_ROOT=$DEPLOY_ROOT,REPO=$REPO,DOPFN_ROOT=$DOPFN_ROOT,MODEL=$TAG,SHARD_ROOT=$shard_root,OUT_ROOT=$out_root,DOPFN_DATA_ROOT=$DATA_ROOT_NEW" \
        "$REPO/benchmarks/cluster/submit_density_eval.sbatch")
    echo "[submit] density $TAG  job $dens_jid  → $out_root"
done

echo "[submit] all jobs queued. dep chain: regen($regen_jid) → reruns(${rerun_jids[*]}) → density"
