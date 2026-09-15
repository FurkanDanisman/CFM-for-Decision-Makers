#!/bin/bash
# Complete the cen3 density dumps to 100 realizations/cell with the FEWEST jobs:
# ONE job per (shift, model), each looping all d at a single context (N=1000).
#   3 shifts x 6 raw-density models = 18 jobs.
#
# Each job is resumable (see submit_density_alld.sbatch): on a wall-clock kill
# only the in-flight realization is lost, and re-running THIS launcher re-submits
# jobs that pick up exactly where they stopped. So the safe pattern is:
#     bash 07_backfill_density_1000.sh          # submit 18
#     ... wait; some hit the 24h wall ...
#     bash 07_backfill_density_1000.sh          # re-submit; backfills the rest
# Nothing already dumped is recomputed.
#
#   export DEPLOY_ROOT=$SCRATCH/rpfn_bench_kit
#   bash $DEPLOY_ROOT/R-PFN/case_study/density_eval/07_backfill_density_1000.sh
#
# Env:
#   SHIFTS    default "0 +2 -2"      MODELS  default the six raw-density models
#   DS        default "2 3 5 10 20 30 40 50"
#   CTX       default 1000           REAL_END default 100
#   RES       default $DEPLOY_ROOT/results_case_study/dvar  (MUST match where the
#             existing dumps live, so resume sees them)
#   DRY_RUN=1 print what would be submitted, submit nothing
set -euo pipefail

DEPLOY_ROOT="${DEPLOY_ROOT:?DEPLOY_ROOT required}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
HERE="$REPO/case_study/density_eval"
SB="$HERE/submit_density_alld.sbatch"
DATA="${DATA:-$DEPLOY_ROOT/case_study_data/d_variation}"
RES="${RES:-$DEPLOY_ROOT/results_case_study/dvar}"
SHIFTS="${SHIFTS:-0 +2 -2}"
DS="${DS:-2 3 5 10 20 30 40 50}"
CTX="${CTX:-1000}"
REAL_END="${REAL_END:-100}"
MODELS="${MODELS:-graph2d uwyk uwyk_v3a uwyk_noanc dopfn_native dopfn_bb}"
# Wall clock passed to sbatch, overriding the 23:59:00 in submit_density_alld.
# A 24 h request queues behind everything; with SCORE_INLINE=0 a realization
# takes ~0.5 s, so a single-case pilot needs minutes and a short limit gets
# backfill-scheduled almost immediately. Raise it for the full grid
# (8 d x 6 cases x 100 real ~ 40 min, so 02:00:00 is ample).
WALLTIME="${WALLTIME:-23:59:00}"

[ -f "$SB" ] || { echo "FATAL: missing $SB" >&2; exit 1; }
[ -d "$DATA" ] || { echo "FATAL: data root not found: $DATA" >&2; exit 1; }

export DOPFN_ROOT="${DOPFN_ROOT:-$DEPLOY_ROOT/external/dopfn}"
export CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
export UWYK="${UWYK:-$DEPLOY_ROOT/external/uwyk}"
GRAPH2D_CKPT="${GRAPH2D_CKPT:-$REPO/Required_checkpoints/graph2d_step_50000.pt}"
DOPFNBB_CKPT="${DOPFNBB_CKPT:-$REPO/Required_checkpoints/dopfn_bb_j10_step_150000.pt}"
UWYK_DIR="${UWYK_DIR:-$UWYK/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0}"

# graph2d + every uwyk variant load the UWYK checkpoint (one process emits both
# heads), so fail fast if it's absent.
case " $MODELS " in *" graph2d "*|*" uwyk "*|*" uwyk_v3a "*|*" uwyk_noanc "*)
    for f in "$UWYK_DIR/best_model.pt" "$UWYK_DIR/best_model_config.yaml"; do
        [ -f "$f" ] || { echo "FATAL: UWYK input missing: $f (override UWYK_DIR)" >&2; exit 1; }
    done ;;
esac

# Overridable so a single case study can be piloted across all models
# before committing the full grid.
CASES="${CASES:-Observed_Confounder,Backdoor_Criterion,Observed_Mediator,Observed_Mediator_and_Confounder,Unobserved_Confounder,Frontdoor_Criterion}"

n=0
for s in $SHIFTS; do
  for m in $MODELS; do
    # Reset per-model env, then set what this model needs.
    unset MODEL_FAMILY UWYK_CKPT UWYK_CFG CKPT DOPFN_JOINT_CKPT ANC_VARIANT
    case "$m" in
      graph2d)      MODEL_FAMILY=uwyk; ANC_VARIANT=full ;;
      uwyk)         MODEL_FAMILY=uwyk; ANC_VARIANT=full ;;
      uwyk_v3a)     MODEL_FAMILY=uwyk; ANC_VARIANT=paper_anc ;;
      uwyk_noanc)   MODEL_FAMILY=uwyk; ANC_VARIANT=noanc ;;
      dopfn_native) MODEL_FAMILY=dopfn ;;
      dopfn_bb)     MODEL_FAMILY=dopfn; DOPFN_JOINT_CKPT="$DOPFNBB_CKPT" ;;
      *) echo "FATAL: unknown model $m" >&2; exit 2 ;;
    esac
    # SCORE_INLINE is forwarded explicitly rather than relying on sbatch's
    # --export=ALL default: SCORE_INLINE=0 turns each eval into a dump-only
    # run (no per-query 4096-point quadrature), which is the difference
    # between ~12 h and minutes per cell. See eval_density_tauC.py.
    export REPO HERE DEPLOY_ROOT DATA RES MODEL="$m" SHIFT="$s" CTX DS REAL_END \
           CASES SCM_N_QUERY="${SCM_N_QUERY:-100}" \
           SCORE_INLINE="${SCORE_INLINE:-1}" \
           DOPFN_ROOT CAUSALPFN UWYK MODEL_FAMILY ANC_VARIANT
    case "$m" in
      graph2d|uwyk|uwyk_v3a|uwyk_noanc)
        export UWYK_CKPT="$UWYK_DIR/best_model.pt" \
               UWYK_CFG="$UWYK_DIR/best_model_config.yaml" CKPT="$GRAPH2D_CKPT" ;;
      dopfn_bb) export DOPFN_JOINT_CKPT ;;
    esac

    if [ "${DRY_RUN:-0}" = 1 ]; then
      echo "[dry] shift$s $m  family=$MODEL_FAMILY anc=${ANC_VARIANT:-}" \
           "ds=[$DS] score_inline=${SCORE_INLINE:-1}"
    else
      sbatch --time="$WALLTIME" --job-name="densAll-${m}-s${s}" "$SB" >/dev/null
      echo "submitted shift$s $m"
    fi
    n=$((n+1))
  done
done
echo
echo "[07] $n jobs (shifts=[$SHIFTS] x models=[$MODELS]) ctx=$CTX real_end=$REAL_END -> $RES"
echo "Re-run this SAME command after any timeouts; it resumes, never recomputes."
echo
echo "When all cells reach $REAL_END, aggregate with:"
echo "  python $HERE/dsweep_density_report.py --root $RES --data-root $DATA \\"
echo "      --combine-shifts shift0 shift+2 shift-2 --combine-label cen3 \\"
echo "      --only-n $CTX --min-n 60 --jobs 32 \\"
echo "      --out $DEPLOY_ROOT/results_case_study/dvar/combined_cen3_density.csv"
