#!/bin/bash
#SBATCH --job-name=cs-score
#SBATCH --output=logs_density/score_%A_%a.out
#SBATCH --error=logs_density/score_%A_%a.err
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-23
#
# CPU ONLY -- no --gres. Scores the cpfn-SCHEMA tree (dvar_cpfn) with
# UWYK_Fig3_4/cate_density_metrics.py: coverage95 / length / IS_0.05 for CATE
# and ATE, every model through the SAME scorer.
#
# 24 tasks = 3 shifts x 8 d; each task does all 6 case studies, both targets.
#   task/8 -> shift, task%8 -> d
#
# This is fast because the tree is already discrete: p(tau) is an
# anti-diagonal sum, not the n_y0 quadrature the tauC scorer needs.
#
# Env:
#   CPF      cpfn-schema tree   (default $DEPLOY_ROOT/results_case_study/dvar_cpfn)
#   MODELS   method dirs to score (default: the four unaffected by the
#            ANC_VARIANT bug; add the uwyk family once re-dumped)
#
#   export DEPLOY_ROOT=$SCRATCH/rpfn_bench_kit
#   sbatch case_study/density_eval/submit_score_cpfn_schema.sbatch

set -uo pipefail          # NOT -e: one bad cell must not abort the rest
mkdir -p logs_density

DEPLOY_ROOT="${DEPLOY_ROOT:?DEPLOY_ROOT required}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
CPF="${CPF:-$DEPLOY_ROOT/results_case_study/dvar_cpfn}"
CTX="${CTX:-1000}"
# MODELS="" uses cate_density_metrics' DEFAULT METHODS table, which carries
# the anc TAGS (uwyk1d-noanc / uwyk1d-v3a / graph2d-noanc / graph2d-v3a).
# Passing --method-dirs instead gives one untagged row per directory, so the
# ancestor variants collapse into whichever mode was written un-suffixed --
# wrong for any tree whose dirs use the canonical METHODS names.
MODELS="${MODELS-cpfn1d cpfn2d dopfn_native dopfn_bb}"

source "${VENV:-$DEPLOY_ROOT/venv}/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

SHIFTS=(shift0 shift+2 shift-2)
DS=(2 3 5 10 20 30 40 50)
S="${SHIFTS[$(( SLURM_ARRAY_TASK_ID / 8 ))]}"
D="${DS[$(( SLURM_ARRAY_TASK_ID % 8 ))]}"
ROOT="$CPF/$S/d$D/ctx$CTX"

CASES=(Observed_Confounder Backdoor_Criterion Observed_Mediator
       Observed_Mediator_and_Confounder Unobserved_Confounder
       Frontdoor_Criterion)

echo "[$(date '+%F %T')] task=$SLURM_ARRAY_TASK_ID  $S d$D  models=[$MODELS]"
[ -d "$ROOT" ] || { echo "no tree at $ROOT -- nothing to score"; exit 0; }

for T in cate ate; do
    echo "--- target=$T ---"
    MD=()
    [ -n "$MODELS" ] && MD=(--method-dirs $MODELS)
    python -u "$REPO/UWYK_Fig3_4/cate_density_metrics.py" \
        --root "$ROOT" --target "$T" \
        "${MD[@]}" \
        --dataset "${CASES[@]}" \
      || echo "WARN: $S d$D target=$T exited nonzero"
done
echo "[$(date '+%F %T')] done $S d$D"
