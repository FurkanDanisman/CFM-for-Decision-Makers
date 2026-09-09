#!/bin/bash
#SBATCH --job-name=tauC-malc
#SBATCH --output=logs_density_tauC_malc/eval_%A_%a.out
#SBATCH --error=logs_density_tauC_malc/eval_%A_%a.err
#SBATCH --account=aip-rgrosse
#SBATCH --time=02:59:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --array=0-1

# Tier-C density eval, MALC arm: p(tau | x) with region 0 smoothed by a K=1
# log-concave MLE. Companion to submit_density_tauC.sbatch (the RAW arm);
# spec in density_eval_pipeline.md.
#
# NO GPU, NO CHECKPOINTS. This runs entirely off the prediction dumps the raw
# job wrote with SAVE_PREDICTIONS=1, so it needs neither model, and the two
# arms are guaranteed to have scored identical logits. Per RUN_KILLARNEY.md:
# do not set --partition (Slurm routes by --time) and do not request --gres;
# CPU-only jobs are accepted on this cluster even though every partition is
# gpubase_*.
#
#   task 0 -> IHDP   (100 realizations, ~75 queries each)
#   task 1 -> ACIC   (10 realizations, ~481 queries each)
#
# PARALLELISM is over QUERIES, inside one realization at a time. Each query is
# ~2 s for all three methods at B=100, so a 32-worker pool stays balanced;
# splitting over realizations instead would leave ACIC's 10 shards idling 22
# of the 32 cores.
#
# Cost, measured on a login node, sequential, B=100:
#   141 s per IHDP realization (75 queries x 3 methods, ~0.6 s/method/query)
# At B=1000 add the fit cost, ~0.9 s/method/query, so ~340 s/realization.
# With 32 workers: IHDP ~20 min at B=100, ~45 min at B=1000. ACIC is ~6.4x
# the queries per realization but a tenth the realizations, so similar.
#
# MALC_B IS THE ONE KNOB THAT MATTERS. The log-concave MLE is supported on the
# convex hull of its B synthetic points and is exactly zero outside, so p(tau*)
# can be 0 and the NLL infinite. Those queries fall back to the RAW density and
# are counted in n_fallback_<method>; the rate is B-dependent:
#   B=100    0.8% (joint)  2.7% (uwyk_native)  0.8% (uwyk_matched)
#   B=1000   ~0            ~0                  ~0
#   B=3000   ~0, at 4x the cost of B=1000 -- no reason to go there
# Use B=100 for a first look, B=1000 for anything that goes in the paper.
#
# MALC_K is FORCED TO 1 in the eval and is not exposed here. It is well
# specified (the DGP's p(y0,y1|x) is a single bivariate Gaussian, which is
# log-concave) and K>1 is not usable at these resolutions anyway: malc_2d's
# _init_assignments_2d loops over every bin in Python, twice, which hangs on
# the 1000x1000 UWYK product, and its mixture E-step integrates each component
# on a fixed mesh that covers 0.5% of the bins there, so BIC cannot select K.
#
# Submit:  DUMPS_ROOT=results_density_tauC/<RAW_JOBID> \
#            sbatch benchmarks/cluster/submit_density_tauC_malc.sbatch
# Smoke:   DUMPS_ROOT=... REAL_END=1 MALC_B=100 \
#            sbatch --array=0 benchmarks/cluster/submit_density_tauC_malc.sbatch
# Then:    python benchmarks/eval_graph2d/summarize_density_tauC.py \
#            results_density_tauC_malc/<JOBID>
#          (the same summarizer -- the MALC output schema is identical, so the
#           raw and MALC tables are directly comparable row by row)

set -euo pipefail
mkdir -p logs_density_tauC_malc

REPO="/home/lukez/projects/aip-rgrosse/lukez/CFM-for-Decision-Makers"

# The raw job's OUT_ROOT, i.e. the directory holding <DATASET>/predictions/.
DUMPS_ROOT="${DUMPS_ROOT:?DUMPS_ROOT required -- the results_density_tauC/<JOBID> the RAW job wrote with SAVE_PREDICTIONS=1}"

export MALC_B="${MALC_B:-1000}"
export MALC_N_Y0="${MALC_N_Y0:-512}"     # interior; measured 1.1e-4 max err
export N_Y0="${N_Y0:-4096}"              # tails; MUST match the raw run
export N_WORKERS="${N_WORKERS:-$SLURM_CPUS_PER_TASK}"
export SAVE_FALLBACK_POINTS="${SAVE_FALLBACK_POINTS:-1}"
export OUT_ROOT="${OUT_ROOT:-$REPO/results_density_tauC_malc/$SLURM_ARRAY_JOB_ID}"

_RS_IN="${REAL_START:-}"
_RE_IN="${REAL_END:-}"
_DS_IN="${DATASET:-}"

case "$SLURM_ARRAY_TASK_ID" in
    0) DATASET=IHDP ;;
    1) DATASET=ACIC ;;
    *) echo "invalid task id: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac
DATASET="${_DS_IN:-$DATASET}"

export DATASET
export DUMPS="$DUMPS_ROOT/$DATASET/predictions"
export OUT="$OUT_ROOT/$DATASET"
[ -n "$_RS_IN" ] && export REAL_START="$_RS_IN"
[ -n "$_RE_IN" ] && export REAL_END="$_RE_IN"

[ -d "$DUMPS" ] || { echo "FATAL: no dumps at $DUMPS" >&2; exit 1; }
ls "$DUMPS"/*.npz >/dev/null 2>&1 || { echo "FATAL: $DUMPS holds no npz" >&2; exit 1; }

source "$REPO/.venv/bin/activate"
cd "$REPO"

echo "[submit] $DATASET  dumps=$DUMPS  B=$MALC_B K=1  workers=$N_WORKERS"
echo "[submit] out=$OUT"

# The model-free gates first: a wrong p_mat orientation returns p(-tau) with
# the mass still integrating to 1, so it cannot be caught downstream.
python -u benchmarks/eval_graph2d/test_density_malc.py || {
    echo "FATAL: MALC gates failed -- not running the eval" >&2; exit 1; }

python -u benchmarks/eval_graph2d/eval_density_tauC_malc.py
