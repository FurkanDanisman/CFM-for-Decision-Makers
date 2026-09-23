#!/bin/bash
#SBATCH --job-name=tauC-malcT
#SBATCH --output=logs_density_tauC_malcT/eval_%A_%a.out
#SBATCH --error=logs_density_tauC_malcT/eval_%A_%a.err
#SBATCH --account=aip-rgrosse
#SBATCH --time=02:59:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --array=0-1

# Tier-C density eval, MALC VARIANT T: the log-concave fit sits in tau space,
# AFTER the anti-diagonal projection (2D heads) or the independence convolution
# (1D heads). Companion to submit_density_tauC_malc.sbatch, which is variant J
# and fits on the joint plane instead. Spec in density_eval_pipeline.md.
#
# NO GPU, NO CHECKPOINTS. This runs entirely off the prediction dumps the raw
# job wrote with SAVE_PREDICTIONS=1, so the raw, J and T arms are guaranteed to
# have scored identical logits. Per RUN_KILLARNEY.md: do not set --partition
# (Slurm routes by --time) and do not request --gres; CPU-only jobs are
# accepted on this cluster even though every partition is gpubase_*.
#
#   task 0 -> IHDP     task 1 -> ACIC
#
# PARALLELISM is over QUERIES, inside one realization at a time -- ACIC has a
# tenth the realizations and ~6.4x the queries each, so splitting over
# realizations would leave most of the pool idle there.
#
# T IS CHEAPER THAN J. MALC_1D's objective is closed form (the density is
# piecewise log-linear with knots at the data, so each piece integrates to
# d*exp(s)*sinh(t)/t), where MALC_2D solves a conic program over a Delaunay
# triangulation with Dunavant quadrature. Measured per 1D fit in malc_1d.py:
# B=100 -> 0.09 s, B=400 -> 0.24 s, B=1000 -> 1.6 s.
#
# MALC_B IS STILL THE KNOB THAT MATTERS, for the same reason as in J: the MLE
# is supported on the convex hull of its B synthetic points -- in 1D an
# interval -- and is exactly zero outside, so p(tau*) can vanish. Those queries
# fall back to the RAW density and are counted in n_fallback_<method>. Use
# B=100 for a first look, B=1000 for anything that goes in the paper.
#
# THE COARSE HEADS ARE THE ONES TO WATCH. Variant T hands MALC_1D only 2J-1
# atoms: 1999 for UWYK's K=1000 arms, 63 for a J=32 joint, but just 19 for
# DoPFN's J=10 joint. The J arm already measured that head at 11/75 fallbacks
# (14.7%) at B=100 with 100 bins to work from. Read n_fallback_* before
# trusting a row, and do not read a B=100 DoPFN joint number as a result.
#
# CausalPFN rows are new here -- variant J never supported that family, so they
# have no J counterpart unless the J arm is backfilled. Their 1D heads carry no
# tail model, so the smoothed interior IS the whole predictive and an
# out-of-hull tau* means a raw fallback rather than a mostly-tail density.
#
# MALC_K is 1 by default. In J that was partly forced -- malc_2d's
# _init_assignments_2d loops over every bin in Python and hangs on the
# 1000x1000 UWYK product -- but in 1D a mixture over a 1999-atom lattice is
# cheap, so a fixed K>1 is genuinely available here for the first time. Keep
# K=1 for the headline table so T and J differ in one variable only.
#
# A BIC SCAN OVER K IS NOT REACHABLE FROM HERE. MALC_1D selects K itself when
# K is None, but tau_smoother.SmootherConfig coerces with int(K), so None
# cannot pass through and MALC_K only accepts a fixed integer. Running the scan
# would mean either widening SmootherConfig or calling MALC_1D directly.
#
# THE SILENT-FAILURE CHECK IS THE `mass` COLUMN. smooth_tau_pmf returns a pmf
# normalised to 1, so the interior mass has to be multiplied back in; get that
# wrong and the density still scores, it just scores as a biased model. mass
# must land where the raw arm's does (1.0000 on UWYK r000, measured).
#
# Submit:  DUMPS_ROOT=results_density_tauC/<RAW_JOBID> \
#            sbatch benchmarks/cluster/submit_density_tauC_malcT.sbatch
# Smoke:   DUMPS_ROOT=... REAL_END=1 MALC_B=100 \
#            sbatch --array=0 benchmarks/cluster/submit_density_tauC_malcT.sbatch
# Then:    python benchmarks/eval_graph2d/summarize_density_tauC.py \
#            results_density_tauC_malcT/<JOBID>
#          (the same summarizer -- the output schema is identical, so the raw,
#           J and T tables are directly comparable row by row)

set -euo pipefail
mkdir -p logs_density_tauC_malcT

# THE DUMPS WERE NOT ALL MADE ON THIS CLUSTER. The recorded roots differ by
# family -- DoPFN under /home/lukez/..., UWYK under
# /home/lukez/projects/aip-rgrosse/lukez/..., CausalPFN under
# /project/6105522/lukez/... -- so set this to the checkout you are submitting
# from, not to whatever a dump's metadata says.
REPO="${REPO:-/home/lukez/projects/aip-rgrosse/lukez/CFM-for-Decision-Makers}"

# The raw job's OUT_ROOT, i.e. the directory holding <DATASET>/predictions/.
DUMPS_ROOT="${DUMPS_ROOT:?DUMPS_ROOT required -- the results_density_tauC/<JOBID> the RAW job wrote with SAVE_PREDICTIONS=1}"

export MALC_B="${MALC_B:-1000}"
export MALC_K="${MALC_K:-1}"
# Base offset on the per-query seeds. 0 reproduces the J arm's seed recipe;
# bump it to redraw the Beta jitter and see how much of a result is the draw.
export MALC_SEED="${MALC_SEED:-0}"
export MALC_N_TAU="${MALC_N_TAU:-12001}"   # matches TAU_CENTERS' resolution
export N_Y0="${N_Y0:-4096}"                # tails; MUST match the raw run
# uwyk | dopfn | causalpfn | all | auto. 'auto' scores whichever families the
# dump holds, which is what you want per-shard.
export MODEL_FAMILY="${MODEL_FAMILY:-auto}"
# Uniform grid DoPFN's 1D arms are rebinned onto before MALC sees them; MALC_1D
# raises on a non-uniform grid and DoPFN's borders are quantile spaced. No
# effect on UWYK or CausalPFN, whose bins are already uniform.
export DOPFN_MALC_BINS="${DOPFN_MALC_BINS:-1024}"
export N_WORKERS="${N_WORKERS:-$SLURM_CPUS_PER_TASK}"
export OUT_ROOT="${OUT_ROOT:-$REPO/results_density_tauC_malcT/$SLURM_ARRAY_JOB_ID}"

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

echo "[submit] $DATASET  dumps=$DUMPS  family=$MODEL_FAMILY  B=$MALC_B K=$MALC_K  dopfn_bins=$DOPFN_MALC_BINS  workers=$N_WORKERS"
echo "[submit] out=$OUT"

# NOTE: test_density_malc.py is NOT run here. Its gates are 2D-specific -- gate
# 1 pins the MALC_2D transpose, which has no analogue in a 1D fit, and gates
# 6-7 pin the DoPFN rebin as it reaches MALC_2D. The T arm's own silent-failure
# mode is the interior mass restoration, and the `mass` column catches it.

python -u benchmarks/eval_graph2d/eval_density_tauC_malcT.py
