#!/bin/bash
# Stage 19: the case-study by-d figures (PEHE, eps) and the IHDP density figure with the temperature-off
# Do-PFN 2D; written to revision/results/d262t1/figures and pushed.
#   cd $KIT && git -C R-PFN pull --rebase && sbatch --account=aip-rgrosse --time=3:00:00 --cpus-per-task=16 --mem=64G \
#        -o logs_hist/t1_fig_%j.out --wrap "bash R-PFN/revision/experiments/D262k/19_figures_t1.sh"
set -uo pipefail
export D262_NAME=dopfn_repro_joint2d_262k_t1 DOPFN_SOFTMAX_TEMP=0 IHDP_DENS_ROOT=/nonexistent   # density via stage 10
KIT="${KIT:-$PWD}"; D="$KIT/R-PFN/revision/experiments/D262k"
bash "$D/9_figures.sh"
bash "$D/10_density_fig.sh"
