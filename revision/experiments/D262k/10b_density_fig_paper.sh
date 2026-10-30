#!/bin/bash
# The IHDP ATE-density figure (fig:ot-illustration) in the paper's figure style: the same dumps and models as
# 10_density_fig.sh run from stage 19 (Do-PFN 2D = 262k, temperature off), drawn with plot_ate_density_2x4.py --paper.
# Writes revision/results/d262t1/figures/ihdp_r0_ate_2x4_paper.{png,pdf} and pushes.
#   cd $KIT && git -C R-PFN pull --rebase && sbatch --account=aip-rgrosse --time=1:00:00 --cpus-per-task=4 --mem=32G \
#        -o logs_hist/density_paper_%j.out --wrap "bash R-PFN/revision/experiments/D262k/10b_density_fig_paper.sh"
set -uo pipefail
export D262_NAME=dopfn_repro_joint2d_262k_t1 DOPFN_SOFTMAX_TEMP=0
KIT="${KIT:-$PWD}"; source "$KIT/R-PFN/revision/experiments/D262k/common.sh"
source "$KIT/venv/bin/activate"; export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks:$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
F="$REPO/revision/results/$TAG/figures"; mkdir -p "$F"
V="$SCRATCH/ihdp_dens_d262_view_paper"; rm -rf "$V"; mkdir -p "$V"
for m in dopfn_native uwyk1d graph2d cpfn1d; do ln -s "$SCRATCH/rc_dens_uni/$m" "$V/$m"; done
ln -s "$SCRATCH/rc_dens_eta0/cpfn2d_pooled" "$V/cpfn2d_eta0"
ln -s "$RC_OUT/dopfn_native" "$V/dopfn_repro_joint2d"
for m in "$V"/*; do ls "$m/IHDP" >/dev/null || { echo "missing $m/IHDP"; exit 1; }; done
python -u "$REPO/benchmarks/plots/plot_ate_density_2x4.py" --root "$V" --dataset IHDP --realization 0 \
  --repo "$REPO" --paper --xlim -10 10 --out "$F/ihdp_r0_ate_2x4_paper.png" || exit 1
cd "$REPO" && git add "revision/results/$TAG/figures/ihdp_r0_ate_2x4_paper.png" "revision/results/$TAG/figures/ihdp_r0_ate_2x4_paper.pdf" \
  && git commit -q -m "$TAG: IHDP ATE-density figure in the paper style" && git pull -q --rebase --autostash origin main \
  && git push -q origin main && echo pushed
