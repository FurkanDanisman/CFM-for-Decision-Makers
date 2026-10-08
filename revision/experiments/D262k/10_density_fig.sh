#!/bin/bash
# Stage 10: only the IHDP density figure (ihdp_r0_ate_2x4.png) with the 262k Do-PFN 2D, no title.
# Each panel reads the dumps the calibration tables score (E1_E8_rescoring/submit_hist.sbatch):
#   rc_dens_uni: dopfn_native, uwyk1d, graph2d, cpfn1d;  rc_dens_eta0: cpfn2d_pooled;  Do-PFN 2D: the 262k dumps.
#   cd $KIT && sbatch --account=aip-rgrosse --time=1:00:00 \
#        --cpus-per-task=8 --mem=64G -o logs_hist/d262_dens_%j.out \
#        --wrap "bash R-PFN/revision/experiments/D262k/10_density_fig.sh"
set -uo pipefail
KIT="${KIT:-$PWD}"; source "$KIT/R-PFN/revision/experiments/D262k/common.sh"
source "$KIT/venv/bin/activate"; export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks:$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
F="$REPO/revision/results/$TAG/figures"; mkdir -p "$F"
V="$SCRATCH/ihdp_dens_d262_view"; rm -rf "$V"; mkdir -p "$V"
for m in dopfn_native uwyk1d graph2d cpfn1d; do ln -s "$SCRATCH/rc_dens_uni/$m" "$V/$m"; done
ln -s "$SCRATCH/rc_dens_eta0/cpfn2d_pooled" "$V/cpfn2d_eta0"
ln -s "$RC_OUT/dopfn_native" "$V/dopfn_repro_joint2d"
for m in "$V"/*; do ls "$m/IHDP" >/dev/null || { echo "missing $m/IHDP"; exit 1; }; done
python -u "$REPO/benchmarks/plots/plot_ate_density_2x4.py" --root "$V" --dataset IHDP --realization 0 \
  --repo "$REPO" --out "$F/ihdp_r0_ate_2x4.png" || exit 1
cd "$REPO" && git add revision/results/$TAG/figures/ihdp_r0_ate_2x4.png \
  && git commit -q -m "$TAG: IHDP density figure with the 262k Do-PFN 2D" && git push -q && echo pushed
