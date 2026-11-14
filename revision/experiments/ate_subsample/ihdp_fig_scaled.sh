#!/bin/bash
# The IHDP illustration (fig:ot-illustration) with the scaled ATE density: the per-query ITE densities and the ATE
# density (barycenter rescaled by each model's in-prior s; single-outcome models at rho = 0) of IHDP realization 0,
# from the RealCause dumps of rc_dumps.sbatch. Same models and style as D262k/10b_density_fig_paper.sh.
# Writes R-PFN/revision/results/ate_ot/figures/ihdp_r0_ate_2x4_paper.{png,pdf} and pushes. CPU, a few minutes:
#   cd $KIT && git -C R-PFN pull && sbatch --account=aip-rgrosse --time=0:30:00 --cpus-per-task=2 --mem=16G \
#        -o logs_ate/ihdp_fig_%j.out --wrap "bash R-PFN/revision/experiments/ate_subsample/ihdp_fig_scaled.sh"
set -uo pipefail
KIT="${KIT:-$PWD}"; REPO="$KIT/R-PFN"
source "$KIT/venv/bin/activate"; export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks:$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
D="$SCRATCH/ate_rc_dumps"; V="$SCRATCH/ihdp_ate_scaled_view"; rm -rf "${V:?}"; mkdir -p "$V"
ln -s "$D/dopfn_native/dopfn_native" "$V/dopfn_native"
ln -s "$D/dopfn_joint2d_262k/dopfn_native" "$V/dopfn_repro_joint2d"
ln -s "$D/uwyk1d/uwyk1d" "$V/uwyk1d"
ln -s "$D/graph2d/graph2d" "$V/graph2d"
ln -s "$D/cpfn1d/cpfn1d" "$V/cpfn1d"
ln -s "$D/cpfn2d/cpfn2d_pooled" "$V/cpfn2d"
for m in "$V"/*; do ls "$m/IHDP" >/dev/null || { echo "missing $m/IHDP"; exit 1; }; done
F="$REPO/revision/results/ate_ot/figures"; mkdir -p "$F"
python -u "$REPO/benchmarks/plots/plot_ate_density_2x4.py" --root "$V" --dataset IHDP --realization 0 \
  --repo "$REPO" --paper --xlim -10 10 --scale-dir "$SCRATCH/ate_bary" --out "$F/ihdp_r0_ate_2x4_paper.png" || exit 1
cd "$REPO" && git add "$F/ihdp_r0_ate_2x4_paper.png" "$F/ihdp_r0_ate_2x4_paper.pdf" \
  && git commit -q -m "ate_ot: IHDP illustration with the scaled ATE density" && git pull -q --rebase --autostash origin main \
  && git push -q origin main && echo pushed
