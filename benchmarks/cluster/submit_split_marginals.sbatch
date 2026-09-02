#!/bin/bash
#SBATCH --job-name=split-marginals
#SBATCH --account=aip-rgrosse
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/split_marginals_%j.out
#SBATCH --error=logs/split_marginals_%j.err

# Re-render a single (realization, query) IHDP example as 5 BARE per-method
# marginal panels (Do-PFN-bb, Do-PFN, fn=50, UWYK-NoAnc, UWYK-FullAnc).
# Bare = real axes (ticks + numbers) kept, but no title / axis labels / legend.
# Env vars (all have defaults):
#   R          (default 0)   realization to plot
#   Q          (default 37)  query index to plot
#   OUT        (default $DEPLOY_ROOT/ihdp_r${R}_q${Q})  output prefix
#   LEGEND     (unset)       set to 1 to keep the legend on the Do-PFN-bb panel

set -euo pipefail
: "${DEPLOY_ROOT:?DEPLOY_ROOT must be set}"
: "${R:=0}"
: "${Q:=37}"
: "${OUT:=$DEPLOY_ROOT/ihdp_r${R}_q${Q}}"

cd "$DEPLOY_ROOT/R-PFN"
# Activate the deploy venv (created by benchmarks/cluster/setup_env.sh).
if   [[ -f "$DEPLOY_ROOT/venv/bin/activate" ]]; then . "$DEPLOY_ROOT/venv/bin/activate"
elif [[ -f venv/bin/activate ]];               then . venv/bin/activate
elif [[ -f .venv/bin/activate ]];              then . .venv/bin/activate; fi

LEGEND_FLAG=""
if [[ "${LEGEND:-0}" == "1" ]]; then LEGEND_FLAG="--split-legend"; fi

echo "[$(date)] plotting r=$R q=$Q -> ${OUT}_marginals_*.png"
python benchmarks/plots/find_and_plot_example.py \
    --dataset ihdp \
    --bb-shards         "$DEPLOY_ROOT/ihdp_l2_dopfnbb_j10_s150k_2dmarg_B1000K1.r*.npz" \
    --dopfn-uwyk-shards "$DEPLOY_ROOT/ihdp_l2_dopfn_uwyk_strict.r*.npz" \
    --fn50-shards       "$DEPLOY_ROOT/ihdp_l2_fn50_2dmarg_B1000K1.r*.npz" \
    --repo "$DEPLOY_ROOT/R-PFN" --causalpfn "$DEPLOY_ROOT/external/causalpfn" \
    --checkpoint-dopfn-bb "$DEPLOY_ROOT/R-PFN/Required_checkpoints/dopfn_bb_j10_step_150000.pt" \
    --mode winner --r "$R" --q "$Q" \
    --split $LEGEND_FLAG \
    --out "$OUT"

echo "[$(date)] DONE"
