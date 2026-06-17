#!/bin/bash
#SBATCH --account=def-zhijing
#SBATCH --job-name=vanilla-uwyk
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/vanilla_%j.out
#SBATCH --error=logs/vanilla_%j.err

# Control experiment: run UWYK's vanilla InterventionalDataset for
# 25,000 iterations on the cluster. NO modifications, NO mutation
# of private state, NO paired propagation. Same SCM config we use,
# same preprocessing config as UWYK's complexmech_gcn_softatt.yaml.
#
# If this completes -> the bug is in OUR code (paired propagation /
#                      private state mutation). UWYK + cluster are fine.
# If this crashes  -> the bug is in UWYK / its dependencies / the
#                      cluster env, independent of our changes.

set -e
cd $SCRATCH/CFM-for-Decision-Makers
mkdir -p logs

export UWYK_SRC=$SCRATCH/g4cfm/src
if [ ! -d "$UWYK_SRC" ]; then
    echo "ERROR: UWYK_SRC not found at $UWYK_SRC"
    echo "Clone it on the login node:"
    echo "    cd \$SCRATCH && git clone --depth 1 https://github.com/ArikReuter/Graphs4CausalFoundationModels.git g4cfm"
    exit 1
fi

module purge
module load python/3.11
module load scipy-stack
source .venv/bin/activate

echo "=== Node info ==="
hostname
date
echo "CPUs available: $(nproc)"
echo "================="

export PYTHONUNBUFFERED=1
export N_ITERATIONS=25000

time python test_vanilla_uwyk.py
