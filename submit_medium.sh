#!/bin/bash
#SBATCH --account=def-zhijing
#SBATCH --job-name=cfm-medium
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/medium_%j.out
#SBATCH --error=logs/medium_%j.err

# 5,000-step run on 1 H100, cached 5 SCM tasks. Same model as smoke test,
# just longer. Sanity check that nothing blows up over a longer run, and
# gives a real wall-clock number to extrapolate to UWYK scale.

set -e
cd $SCRATCH/CFM-for-Decision-Makers
mkdir -p logs

module purge
module load python/3.11
module load scipy-stack
source .venv/bin/activate

echo "=== Node info ==="
hostname
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "================="

export N_STEPS=5000
export LOG_EVERY=100

time python train_cfm_small.py
