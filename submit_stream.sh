#!/bin/bash
#SBATCH --account=def-zhijing
#SBATCH --job-name=cfm-stream
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/stream_%j.out
#SBATCH --error=logs/stream_%j.err

# 1000-step streaming run on 1 H100. Replaces the cached 5-task loop with
# fresh SCMs sampled per step via PairedInterventionalDataset. This is the
# first test of the "real" data pipeline; not a real training run, just
# a check that streaming throughput is acceptable and the loss curve looks
# different from the memorization plateau we saw with cached data.

set -e
cd $SCRATCH/CFM-for-Decision-Makers
mkdir -p logs

# UWYK source path — required for SCMSampler / BinarizingMechanism
export UWYK_SRC=$SCRATCH/g4cfm/src
if [ ! -d "$UWYK_SRC" ]; then
    echo "Cloning UWYK source to $UWYK_SRC ..."
    git clone --depth 1 https://github.com/ArikReuter/Graphs4CausalFoundationModels.git $SCRATCH/g4cfm
fi

module purge
module load python/3.11
module load scipy-stack
source .venv/bin/activate

echo "=== Node info ==="
hostname
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "================="

export STREAM_DATA=1
export STREAM_WORKERS=4
export STREAM_WARMUP=4
export N_STEPS=1000
export LOG_EVERY=50

time python train_cfm_small.py
