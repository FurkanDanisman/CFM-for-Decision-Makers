#!/bin/bash
#SBATCH --account=def-zhijing
#SBATCH --job-name=cfm-smoke
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err

# Trillium smoke test: 200-step demo run of train_cfm_small.py on 1 H100.
# Verifies the pipeline (UWYK body + 2D BarDistribution head + paired data)
# runs end-to-end on GPU with the cluster's torch wheels.

set -e
cd $SCRATCH/CFM-for-Decision-Makers
mkdir -p logs

module purge
module load python/3.11
module load scipy-stack
source .venv/bin/activate

echo "=== Node info ==="
hostname
nvidia-smi
echo "=== Python env ==="
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
echo "================="

python train_cfm_small.py
