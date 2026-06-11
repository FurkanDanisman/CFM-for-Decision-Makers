#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --job-name=cfm-smoke
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/lukez/projects/aip-rgrosse/lukez/CFM-for-Decision-Makers/logs/smoke_%j.out
#SBATCH --error=/home/lukez/projects/aip-rgrosse/lukez/CFM-for-Decision-Makers/logs/smoke_%j.err

# Smoke test: 200-step demo run of train_cfm_small.py on 1 GPU.
# Verifies the pipeline (UWYK body + 2D BarDistribution head + paired data)
# runs end-to-end on GPU with the cluster's torch wheels.

set -e
PROJ_DIR="/home/lukez/projects/aip-rgrosse/lukez/CFM-for-Decision-Makers"
cd "$PROJ_DIR"
mkdir -p logs

# venv was built by uv against the module python/3.11 (3.11.5) and is
# self-contained (include-system-site-packages=false), so all deps live inside
# .venv. Only the base python module is needed; scipy-stack would be ignored.
module load python/3.11
source .venv/bin/activate

echo "=== Node info ==="
hostname
nvidia-smi
echo "=== Python env ==="
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
echo "================="

python train_cfm_small.py
