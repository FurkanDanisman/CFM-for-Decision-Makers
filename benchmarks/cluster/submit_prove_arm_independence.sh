#!/bin/bash
#SBATCH --job-name=arm_independence
#SBATCH --account=aip-rgrosse
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=arm_independence_%j.out
#SBATCH --error=arm_independence_%j.err

# Prevent NumPy/OpenBLAS from trying to spawn lots of threads
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd ~/projects/aip-rgrosse/lukez/CFM-for-Decision-Makers

source .venv/bin/activate

python -u benchmarks/empirical_tests/prove_arm_independence.py