#!/bin/bash
# CausalPFN 2D joint head, J=32, random init -> cpfn2d_j32_random_step_50000.pt
# Thin wrapper; the real script is benchmarks/cluster/.
#   sbatch Reproduce/Training/CausalPFN2D/train.sbatch
exec sbatch "${REPO:-$SCRATCH/rpfn_bench_kit/R-PFN}/benchmarks/cluster/submit_train_cpfn2d_j32_random.sbatch" "$@"
