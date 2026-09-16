#!/bin/bash
# CausalPFN 1D, j1024 head-random -> cpfn1d_j1024_headrand_step_50000.pt
exec sbatch "${REPO:-$SCRATCH/rpfn_bench_kit/R-PFN}/benchmarks/cluster/submit_train_causalpfn_j1024_headrand.sbatch" "$@"
