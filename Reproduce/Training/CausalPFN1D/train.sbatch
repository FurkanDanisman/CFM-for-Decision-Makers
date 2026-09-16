#!/bin/bash
# Thin wrapper -- the real script is benchmarks/cluster/submit_train_causalpfn_j1024_headrand.sbatch
# Kept as an entry point so Training/ lists every model in one place.
exec sbatch "${REPO:-$SCRATCH/rpfn_bench_kit/R-PFN}/benchmarks/cluster/submit_train_causalpfn_j1024_headrand.sbatch" "$@"
