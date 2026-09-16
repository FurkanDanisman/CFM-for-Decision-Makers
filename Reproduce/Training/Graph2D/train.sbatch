#!/bin/bash
# Thin wrapper -- the real script is benchmarks/cluster/submit_train_graph2d.sbatch
# Kept as an entry point so Training/ lists every model in one place.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/env.sh"
exec sbatch --export=ALL "${REPO:-$SCRATCH/rpfn_bench_kit/R-PFN}/benchmarks/cluster/submit_train_graph2d.sbatch" "$@"
