#!/bin/bash
# Graph-conditioned 2D joint -> graph2d_step_50000.pt
exec sbatch "${REPO:-$SCRATCH/rpfn_bench_kit/R-PFN}/benchmarks/cluster/submit_train_graph2d.sbatch" "$@"
