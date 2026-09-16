# ComplexMech

UWYK's Fig-4 (ComplexMech/TFM) prior, modified so PEHE is computable:
binarised treatment, exogenous noise fixed across the do(0)/do(1) passes.
Deviations from UWYK's own setup are documented in
`UWYK_Fig3_4/generate_pehe_benchmark.py` (D1–D5).

```bash
export DEPLOY_ROOT=$SCRATCH/rpfn_bench_kit
export OUT_ROOT=$SCRATCH/cmech_1d2d

bash   Reproduce/ComplexMech/DataGeneration/generate.sh    # CPU, ~3 min
sbatch benchmarks/cluster/submit_cmech_pehe_1d_vs_2d.sbatch   # 210 GPU tasks
bash   Reproduce/ComplexMech/PEHE_ATE/summarize.sh
sbatch Reproduce/ComplexMech/Calibration/score.sbatch
bash   Reproduce/ComplexMech/PlotGeneration/plots.sh
```

## Grid

7 model-runs x 6 node counts (5,10,20,30,40,50) x 5 context sizes
(50,100,250,500,1000) = 210 tasks. `SUBSET=nonzero` by default; re-run with
`SUBSET=zero` for the zero-effect queries.

**The two subsets are disjoint in QUERIES, not realizations.** A realization
with both zero- and non-zero-effect queries appears in both, so their file
counts sum to more than 100 per node count. `--subset total` pools them, and
that is the number to report.

## What this benchmark can and cannot show

`UWYK_Fig3_4/learnable_heterogeneity.py` measures how much of the true CATE
is learnable from X. On the LinGaus prior it is 0.000 everywhere — PEHE there
is an ATE benchmark wearing a different name. On ComplexMech it falls from
0.781 at n=5 to 0.143 at n=50, so the low node counts are where a CATE method
can actually be separated from a constant predictor.
