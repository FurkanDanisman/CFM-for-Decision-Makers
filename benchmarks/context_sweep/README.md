# Context-size sweep

How OURS's per-query CATE estimates change with observational context size N.

500 SCMs × 6 context sizes × 2 SCM sources = 6000 (source, seed, N) jobs.

## Sources

- **`prior`** — our own SCM prior (the distribution the model was trained on).
  Uses `scm_prior.generate_paired_sample_with_raw`, which delegates to UWYK's
  `SCMSampler` + `BinarizingMechanism` and our
  `data.PairedInterventionalDataset` paired-propagation helpers.
- **`poly`** — CausalPFN's `PolynomialDataset` (held out from training).
  Polynomial mechanisms + configurable noise.

## Context sizes

`N ∈ {50, 250, 500, 1000, 5000, 10000}`. The test set is fixed at 50 queries
per SCM.

## Layout

```
benchmarks/context_sweep/
├── README.md
├── scm_prior.py            # Option-A sampler (our prior)
├── scm_polynomial.py       # Option-B sampler (CausalPFN Polynomial)
├── run_one.py              # per-(source, seed, N) job → npz
├── aggregate.py            # npz files → two tables (prior, poly)
└── submit_sweep.sbatch     # SLURM array (120 tasks × 50 SCMs each)

# NB: the joint + marginal plots live at benchmarks/plots/ alongside
#     context_sweep_TE_distribution.png, MASK_example.png, and
#     RAW_MALC_example.png — all four figures are from the same
#     SCM seed=2 and same 6 queries, so they belong together.
```

## Running (on killarney)

```bash
# 1. rsync the payload (same layout as the main benchmark)
rsync -avz /tmp/rpfn_bench_kit/ furkanbd@killarney...:/path/to/rpfn_bench_kit/

# 2. submit
cd /path/to/rpfn_bench_kit
sbatch --account=aip-rgrosse R-PFN/benchmarks/context_sweep/submit_sweep.sbatch
```

Wallclock estimate: ~15-30 s per job × 6000 / 50-concurrency ≈ **30-60 min**.

## Aggregating

```bash
python R-PFN/benchmarks/context_sweep/aggregate.py \
    --results ./results_sweep \
    --out     context_sweep_tables.txt
cat context_sweep_tables.txt
```

Produces two tables (one per source), each with rows = OURS variants and
columns = context sizes.

## Generating the joint / marginal plots

The plot script lives at `benchmarks/plots/plot_joint_marginals.py` (grouped
with the other SCM-seed=2 figures — MASK_example, RAW_MALC_example, and
context_sweep_TE_distribution).

```bash
# Locally (uses SCM seed=2, same 6 queries as the existing TE-distribution plot)
UWYK_SRC=/tmp/g4cfm_uwyk/src \
python R-PFN/benchmarks/plots/plot_joint_marginals.py
```

Writes:
- `benchmarks/plots/scm_joint_by_context.png` — 2D joint density per (query, N)
- `benchmarks/plots/scm_marginals_by_context.png` — marginals p(Y_do0), p(Y_do1) per (query, N)

Configuration knobs (env-driven):
```
SCM_SEED=2         CONTEXT_SIZES=500,1000,2000
QUERY_IDXS=357,419,472,339,64,105    N_TRAIN=2000    N_TEST=500
```

## Which npz field is which

`results_sweep/<source>_seed<seed>_N<N>.npz`:

```
source, seed, n_context, n_test, true_ate, runtime_s

pehe_ours_mean,          err_ours_mean,          ate_ours_mean
pehe_ours_malc_mean,     err_ours_malc_mean,     ate_ours_malc_mean
pehe_ours_malc_mean_msk, err_ours_malc_mean_msk, ate_ours_malc_mean_msk
pehe_ours_malc_mode,     err_ours_malc_mode,     ate_ours_malc_mode
pehe_ours_malc_mode_msk, err_ours_malc_mode_msk, ate_ours_malc_mode_msk
ate_ours_ot_mode,        err_ours_ot_mode
```

`OT-mode` gives only a population ATE (single scalar), so no per-query `pehe_`.
