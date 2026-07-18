# Table 3 reproduction pipeline

Reproduces UWYK's Table 3 (RealCause benchmark) end-to-end for **Do-PFN**,
**UWYK (Ancestral Info)**, and our **6 method variants** across 5 datasets:
IHDP, ACIC 2016, Lalonde CPS, Lalonde PSID, and PSID (balanced).

Metrics: `√PEHE` and `ε_ATE = |τ̂ − τ| / |τ|`, per UWYK's `eval.py`.

## Layout

```
benchmarks/
├── run_one.py          # one-shot per (dataset, realization) — the workhorse
├── aggregate.py        # combines per-job npz files → Table 3
├── methods/            # method-specific pipelines
│   ├── dopfn.py        #   Do-PFN (DoPFNRegressor.fit + predict_cate)
│   ├── uwyk.py         #   UWYK Ancestral (target-encoded T + full-graph adj)
│   └── ours.py         #   OURS 6 variants (raw / MALC-smoothed / masked / OT)
├── cluster/            # SLURM deployment
│   ├── deploy_local.sh #   bundle payload on your machine
│   ├── setup_env.sh    #   create venv on cluster
│   └── submit.sbatch   #   500-task SLURM array (5 datasets × 100 realizations)
└── plots/              # illustrative figures
    ├── MASK_example.png                      # what diagonal masking does
    ├── context_sweep_TE_distribution.png     # how p(τ) sharpens with more context
    └── plot_mask_example.py                  # regenerator for MASK_example.png
```

## Protocol (matches paper exactly)

- 100 realizations per dataset (10 for ACIC — the CausalPFN loader only
  exposes 10 configs). Paper uses the same.
- **No** context subsampling — full train set / full test set for every job.
- PSID-balanced uses "all T=1 + up to 500 T=0" training controls (UWYK's own
  balancing protocol).
- UWYK uses the paper's exact checkpoint
  (`full_conditioned_model/final_earlytest_full_conditioning_16773252.0/`)
  with target-encoded treatment and full-graph adjacency
  (mirrors UWYK's `dofm_full_conditioning.py`).
- Do-PFN uses the vanilla `DoPFNRegressor` interface, same as their
  `inference_example.py`.
- Metrics are per UWYK's `RealCauseEval/run_baselines/eval.py`.

## Running

### 1. Bundle the payload locally

```bash
bash benchmarks/cluster/deploy_local.sh /tmp/rpfn_bench_kit
```

Prints the payload size (~1 GB) when done.

### 2. Rsync to cluster

```bash
rsync -avz --progress /tmp/rpfn_bench_kit/ \
    <user>@<cluster>:/path/to/rpfn_bench_kit/
```

### 3. Set up the environment on cluster (one-time)

```bash
cd /path/to/rpfn_bench_kit
export DEPLOY_ROOT=$PWD
bash R-PFN/benchmarks/cluster/setup_env.sh
```

### 4. Submit the array

```bash
sbatch --account=<your-account> R-PFN/benchmarks/cluster/submit.sbatch
```

Wallclock: ~30–90 min depending on queue and dataset (CPS is the bottleneck
with full 15k context).

### 5. Aggregate results into the table

```bash
source venv/bin/activate
python R-PFN/benchmarks/aggregate.py --results ./results --out table3.txt
cat table3.txt
```

## Adding a new method

Drop a file `benchmarks/methods/mymethod.py` exposing a function

```python
def mymethod_pipeline(cate_dataset, ...) -> np.ndarray:
    """Return length-N CATE predictions on cate_dataset.X_test."""
```

then import + call it inside `run_one.py::main()` next to the existing three,
and add its `pehe_` / `err_` fields to the output npz. The aggregator picks
up any new field automatically as long as it's added to the `METHODS` list
in `aggregate.py`.

## What each npz file contains

`results/<dataset>_r<realization>.npz` — one per (dataset, realization) job.
Fields:

```
dataset          e.g. "IHDP"
realization      integer
true_ate         float
n_queries, n_context, runtime_s

pehe_dopfn,              err_dopfn,              ate_dopfn
pehe_uwyk_anc,           err_uwyk_anc,           ate_uwyk_anc
pehe_ours_mean,          err_ours_mean,          ate_ours_mean
pehe_ours_malc_mean,     err_ours_malc_mean,     ate_ours_malc_mean
pehe_ours_malc_mean_msk, err_ours_malc_mean_msk, ate_ours_malc_mean_msk
pehe_ours_malc_mode,     err_ours_malc_mode,     ate_ours_malc_mode
pehe_ours_malc_mode_msk, err_ours_malc_mode_msk, ate_ours_malc_mode_msk
ate_ours_ot_mode,        err_ours_ot_mode
```

`ours_ot_mode` only has a population ATE (not per-query CATE), so it doesn't
have a `pehe_` field.
