# Case-study evaluation on the cluster

Runs the six `realcause_eval/Table1` models on **our generated case-study
datasets** (`case_study/generation.py`), swept over context size N, and
aggregates PEHE + L1-ATE per (model, context, case).

It reuses the existing, proven eval machinery — the per-model dispatcher
`benchmarks/cluster/submit_eval_scm_case_studies.sbatch` and every underlying
`benchmarks/eval_*` script — unchanged. The only new hook is an **opt-in npz
backend** in `benchmarks/scm_case_study_dataset.py`: when `CASE_STUDY_DATA_ROOT`
is set it reads our npz (context size = `CASE_STUDY_N`) instead of DoPFN's pkls.
That is the single reliable chokepoint — every eval script force-inserts
`benchmarks/` onto `sys.path` and imports `scm_case_study_dataset` from there,
so a `PYTHONPATH` shim would be silently bypassed. With the env var unset,
behaviour is exactly as before (DoPFN pkls).

## Layout

```
case_study/cluster/
├── 01_generate.sbatch   regenerate the npz on the cluster (CPU, deterministic)
├── 02_submit_eval.sh    submit models × contexts × 6 cases (reuses the dispatcher)
├── 03_aggregate.py      per-(model, context, case) PEHE + L1-ATE table + CSV
├── run_all.sh           1 → 2 (with dependency) → prints the 3 command
└── README.md
```

Outputs land at `SWEEP/ctx<N>/<model>/<case>/<case>_r###.npz`, the same layout
`realcause_eval/aggregate_scm_ctx_sweep.py` expects (03_aggregate.py reuses that
script's per-model cell reader, parameterized for our contexts).

## One-shot

```bash
export DEPLOY_ROOT=$SCRATCH/rpfn_bench_kit     # your Killarney deploy root
cd $DEPLOY_ROOT                                 # logs_*/ are relative to here
bash $DEPLOY_ROOT/R-PFN/case_study/cluster/run_all.sh
```

Defaults: `CATE_SHIFT=2`, `CONTEXTS="200 500 1000"`, all eight model variants.
Data → `$DEPLOY_ROOT/case_study_data/shift+2/`, results →
`$DEPLOY_ROOT/results_case_study/shift+2/`. When the array jobs finish:

```bash
python $DEPLOY_ROOT/R-PFN/case_study/cluster/03_aggregate.py \
    --sweep $DEPLOY_ROOT/results_case_study/shift+2 \
    --contexts 200 500 1000 --out $DEPLOY_ROOT/results_case_study/shift+2/summary.csv
```

## Step by step

```bash
# 1. generate the npz (β = +2, contexts 200/500/1000, 100 realizations each)
DATA_ROOT=$DEPLOY_ROOT/case_study_data/shift+2 CATE_SHIFT=2 \
  sbatch $DEPLOY_ROOT/R-PFN/case_study/cluster/01_generate.sbatch

# 2. submit the eval sweep against that npz
bash $DEPLOY_ROOT/R-PFN/case_study/cluster/02_submit_eval.sh \
     $DEPLOY_ROOT/case_study_data/shift+2 \
     $DEPLOY_ROOT/results_case_study/shift+2

# 3. aggregate (after jobs finish)
python .../03_aggregate.py --sweep $DEPLOY_ROOT/results_case_study/shift+2 \
     --contexts 200 500 1000
```

## Sweeping the four shift variants

```bash
for B in +2 -2 +5 -5; do
  CATE_SHIFT=$B bash $DEPLOY_ROOT/R-PFN/case_study/cluster/run_all.sh
done
```

Each writes to its own `shift$B` data + results root, so they don't collide.

## Knobs

- `CONTEXTS` — context sizes; must be a subset of what step 1 generated.
- `MODELS` — subset of `dopfn_native dopfn_bb cpfn2d cpfn1d graph2d uwyk
  uwyk_v3a uwyk_noanc` (space-separated) passed to `02_submit_eval.sh`.
- Checkpoint overrides: `CPFN2D_CKPT`, `CPFN1D_CKPT`, `GRAPH2D_CKPT`,
  `DOPFNBB_CKPT`, `UWYK_DIR` (defaults match the RealCause sweep launcher).
- `SCM_N_QUERY` — CATE query rows per realization (default 100). The full N
  rows are always the training context.

## Metrics

- **PEHE** = mean over realizations of `sqrt(mean((cate_pred − true_cate)²))`.
- **L1-ATE** = mean over realizations of `|mean(cate_pred) − true_ate|`.
  Computed uniformly by the aggregator from each realization's `ate_pred` +
  `true_ate`, so it does not depend on any single script's own `err`
  convention. The generation-time shift keeps `true_ate` away from 0 for the
  four direct-effect cases (Backdoor / Frontdoor stay at 0 by design).

## Local (single-machine) equivalent

`case_study/eval/*.py` are thin single-machine drivers of the same underlying
scripts (same npz backend), useful for a quick one-model/one-cell run without
Slurm. The cluster scripts here are the way to produce the full results table.
