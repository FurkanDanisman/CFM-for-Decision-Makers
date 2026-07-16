# Killarney deployment — R-PFN Table 3 sweep

Reproduces UWYK Table 3 (Do-PFN, UWYK Ancestral) and adds our full method
suite: `ours_mean`, `ours_malc_{mean, mean_msk, mode, mode_msk}`, `ours_ot_mode`.

## What runs

**Datasets** (5): IHDP, ACIC 2016, Lalonde CPS, Lalonde PSID, Lalonde PSID
(balanced sampling: all T=1 + up to 500 T=0).

**Methods per job** (one Python process):
- Do-PFN (`DoPFNRegressor` from the Do-PFN repo)
- UWYK `Ancestral Info` (`PreprocessingGraphConditionedPFN`, `full_conditioned_model`)
- OURS — 6 variants derived from the same model forward pass:
  - `ours_mean`         – E[τ] from raw `p_mat` marginals
  - `ours_malc_mean`    – E[τ] from MALC-smoothed `p(τ)`
  - `ours_malc_mean_msk`– same, after diagonal masking on `p_mat`
  - `ours_malc_mode`    – argmax MALC-smoothed `p(τ)`
  - `ours_malc_mode_msk`– argmax MALC-smoothed masked `p(τ)`
  - `ours_ot_mode`      – argmax of W2 barycenter of per-query masked densities

**Metrics**: √PEHE (IHDP, ACIC), ε_ATE (all). `ours_ot_mode` has no PEHE
(it's an ATE-only method by construction).

## Step 0 — local bundle

```bash
bash /Users/furkandanisman/R-PFN/experiments/killarney/deploy_local.sh /tmp/rpfn_bench_kit
```

Produces `/tmp/rpfn_bench_kit/` with `R-PFN/`, `external/{dopfn,uwyk,causalpfn}/`, empty
`results/` and `logs/`. Total payload ~ 1 GB (dominated by the OURS checkpoint
+ CausalPFN's realcause CSVs + UWYK checkpoint).

## Step 1 — rsync to killarney

```bash
rsync -avz --progress /tmp/rpfn_bench_kit/ \
    <you>@killarney:/scratch/<you>/rpfn_bench_kit/
```

## Step 2 — set up the venv (one time)

```bash
ssh <you>@killarney
cd /scratch/<you>/rpfn_bench_kit
export DEPLOY_ROOT=$PWD
bash R-PFN/experiments/killarney/setup_env.sh
```

If the login node blocks pip, do this on a compute node:
```bash
srun --time=00:30:00 --cpus-per-task=4 --mem=8G --pty bash
export DEPLOY_ROOT=/scratch/<you>/rpfn_bench_kit
bash $DEPLOY_ROOT/R-PFN/experiments/killarney/setup_env.sh
exit
```

The setup ends with a smoke test: prints IHDP shape + our checkpoint config.

## Step 3 — submit the array

**Phase 1 (verify Do-PFN matches Table 3):** 10 realizations, DoPFN + UWYK
only.

```bash
cd /scratch/<you>/rpfn_bench_kit
mkdir -p logs results
sbatch --array=0-49%50 \
       --export=ALL,DEPLOY_ROOT=$PWD,N_REAL=10 \
       R-PFN/experiments/killarney/submit.sbatch
```

Watch progress:
```bash
squeue -u $USER
tail -f logs/task_*.out
```

Aggregate when done:
```bash
source venv/bin/activate
python R-PFN/experiments/killarney/aggregate.py --results ./results \
       --out table3_phase1.txt
```

Check Do-PFN row against paper:

| dataset | paper Do-PFN √PEHE | paper Do-PFN ε_ATE |
|---|---|---|
| IHDP    | 6.07 ± 8.94  | 0.93 ± 3.98 |
| ACIC    | 4.11 ± 1.64  | 0.66 ± 0.12 |
| CPS     | —            | 0.88 ± 0.06 |
| PSID    | —            | 0.93 ± 0.07 |
| PSIDbal | —            | 1.09 ± 0.12 |

If our numbers land within ~1σ (of the paper's σ), we're validated.

**Phase 2 (full table, 30 realizations, all methods):**

```bash
rm -f results/*.npz   # start fresh
sbatch --array=0-149%50 \
       --export=ALL,DEPLOY_ROOT=$PWD,N_REAL=30 \
       R-PFN/experiments/killarney/submit.sbatch
```

Wallclock estimate: 30 realizations × 5 datasets ÷ 50-concurrency × ~5 min ≈ **15 min**.
Total core-hours: ~40.

Aggregate again:
```bash
python R-PFN/experiments/killarney/aggregate.py --results ./results \
       --out table3_final.txt
```

## Step 4 — pull results back

```bash
# from your laptop
rsync -avz <you>@killarney:/scratch/<you>/rpfn_bench_kit/results/ \
    /Users/furkandanisman/R-PFN/experiments/killarney_results/
rsync -avz <you>@killarney:/scratch/<you>/rpfn_bench_kit/table3_final.txt \
    /Users/furkandanisman/R-PFN/experiments/killarney_results/
```

## Knobs to tune before Phase 2

- `N_REAL=100` — matches paper's N. Wallclock ~50 min at 50-concurrency, ~5 hr at 10.
- `MAX_CONTEXT=99999` — remove context subsampling. Do-PFN inference gets slow on
  full CPS/PSID (~30 s/query). Job time budget in `submit.sbatch` may need to
  grow (`--time=02:00:00`).
- `--cpus-per-task=32` — more MALC workers per job.

## What the outputs look like

`results/IHDP_r003.npz` — one npz per (dataset, realization) with fields:

```
dataset='IHDP'  realization=3  true_ate=4.15
pehe_dopfn=5.89  err_dopfn=0.31
pehe_uwyk =0.92  err_uwyk =0.03
pehe_ours_mean=…       err_ours_mean=…
pehe_ours_malc_mean=…  err_ours_malc_mean=…
…
ate_ours_ot_mode=…     err_ours_ot_mode=…
n_queries=100  n_context=670  runtime_s=280
```

## Failure recovery

- Jobs write per-file npz — a killed job just leaves its slot empty. Resubmit
  the missing array indices with `--array=<missing>` and the aggregator picks
  them up alongside the survivors.
- Aggregator ignores missing files, so you can inspect partial results early.
- `logs/task_*.err` has the traceback if `run_one.py` raised.

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `ImportError: benchmarks` | CausalPFN path missing | check `$CAUSALPFN/benchmarks/__init__.py` exists in payload |
| `ValueError: check_array()`| Do-PFN unpatched | our copy is patched; re-run `deploy_local.sh` |
| UWYK loads a random model | checkpoint is a git-lfs pointer | payload should have the full 800MB `.pt`; check `ls -lh external/uwyk/experiments/checkpoints/full_conditioned_model/model.pt` |
| Everything is slow | MALC not parallel | `--cpus-per-task` too low; must be ≥ 4 |
| Wrong `numpy` errors | venv mixup | activate the venv on the compute node, not the login node |
