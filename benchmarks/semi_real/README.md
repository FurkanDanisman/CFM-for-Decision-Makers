# Semi-real benchmark (Do-PFN's Amazon Sales + Law School)

Runs Do-PFN, UWYK Ancestral, UWYK No-Ancestral, and Ours (2 checkpoints)
on the two semi-real datasets shipped with the Do-PFN repo:

- **`sales_cate`** — Amazon Sales (Blöbaum et al. 2024).
- **`law_race_cate`** — Law School Admissions (Kusner et al. 2017 / LSAC 1998).

Ground-truth CATE is computed via DoWhy on the paper's agreed graph
(mirrors Do-PFN's `reproduce.ipynb` cell 3). Metric: √PEHE and ε_ATE.

## Files

```
benchmarks/semi_real/
├── README.md
├── run_one.py                # per (dataset, seed) → npz
├── aggregate.py              # npz files → table (rows: methods, cols: datasets × metrics)
└── submit_semi_real.sbatch   # SLURM array (2 datasets × N_SEEDS)
```

## Running on Killarney

**Pre-flight:** the two checkpoints and the Do-PFN repo must already be
in place on the cluster (they are — the Table-3 pipeline used them
already). Ensure `dowhy` is in the venv (`pip install dowhy` if needed).

**Pass 1 — fn=50 checkpoint (baselines + Ours):**
```bash
cd /scratch/furkanbd/rpfn_bench_kit
sbatch --account=aip-rgrosse \
    --export=ALL,CHECKPOINT=/scratch/furkanbd/rpfn_bench_kit/R-PFN/checkpoints/step_50000_final.pt,OUTDIR=/scratch/furkanbd/rpfn_bench_kit/results_semi_real \
    R-PFN/benchmarks/semi_real/submit_semi_real.sbatch
```

**Pass 2 — fn=10 checkpoint (Ours only, skips baselines):**
```bash
sbatch --account=aip-rgrosse \
    --export=ALL,CHECKPOINT=/scratch/furkanbd/rpfn_bench_kit/R-PFN/checkpoints_dopfn/step_50000_final.pt,OUTDIR=/scratch/furkanbd/rpfn_bench_kit/results_semi_real_fn10,OURS_ONLY=1 \
    R-PFN/benchmarks/semi_real/submit_semi_real.sbatch
```

**Aggregate:**
```bash
source venv/bin/activate
python R-PFN/benchmarks/semi_real/aggregate.py \
    --results ./results_semi_real \
    --extra   ./results_semi_real_fn10:'OURS[fn=10]' \
    --out     semi_real_table.txt
cat semi_real_table.txt
```

## Knobs

- `N_SEEDS` (default 5): how many random splits per dataset.
- Adjust `#SBATCH --array=0-9%10` to match `2 * N_SEEDS - 1` if you
  change `N_SEEDS`.
- Wallclock per task is dominated by DoWhy's `gcm.fit` (few minutes on
  Law School; ~10 min on Sales), then Do-PFN + UWYK + Ours (< 2 min each).
