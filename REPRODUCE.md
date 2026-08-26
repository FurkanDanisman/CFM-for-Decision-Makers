# Reproducing our results

## 1. Environment

- Python 3.11
- PyTorch **2.11.0** with CUDA **12.9**
- Install pinned dependencies:
  ```bash
  pip install -r requirements.txt
  ```
  Then install torch matching your CUDA version (the pinned version above is
  what our results were produced with):
  ```bash
  pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu129
  ```

## 2. External dependencies (dataset loaders + baselines)

Two repositories must be cloned alongside `R-PFN/`. Pin these exact commits;
newer versions may change dataset generation or preprocessing and cause the
numbers to shift (see `reproduction_note.tex` for one such case).

```
<workspace>/
├── R-PFN/          (this repo)
└── external/
    ├── causalpfn/           # dataset loaders + BackdoorDGPMetaDataset
    └── uwyk_reproduce/      # UWYK baselines (Predictive / No-Anc / Anc)
```

```bash
mkdir -p external
git clone https://github.com/vdblm/CausalPFN.git external/causalpfn
git -C external/causalpfn checkout 3dd8519
git -C external/causalpfn lfs pull      # ships the semi-real datasets

git clone -b reproduce-realcause-results \
    https://github.com/ArikReuter/Graphs4CausalFoundationModels.git external/uwyk_reproduce
git -C external/uwyk_reproduce checkout c27fba6
git -C external/uwyk_reproduce lfs pull  # ships the UWYK checkpoints
```

Our code discovers these at `../external/{causalpfn,uwyk_reproduce}` by
default. Override with `CAUSALPFN_ROOT` / `UWYK_ROOT` env vars if you keep
them elsewhere.

## 3. Reproducing Table 1 (Predictive / No-Anc / Anc + fn=50)

All commands assume a SLURM cluster and are launched from the repo root.

```bash
# Runs UWYK's three baselines (predictive/no-anc/anc) on the four datasets,
# array-jobbed across (dataset x method).
sbatch benchmarks/cluster/submit_table1_all_datasets.sbatch

# Runs our fn=50 (null-t and pred-mirror variants) on the same four datasets.
sbatch benchmarks/cluster/submit_table1_ours_fn50.sbatch

# When jobs finish, aggregate:
JOB_ID_MAIN=<uwyk_job_id> \
JOB_ID_PREDSTYLE=<predstyle_job_id> \
JOB_ID_ACIC_UWYK=<acic_uwyk_job_id> \
JOB_ID_ACIC_FN50=<acic_fn50_null_t_id> \
JOB_ID_ACIC_FN50P=<acic_fn50_predstyle_id> \
python benchmarks/uwyk_table1/aggregate_all_datasets.py
```

**Note on ACIC / CPS reproduction of UWYK rows:** small (~1-2%) drift is
expected on these two datasets due to a known unseeded random subsample in
the UWYK reproduce-branch inference code. See `reproduction_note.tex` for
details. IHDP and PSID-balanced reproduce byte-for-byte.

## 4. Training our fn=50 model from scratch

If you want to re-train instead of using the shipped checkpoint at
`checkpoints/step_50000_final.pt`:

```bash
sbatch benchmarks/cluster/submit_train_causalpfn2d.sbatch
```

Wall-clock: ~3-7 days on a single H100 for the full 262k steps.
