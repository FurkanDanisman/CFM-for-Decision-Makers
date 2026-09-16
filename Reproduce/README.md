# Reproduce

Everything needed to regenerate every number in the paper, and nothing else.
This branch is **pruned**: if a file is here, some entry point below uses it.

Each benchmark folder is an entry point, not a copy — the wrappers call the
pipeline code in place, so there is exactly one implementation of each step.

```
Reproduce/
  Training/        the four checkpoints, from scratch
  RealCause/       IHDP · ACIC · CPS · PSID · PSID_bal
  ComplexMech/     UWYK Fig-4 prior, binarised treatment, PEHE-able
  CaseStudy/       6 DoPFN SCM case studies x 8 covariate counts x 3 shifts
```

## The one thing not self-contained

`Required_checkpoints/` holds every trained checkpoint (git LFS) **except
DoPFN-native**, which has no checkpoint of ours — it loads DoPFN's own
released artifacts. See `Training/DoPFN_BB/README.md` for how to place them.

## Order of operations

1. `Training/` — or skip it and use `Required_checkpoints/`
2. `<Benchmark>/DataGeneration/` — RealCause has none; its data ships with the
   CausalPFN repo
3. `<Benchmark>/PEHE_ATE/` — point estimates
4. `<Benchmark>/Calibration/` — coverage / length / IS_0.05, CATE and ATE
5. `<Benchmark>/PlotGeneration/` — figures

Steps 3 and 4 read the SAME dumps: each eval harness writes its predictive
density alongside its point estimate, so PEHE and the intervals always
describe one estimator. That is not incidental — see
`Calibration/README.md` for why it matters.

## Conventions that hold everywhere

- Every model is scored by ONE scorer, `UWYK_Fig3_4/cate_density_metrics.py`,
  on its own native bin resolution.
- `DENSITY_DUMP=1` makes a harness write `p_y0_scaled` / `p_joint_scaled`
  beside its point estimate.
- Nothing runs on a login node. Every step here is an `sbatch` or an `srun`.
