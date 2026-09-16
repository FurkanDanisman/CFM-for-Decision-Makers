# RealCause

IHDP · ACIC · CPS · PSID · PSID_bal. No DataGeneration step — the datasets
ship with the CausalPFN repo (`$CAUSALPFN`).

One dump serves both PEHE/ATE and Calibration: each harness writes its
predictive density beside its point estimate, so the two always describe the
same estimator.

```bash
export DEPLOY_ROOT=$SCRATCH/rpfn_bench_kit
export OUT_ROOT=$SCRATCH/rc_dens_uni

# 1. dumps: 6 models x 5 datasets = 30 GPU tasks (~1h each)
sbatch benchmarks/cluster/submit_realcause_density_unified.sbatch

# 2. PEHE / ATE
bash Reproduce/RealCause/PEHE_ATE/summarize.sh

# 3. Calibration
sbatch Reproduce/RealCause/Calibration/score.sbatch
```

## Models

| dir | what | checkpoint |
|---|---|---|
| `dopfn_native` | DoPFN as released | none — DoPFN's own artifacts, see Training/DoPFN_BB |
| `dopfn_bb` | DoPFN backbone + 2D head | `dopfn_bb_j10_step_150000.pt` |
| `uwyk1d` | UWYK 1D bar head | `uwyk_reproduce_best_model.pt` |
| `graph2d` | graph-conditioned 2D joint | `graph2d_step_50000.pt` |
| `cpfn1d` | CausalPFN 1D | `cpfn1d_j1024_headrand_step_50000.pt` |
| `cpfn2d_pooled` | CausalPFN 2D joint | `cpfn2d_j32_random_step_50000.pt` |

`uwyk1d` and `graph2d` run `ANC_MODE=v3ab_only`, emitting noanc + v3a + v3b
in one pass. RealCause has no per-case DAG, so `case_family` does not apply
here (it does for the case studies).

## Two deviations, both deliberate

- **cpfn1d runs `STD_MODE=pooled`, not `per_arm`.** The density dump asserts
  both arms share `y_shift`/`y_scale`, because one affine map cannot un-scale
  a difference of two differently-scaled arms. Its PEHE here is therefore
  pooled-scaled and differs slightly from a `per_arm` run.
- **No MALC smoothing.** The older RealCause CI pipeline smoothed the joint
  before projecting tau; ComplexMech and the case studies score the raw
  density, so this does too. Numbers will not match older MALC-based tables.
