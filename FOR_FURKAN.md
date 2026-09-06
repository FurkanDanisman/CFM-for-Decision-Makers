# RealCause generates independent y0,y1 samples
### Codebase
The generative model is a TARNet-style network: a shared representation of w, then two separate heads producing the parameters of p(Y | T=0, w) and p(Y | T=1, w) (models/tarnet.py:39-51). Sampling is in models/nonlinear.py:275:

```
y0_, y1_ = self.mlp_y_tw(..., ret_counterfactuals=True)
y0_samples = self.outcome_distribution.sample(y0_)
y1_samples = self.outcome_distribution.sample(y1_)
```

Two independent .sample() calls. And no sampler in models/distributions/distributions.py accepts a noise argument — each one generates its own randomness internally (torch.randn(*mean.shape) in gaussian_sampler, torch.rand_like in logistic_sampler and exponential_sampler, a fresh u = torch.rand(...) for the atom/continuous mixture selection in MixedDistribution.sample). So even the "is this observation exactly zero" decision in the LaLonde zero-inflated models is drawn independently for the two arms.

The observed outcome is then y0 * (1 - t) + y1 * t (models/base.py:311, and again in make_datasets.py), so consistency does hold — the factual y is literally the corresponding potential outcome draw. It's only the counterfactual arm that gets fresh noise.

### Empirical Test
Two designs: IHDP/ACIC ship both potential outcomes and both μ_t, so noise is directly observable as eps_t = y_t − mu_t(x) and you test (eps0, eps1) pooled. CPS/PSID have no μ, but the 100 CSVs are resamples of the same units, so each unit gives 100 iid draws — test per unit, pool across units.

| Dataset | Pearson \(r\) |              95% CI | Spearman \(\rho\) | \(G\)-test/permutation \(p\) |     MDE |
| ------- | ------------: | ------------------: | ----------------: | ---------------------------: | ------: |
| IHDP    |       0.00083 | [-0.00634, 0.00800] |          -0.00098 |                        0.980 |  0.0103 |
| ACIC    |       0.00237 | [-0.00657, 0.01131] |           0.00405 |                        0.234 |  0.0128 |
| CPS     |      -0.00103 | [-0.00259, 0.00053] |                 — |                        0.318 | 0.00224 |
| PSID    |       0.00327 | [-0.00057, 0.00712] |                 — |                        0.821 | 0.00550 |


More details in python /project/6105522/lukez/CFM-for-Decision-Makers/benchmarks/empirical_tests/prove_arm_independence.py

# Plan
- Datasets: IHDP, ACIC, CPD, PSID
- Evaluation: 
    - Tier A: Marginals. UWYK: the two passes, we get this directly. Ours: the exact marginal of the 9-region mixture — not p_mat.sum(-1).
    - Tier B/C: 
        - Prescreening (B): Full 2D Distribution Evaluation. Assumes independence anyway, might as well capture more details.
        - CATE (C): Ours: diagonal integration, p(τ) = ∫ f(y₀, y₀+τ) dy₀. UWYK build f₀ ⊗ f₁ = f(y₀, y₁), requires independence assumption. 
- Metrics:
- 
| Tier / Metric  | Scope       | Eval  | Model score                                       | Truth / reference                                                |
| -------------- | ----------- | ----- | ------------------------------------------------- | ---------------------------------------------------------------- |
| Tier 1 — NLL   | All 5       | **A** | $-\log g_0(y_0^*) - \log g_1(y_1^*)$              | $N(\mu_t,\sigma^2)$ on `Y_CENTERS`; inner-conditional both sides |
| Tier 1 — NLL   | All 5       | **B** | $-\log f(y_0^*,y_1^*)$                            | $\otimes$ product                                                |
| Tier 1 — NLL   | All 5       | **C** | $-\log p_\tau(\tau^*)$                            | $N(\mu_1-\mu_0, 2\sigma^2)$ on `TAU_CENTERS`                     |
| Tier 2 — L2/KL | IHDP + ACIC | **A** | L2 / KL between predicted and true marginals      | $N(\mu_t,\sigma^2)$ on `Y_CENTERS`; inner-conditional both sides |
| Tier 2 — L2/KL | IHDP + ACIC | **B** | —                                                 | 2D L2 not defined in `density_calc.md`                           |
| Tier 2 — L2/KL | IHDP + ACIC | **C** | L2 / KL between predicted and true $\tau$ density | $N(\mu_1-\mu_0, 2\sigma^2)$ on `TAU_CENTERS`                     |


### v1:
- IHDP, ACIC
- Jumping to CATE to directly see the impact of independent draws in RealCause. 
- Comparing Ours @ 32x32 bins to UWYK @ 1000 bins. Raw densities, no malc. If we use MALC on Ours then we have to use it on UWYK too, but we have documented bugs on UWYK 1D MALC. We also need the raw density evaluation anyway: 
    1. it is a intermediate step for MALC. The adjustment I have in mind is to ditch the malc_1d_cvxpy attempt --> UWYK build f₀ ⊗ f₁ from its marginals, MALC the 2D product f₀⊗f₁, then diagonal-integrate. this means same estimator, same input shape, same output path as the joint. 
    2. raw density comparison to separate the contribution of MALC from the contribution of the architecture. 
- I use FULL density rather than inner bins. 


### Main flags for v1
- Resolution mismatch
- Claude debugging (did not take the time to understand):
```
One thing I had to change on the way
Trapezoid quadrature is O(h)-wrong at every discontinuity of a staircase density — and the error scales with bin count, so it would have been larger for the J=32 joint than for K=1000 UWYK. A differential bias between the two columns, which is precisely what this eval must not have.

Fixed by doing the interior in closed form. For uniform bins with τ = (d + φ)·bw:


p_int(τ) = (w₀ / bw) · [ (1−φ)·S(d) + φ·S(d+1) ],    S(k) = Σᵢ p_mat[i, i+k]
Exact, O(J) per τ, and integrates to w₀ by construction. Only the 8 smooth tail regions are left to quadrature.
```

# CATE Density Evaluation 

## IHDP   realizations=100  ~75 queries each  anc=v6a  |tau*|>3: 0.00% 

| Method                   |                 NLL |                  L2 |              KL_fwd |               KL_rev |            Mass |
| ------------------------ | ------------------: | ------------------: | ------------------: | -------------------: | --------------: |
| UWYK `(x)indep` `K=1000` |     0.7224 ± 0.0428 |     1.6925 ± 0.1042 |     1.5304 ± 0.0580 |     34.2100 ± 6.9395 | 1.0000 ± 0.0000 |
| UWYK `(x)indep` `J=32`   |     0.7202 ± 0.0429 |     1.6903 ± 0.1041 |     1.5281 ± 0.0579 |     34.4388 ± 6.9797 | 1.0000 ± 0.0000 |
| Joint-2D `J=32`          | **0.0238 ± 0.0400** | **1.3129 ± 0.1133** | **0.8363 ± 0.0626** | **22.3558 ± 5.1941** | 1.0000 ± 0.0000 |

| Comparison                                             |        ΔNLL |      ΔKL_rev |
| ------------------------------------------------------ | ----------: | -----------: |
| Resolution handicap: UWYK native → UWYK matched        |     -0.0023 |      +0.2288 |
| Model gap at equal resolution: UWYK matched → Joint-2D | **-0.6964** | **-12.0830** |

  (negative = joint better; expect the joint to LOSE slightly if it carries a spurious rho -- the truth here factorises)

## ACIC   realizations=10  ~481 queries each  anc=v6a  |tau*|>3: 0.00% 

| Method                   |                  NLL |                  L2 |              KL_fwd |              KL_rev |            Mass |
| ------------------------ | -------------------: | ------------------: | ------------------: | ------------------: | --------------: |
| UWYK `(x)indep` `K=1000` |     -0.0970 ± 0.1207 |     1.6814 ± 0.0529 |     1.1931 ± 0.0787 |    13.4317 ± 1.6076 | 0.9999 ± 0.0000 |
| UWYK `(x)indep` `J=32`   |     -0.0955 ± 0.1193 |     1.6823 ± 0.0533 |     1.1945 ± 0.0781 |    13.5685 ± 1.6080 | 0.9999 ± 0.0000 |
| Joint-2D `J=32`          | **-0.4709 ± 0.1561** | **1.3286 ± 0.1227** | **0.8171 ± 0.1241** | **5.8673 ± 1.1835** | 1.0001 ± 0.0002 |


| Comparison                                             |        ΔNLL |     ΔKL_rev |
| ------------------------------------------------------ | ----------: | ----------: |
| Resolution handicap: UWYK native → UWYK matched        |     +0.0015 |     +0.1368 |
| Model gap at equal resolution: UWYK matched → Joint-2D | **-0.3754** | **-7.7012** |

  (negative = joint better; expect the joint to LOSE slightly if it carries a spurious rho -- the truth here factorises)
