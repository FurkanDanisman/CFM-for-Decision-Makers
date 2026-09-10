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

### Empirical Test (New)

We test whether the RealCause potential outcomes \(Y(0)\) and \(Y(1)\) are dependent **conditional on \(X\)** for IHDP, ACIC, CPS, and PSID. For IHDP/ACIC, we subtract the known conditional means \(\mu_0(X)\) and \(\mu_1(X)\) and test dependence between the resulting residuals. For CPS/PSID, we use 100 repeated RealCause draws of the same individuals and test \(Y(0)\)–\(Y(1)\) dependence across draws while holding each individual's covariates fixed. In addition to Pearson correlation, we use permutation-calibrated omnibus and stratified \(G\)-tests, with Bonferroni correction within each dataset. Failure to reject indicates no detectable dependence, not proof of exact independence.

* **Raw Pearson:** Correlation between the actual potential outcomes $Y(0)$ and $Y(1)$.
* **Conditional Pearson:** Correlation after subtracting the known conditional means:
  $$\epsilon_0 = Y(0) - \mu_0(X), \quad \epsilon_1 = Y(1) - \mu_1(X)$$

| Dataset | \(n\) pairs | Conditional Pearson \(r\) |               95% CI | Pearson \(p\) | Spearman \(\rho\) | Omnibus perm. \(p\) | Stratified perm. \(p\) | Decision     |
| ------- | ----------: | ------------------------: | -------------------: | ------------: | ----------------: | ------------------: | ---------------------: | ------------ |
| IHDP    |      74,700 |                  +0.00095 | [-0.00622, +0.00812] |         0.795 |          -0.00076 |              0.0846 |                 0.2239 | Not rejected |
| ACIC    |      48,020 |                  +0.00242 | [-0.00653, +0.01136] |         0.596 |          +0.00407 |              0.9453 |                 0.9005 | Not rejected |
| CPS     |   1,617,700 |                  -0.00103 | [-0.00259, +0.00053] |         0.189 |                 — |              0.6020 |                 0.9602 | Not rejected |
| PSID    |     267,500 |                  +0.00327 | [-0.00057, +0.00712] |         0.109 |                 — |              0.8109 |                 0.4826 | Not rejected |


<!-- ### Empirical Test (Stale)
Two designs: IHDP/ACIC ship both potential outcomes and both μ_t, so noise is directly observable as eps_t = y_t − mu_t(x) and you test (eps0, eps1) pooled. CPS/PSID have no μ, but the 100 CSVs are resamples of the same units, so each unit gives 100 iid draws — test per unit, pool across units.

| Dataset | Pearson \(r\) |              95% CI | Spearman \(\rho\) | \(G\)-test/permutation \(p\) |     MDE |
| ------- | ------------: | ------------------: | ----------------: | ---------------------------: | ------: |
| IHDP    |       0.00083 | [-0.00634, 0.00800] |          -0.00098 |                        0.980 |  0.0103 |
| ACIC    |       0.00237 | [-0.00657, 0.01131] |           0.00405 |                        0.234 |  0.0128 |
| CPS     |      -0.00103 | [-0.00259, 0.00053] |                 — |                        0.318 | 0.00224 |
| PSID    |       0.00327 | [-0.00057, 0.00712] |                 — |                        0.821 | 0.00550 | -->


More details in python /project/6105522/lukez/CFM-for-Decision-Makers/benchmarks/empirical_tests/prove_arm_independence.py, FULL RESULTS in arm_independence_5312807.out

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

<!-- # CATE Density Evaluation 

## IHDP   realizations=100  ~75 queries each  anc=v6a  |tau*|>3: 0.00% 

| Method                   |                 NLL |                  L2 |              KL_fwd |               KL_rev |            Mass |
| ------------------------ | ------------------: | ------------------: | ------------------: | -------------------: | --------------: |
| UWYK `(x)indep` `K=1000` |     0.7224 ± 0.0428 |     1.6925 ± 0.1042 |     1.5304 ± 0.0580 |     34.2109 ± 6.9398 | 1.0000 ± 0.0000 |
| UWYK `(x)indep` `J=32`   |     0.7202 ± 0.0429 |     1.6904 ± 0.1041 |     1.5281 ± 0.0579 |     34.4410 ± 6.9803 | 1.0000 ± 0.0000 |
| Joint-2D `J=32`          | **0.0238 ± 0.0400** | **1.3128 ± 0.1133** | **0.8363 ± 0.0626** | **22.3580 ± 5.1948** | 1.0000 ± 0.0000 |


| Comparison                                             |        ΔNLL |      ΔKL_rev |
| ------------------------------------------------------ | ----------: | -----------: |
| Resolution handicap: UWYK native → UWYK matched        |     -0.0023 |      +0.2301 |
| Model gap at equal resolution: UWYK matched → Joint-2D | **-0.6964** | **-12.0830** |

  (negative = joint better; expect the joint to LOSE slightly if it carries a spurious rho -- the truth here factorises)

## ACIC   realizations=10  ~481 queries each  anc=v6a  |tau*|>3: 0.00% 

| Method                   |                  NLL |                  L2 |              KL_fwd |              KL_rev |            Mass |
| ------------------------ | -------------------: | ------------------: | ------------------: | ------------------: | --------------: |
| UWYK `(x)indep` `K=1000` |     -0.0970 ± 0.1207 |     1.6815 ± 0.0529 |     1.1930 ± 0.0787 |    13.4325 ± 1.6074 | 1.0000 ± 0.0000 |
| UWYK `(x)indep` `J=32`   |     -0.0955 ± 0.1193 |     1.6823 ± 0.0533 |     1.1945 ± 0.0781 |    13.5690 ± 1.6080 | 1.0000 ± 0.0000 |
| Joint-2D `J=32`          | **-0.4709 ± 0.1561** | **1.3283 ± 0.1227** | **0.8172 ± 0.1242** | **5.8669 ± 1.1838** | 1.0000 ± 0.0000 |


| Comparison                                             |        ΔNLL |     ΔKL_rev |
| ------------------------------------------------------ | ----------: | ----------: |
| Resolution handicap: UWYK native → UWYK matched        |     +0.0015 |     +0.1365 |
| Model gap at equal resolution: UWYK matched → Joint-2D | **-0.3754** | **-7.7021** |

  (negative = joint better; expect the joint to LOSE slightly if it carries a spurious rho -- the truth here factorises)

--- -->

<!-- ### IHDP

realizations=100, ~75 queries each, anc=anc, |tau*|>3: 0.00%

| method               |                nll |                l2 |            kl_fwd |             kl_rev |              mass |
| -------------------- | -----------------: | ----------------: | ----------------: | -----------------: | ----------------: |
| UWYK (x)indep K=1000 |      0.1598±0.0243 |     1.4080±0.1180 |     0.9686±0.0762 |     26.1195±5.9006 | **1.0000±0.0000** |
| UWYK (x)indep J=32   |      0.1612±0.0243 |     1.4083±0.1178 |     0.9701±0.0763 |     26.3955±5.9555 |     0.9999±0.0000 |
| Joint-2D J=32        | **-0.0255±0.0398** | **1.2765±0.1139** | **0.7895±0.0627** | **20.1998±4.8796** | **1.0000±0.0000** |

|              | contrast                                          |                dNLL |              dKLrev |
| ------------ | ------------------------------------------------- | ------------------: | ------------------: |
| **HEADLINE** | model gap as run (uwyk_native -> joint)           | **-0.1852+-0.0227** | **-5.9197+-1.1456** |
| bridge       | resolution handicap (uwyk_native -> uwyk_matched) |     +0.0015+-0.0003 |     +0.2760+-0.0641 |

_negative = joint better; the bridge row should be ~0, which is what licenses reading the headline as a model gap and not a resolution artefact._

### ACIC

realizations=10, ~481 queries each, anc=anc, |tau*|>3: 0.00%

| method               |                nll |                l2 |            kl_fwd |            kl_rev |          mass |
| -------------------- | -----------------: | ----------------: | ----------------: | ----------------: | ------------: |
| UWYK (x)indep K=1000 |     -0.4253±0.1200 |     1.4532±0.0673 |     0.8648±0.0813 |     7.7151±0.9422 | 1.0000±0.0000 |
| UWYK (x)indep J=32   |     -0.4146±0.1168 |     1.4671±0.0656 |     0.8753±0.0788 |     7.8500±0.9437 | 1.0000±0.0000 |
| Joint-2D J=32        | **-0.4600±0.1597** | **1.3369±0.1326** | **0.8276±0.1268** | **5.7765±1.1925** | 1.0000±0.0000 |

|              | contrast                                          |                dNLL |              dKLrev |
| ------------ | ------------------------------------------------- | ------------------: | ------------------: |
| **HEADLINE** | model gap as run (uwyk_native -> joint)           | **-0.0347+-0.0777** | **-1.9386+-0.6032** |
| bridge       | resolution handicap (uwyk_native -> uwyk_matched) |     +0.0108+-0.0035 |     +0.1349+-0.0192 |

_negative = joint better; the bridge row should be ~0, which is what licenses reading the headline as a model gap and not a resolution artefact._

---

### IHDP

realizations=100, ~75 queries each, anc=v3a, |tau*|>3: 0.00%

| method               |                nll |                l2 |            kl_fwd |             kl_rev |              mass |
| -------------------- | -----------------: | ----------------: | ----------------: | -----------------: | ----------------: |
| UWYK (x)indep K=1000 |      0.1598±0.0243 |     1.4080±0.1180 |     0.9686±0.0762 |     26.1195±5.9006 | **1.0000±0.0000** |
| UWYK (x)indep J=32   |      0.1612±0.0243 |     1.4083±0.1178 |     0.9701±0.0763 |     26.3955±5.9555 |     0.9999±0.0000 |
| Joint-2D J=32        | **-0.0504±0.0364** | **1.2536±0.1155** | **0.7623±0.0655** | **20.9813±4.9811** | **1.0000±0.0000** |

|              | contrast                                          |                dNLL |              dKLrev |
| ------------ | ------------------------------------------------- | ------------------: | ------------------: |
| **HEADLINE** | model gap as run (uwyk_native -> joint)           | **-0.2101+-0.0196** | **-5.1382+-0.9769** |
| bridge       | resolution handicap (uwyk_native -> uwyk_matched) |     +0.0015+-0.0003 |     +0.2760+-0.0641 |

_negative = joint better; the bridge row should be ~0, which is what licenses reading the headline as a model gap and not a resolution artefact._

### ACIC

realizations=10, ~481 queries each, anc=v3a, |tau*|>3: 0.00%

| method               |                nll |                l2 |            kl_fwd |            kl_rev |          mass |
| -------------------- | -----------------: | ----------------: | ----------------: | ----------------: | ------------: |
| UWYK (x)indep K=1000 |     -0.4253±0.1200 |     1.4532±0.0673 |     0.8648±0.0813 |     7.7151±0.9422 | 1.0000±0.0000 |
| UWYK (x)indep J=32   |     -0.4146±0.1168 |     1.4671±0.0656 |     0.8753±0.0788 |     7.8500±0.9437 | 1.0000±0.0000 |
| Joint-2D J=32        | **-0.4862±0.1496** | **1.3189±0.1227** | **0.8035±0.1163** | **6.0440±1.1985** | 1.0000±0.0000 |

|              | contrast                                          |                dNLL |              dKLrev |
| ------------ | ------------------------------------------------- | ------------------: | ------------------: |
| **HEADLINE** | model gap as run (uwyk_native -> joint)           | **-0.0608+-0.0680** | **-1.6711+-0.6555** |
| bridge       | resolution handicap (uwyk_native -> uwyk_matched) |     +0.0108+-0.0035 |     +0.1349+-0.0192 |

_negative = joint better; the bridge row should be ~0, which is what licenses reading the headline as a model gap and not a resolution artefact._

---

### IHDP

realizations=100, ~75 queries each, anc=noanc, |tau*|>3: 0.00%

| method               |               nll |                l2 |            kl_fwd |             kl_rev |          mass |
| -------------------- | ----------------: | ----------------: | ----------------: | -----------------: | ------------: |
| UWYK (x)indep K=1000 |     0.3415±0.0330 |     1.5382±0.1098 |     1.1500±0.0673 |     29.5653±6.3824 | 1.0000±0.0000 |
| UWYK (x)indep J=32   |     0.3426±0.0329 |     1.5377±0.1097 |     1.1511±0.0673 |     29.8211±6.4307 | 1.0000±0.0000 |
| Joint-2D J=32        | **0.0103±0.0358** | **1.2952±0.1144** | **0.8201±0.0656** | **24.5587±5.4782** | 1.0000±0.0000 |

|              | contrast                                          |                dNLL |              dKLrev |
| ------------ | ------------------------------------------------- | ------------------: | ------------------: |
| **HEADLINE** | model gap as run (uwyk_native -> joint)           | **-0.3312+-0.0092** | **-5.0066+-0.9900** |
| bridge       | resolution handicap (uwyk_native -> uwyk_matched) |     +0.0011+-0.0003 |     +0.2558+-0.0588 |

_negative = joint better; the bridge row should be ~0, which is what licenses reading the headline as a model gap and not a resolution artefact._

### ACIC

realizations=10, ~481 queries each, anc=noanc, |tau*|>3: 0.00%

| method               |                nll |                l2 |            kl_fwd |            kl_rev |          mass |
| -------------------- | -----------------: | ----------------: | ----------------: | ----------------: | ------------: |
| UWYK (x)indep K=1000 |     -0.2536±0.1118 |     1.5859±0.0576 |     1.0374±0.0725 |    10.7707±1.2125 | 1.0000±0.0000 |
| UWYK (x)indep J=32   |     -0.2478±0.1092 |     1.5911±0.0574 |     1.0429±0.0708 |    10.9066±1.2150 | 1.0000±0.0000 |
| Joint-2D J=32        | **-0.4638±0.1481** | **1.3433±0.1160** | **0.8256±0.1152** | **6.8656±1.3200** | 1.0000±0.0000 |

|              | contrast                                          |                dNLL |              dKLrev |
| ------------ | ------------------------------------------------- | ------------------: | ------------------: |
| **HEADLINE** | model gap as run (uwyk_native -> joint)           | **-0.2102+-0.0672** | **-3.9051+-0.6297** |
| bridge       | resolution handicap (uwyk_native -> uwyk_matched) |     +0.0058+-0.0030 |     +0.1359+-0.0194 |

_negative = joint better; the bridge row should be ~0, which is what licenses reading the headline as a model gap and not a resolution artefact._ -->


<!-- ### IHDP

realizations=100, ~75 queries each, anc=v3a, |tau*|>3: 0.00%

| method               |                nll |                l2 |            kl_fwd |             kl_rev |              mass |
| -------------------- | -----------------: | ----------------: | ----------------: | -----------------: | ----------------: |
| UWYK (x)indep K=1000 |      0.1598±0.0243 |     1.4080±0.1180 |     0.9686±0.0762 |     26.1195±5.9006 | **1.0000±0.0000** |
| UWYK (x)indep J=32   |      0.1612±0.0243 |     1.4083±0.1178 |     0.9701±0.0763 |     26.3955±5.9555 |     0.9999±0.0000 |
| Joint-2D J=32        | **-0.0504±0.0364** | **1.2536±0.1155** | **0.7623±0.0655** | **20.9813±4.9811** | **1.0000±0.0000** |

Point errors in original outcome units, from the same predictions. Full-density means except the interior row; CATE L1 is per-query MAE, ATE error is unnormalised.

| mean estimator               |         sqrt PEHE |           CATE L1 |     ATE abs error |
| ---------------------------- | ----------------: | ----------------: | ----------------: |
| UWYK (x)indep K=1000         |     5.4806±0.7760 |     4.3345±0.5533 |     1.8014±0.1178 |
| UWYK (x)indep J=32           |     5.4810±0.7761 |     4.3349±0.5534 |     1.8014±0.1177 |
| Joint-2D J=32                |     4.3144±0.6278 | **3.1857±0.3999** |     1.0791±0.0780 |
| Joint-2D interior mean (raw) | **4.3143±0.6278** | **3.1857±0.3999** | **1.0790±0.0780** |
  uwyk_native: max |finite-grid moment - full mean| = 0.998319
  uwyk_matched: max |finite-grid moment - full mean| = 0.99857
  joint: max |finite-grid moment - full mean| = 7.95504e-08

|              | contrast                                          |               dNLL |             dKLrev |              dPEHE |
| ------------ | ------------------------------------------------- | -----------------: | -----------------: | -----------------: |
| **HEADLINE** | model gap as run (uwyk_native -> joint)           | **-0.2101±0.0196** | **-5.1382±0.9769** | **-1.1662±0.1629** |
| bridge       | resolution handicap (uwyk_native -> uwyk_matched) |     +0.0015±0.0003 |     +0.2760±0.0641 |     +0.0004±0.0002 |

_negative = joint better; the bridge row should be ~0, which is what licenses reading the headline as a model gap and not a resolution artefact._

### ACIC

realizations=10, ~481 queries each, anc=v3a, |tau*|>3: 0.00%

| method               |                nll |                l2 |            kl_fwd |            kl_rev |          mass |
| -------------------- | -----------------: | ----------------: | ----------------: | ----------------: | ------------: |
| UWYK (x)indep K=1000 |     -0.4253±0.1200 |     1.4532±0.0673 |     0.8648±0.0813 |     7.7151±0.9422 | 1.0000±0.0000 |
| UWYK (x)indep J=32   |     -0.4146±0.1168 |     1.4671±0.0656 |     0.8753±0.0788 |     7.8500±0.9437 | 1.0000±0.0000 |
| Joint-2D J=32        | **-0.4862±0.1496** | **1.3189±0.1227** | **0.8035±0.1163** | **6.0440±1.1985** | 1.0000±0.0000 |

Point errors in original outcome units, from the same predictions. Full-density means except the interior row; CATE L1 is per-query MAE, ATE error is unnormalised.

| mean estimator               |         sqrt PEHE |           CATE L1 |     ATE abs error |
| ---------------------------- | ----------------: | ----------------: | ----------------: |
| UWYK (x)indep K=1000         | **2.6996±0.4419** |     1.9516±0.3321 |     0.5677±0.1565 |
| UWYK (x)indep J=32           |     2.6997±0.4420 |     1.9517±0.3322 |     0.5672±0.1568 |
| Joint-2D J=32                |     2.7840±0.5051 | **1.9171±0.3554** | **0.4155±0.1125** |
| Joint-2D interior mean (raw) |     2.7840±0.5051 | **1.9171±0.3554** | **0.4155±0.1125** |
  uwyk_native: max |finite-grid moment - full mean| = 0.399156
  uwyk_matched: max |finite-grid moment - full mean| = 0.399538
  joint: max |finite-grid moment - full mean| = 2.77443e-08

|              | contrast                                          |               dNLL |             dKLrev |              dPEHE |
| ------------ | ------------------------------------------------- | -----------------: | -----------------: | -----------------: |
| **HEADLINE** | model gap as run (uwyk_native -> joint)           | **-0.0608±0.0680** | **-1.6711±0.6555** |     +0.0843±0.2296 |
| bridge       | resolution handicap (uwyk_native -> uwyk_matched) |     +0.0108±0.0035 |     +0.1349±0.0192 | **+0.0000±0.0001** |

_negative = joint better; the bridge row should be ~0, which is what licenses reading the headline as a model gap and not a resolution artefact._ -->

# Raw

### IHDP

Truth: generator Gaussian, raw sigma=1. Training-residual sigma (diagnostic, raw units): mean=1.0023, range=[0.9485, 1.0683].

realizations=100, ~75 queries each, anc=noanc, |tau*|>3: 0.00%

| method                     |               nll |                l2 |            kl_fwd |             kl_rev |          mass |
| -------------------------- | ----------------: | ----------------: | ----------------: | -----------------: | ------------: |
| UWYK (x)indep K=1000       |     0.3415±0.0330 |     1.5399±0.1098 |     1.1514±0.0672 |     29.6141±6.3778 | 1.0000±0.0000 |
| UWYK (x)indep matched bins |     0.3426±0.0329 |     1.5394±0.1097 |     1.1525±0.0672 |     29.8720±6.4265 | 1.0000±0.0000 |
| UWYK Joint-2D              | **0.0103±0.0358** | **1.2964±0.1144** | **0.8208±0.0656** | **24.6051±5.4753** | 1.0000±0.0000 |

Point errors in original outcome units, from the same predictions. Full-density means except the interior row; CATE L1 is per-query MAE, ATE error is unnormalised.

| mean estimator               |         sqrt PEHE |           CATE L1 |     ATE abs error |
| ---------------------------- | ----------------: | ----------------: | ----------------: |
| UWYK (x)indep K=1000         |     6.2789±0.7908 |     5.1695±0.5638 |     2.7218±0.1030 |
| UWYK (x)indep matched bins   |     6.2802±0.7910 |     5.1709±0.5640 |     2.7221±0.1030 |
| UWYK Joint-2D                | **4.5190±0.6318** |     3.4015±0.4025 | **1.2799±0.0793** |
| Joint-2D interior mean (raw) | **4.5190±0.6318** | **3.4014±0.4025** | **1.2799±0.0793** |
  uwyk_native: max |finite-grid moment - full mean| = 0.364768
  uwyk_matched: max |finite-grid moment - full mean| = 0.365283
  joint: max |finite-grid moment - full mean| = 2.30171e-07

|              | contrast                                          |               dNLL |             dKLrev |              dPEHE |
| ------------ | ------------------------------------------------- | -----------------: | -----------------: | -----------------: |
| **HEADLINE** | model gap as run (uwyk_native -> joint)           | **-0.3312±0.0092** | **-5.0090±0.9872** | **-1.7599±0.1653** |
| bridge       | resolution handicap (uwyk_native -> uwyk_matched) |     +0.0011±0.0003 |     +0.2579±0.0597 |     +0.0013±0.0002 |

_negative = destination method has lower error. The UWYK bridge measures rebinning effects; DoPFN compares native bins with its joint head, without a resolution-matched control._

### ACIC

Truth: generator Gaussian, raw sigma=1. Training-residual sigma (diagnostic, raw units): mean=0.9951, range=[0.9800, 1.0119].

realizations=10, ~481 queries each, anc=noanc, |tau*|>3: 0.00%

| method                     |                nll |                l2 |            kl_fwd |            kl_rev |          mass |
| -------------------------- | -----------------: | ----------------: | ----------------: | ----------------: | ------------: |
| UWYK (x)indep K=1000       |     -0.2536±0.1118 |     1.5791±0.0566 |     1.0328±0.0731 |    10.6714±1.2296 | 1.0000±0.0000 |
| UWYK (x)indep matched bins |     -0.2478±0.1092 |     1.5843±0.0564 |     1.0383±0.0715 |    10.8056±1.2318 | 1.0000±0.0000 |
| UWYK Joint-2D              | **-0.4638±0.1481** | **1.3377±0.1164** | **0.8232±0.1155** | **6.8293±1.3340** | 1.0000±0.0000 |

Point errors in original outcome units, from the same predictions. Full-density means except the interior row; CATE L1 is per-query MAE, ATE error is unnormalised.

| mean estimator               |         sqrt PEHE |           CATE L1 |     ATE abs error |
| ---------------------------- | ----------------: | ----------------: | ----------------: |
| UWYK (x)indep K=1000         |     3.3019±0.4730 |     2.5166±0.3596 |     1.2980±0.1865 |
| UWYK (x)indep matched bins   |     3.3022±0.4730 |     2.5170±0.3596 |     1.2984±0.1866 |
| UWYK Joint-2D                | **2.7751±0.5009** |     1.9218±0.3516 | **0.3581±0.0837** |
| Joint-2D interior mean (raw) | **2.7751±0.5009** | **1.9217±0.3516** | **0.3581±0.0837** |
  uwyk_native: max |finite-grid moment - full mean| = 0.202875
  uwyk_matched: max |finite-grid moment - full mean| = 0.203023
  joint: max |finite-grid moment - full mean| = 1.16339e-07

|              | contrast                                          |               dNLL |             dKLrev |              dPEHE |
| ------------ | ------------------------------------------------- | -----------------: | -----------------: | -----------------: |
| **HEADLINE** | model gap as run (uwyk_native -> joint)           | **-0.2102±0.0672** | **-3.8421±0.6009** | **-0.5268±0.1353** |
| bridge       | resolution handicap (uwyk_native -> uwyk_matched) |     +0.0058±0.0030 |     +0.1342±0.0190 |     +0.0004±0.0002 |

_negative = destination method has lower error. The UWYK bridge measures rebinning effects; DoPFN compares native bins with its joint head, without a resolution-matched control._

### IHDP

Truth: generator Gaussian, raw sigma=1. Training-residual sigma (diagnostic, raw units): mean=1.0023, range=[0.9485, 1.0683].

realizations=100, ~75 queries each, anc=v3a, |tau*|>3: 0.00%

| method               |                nll |                l2 |            kl_fwd |             kl_rev |              mass |
| -------------------- | -----------------: | ----------------: | ----------------: | -----------------: | ----------------: |
| UWYK (x)indep K=1000 |      0.1598±0.0243 |     1.4097±0.1180 |     0.9697±0.0761 |     26.1624±5.8964 | **1.0000±0.0000** |
| UWYK (x)indep J=32   |      0.1612±0.0243 |     1.4101±0.1178 |     0.9712±0.0762 |     26.4405±5.9516 |     0.9999±0.0000 |
| Joint-2D J=32        | **-0.0504±0.0364** | **1.2545±0.1156** | **0.7628±0.0655** | **21.0218±4.9739** | **1.0000±0.0000** |

Point errors in original outcome units, from the same predictions. Full-density means except the interior row; CATE L1 is per-query MAE, ATE error is unnormalised.

| mean estimator               |         sqrt PEHE |           CATE L1 |     ATE abs error |
| ---------------------------- | ----------------: | ----------------: | ----------------: |
| UWYK (x)indep K=1000         |     5.4806±0.7760 |     4.3345±0.5533 |     1.8014±0.1178 |
| UWYK (x)indep J=32           |     5.4810±0.7761 |     4.3349±0.5534 |     1.8014±0.1177 |
| Joint-2D J=32                |     4.3144±0.6278 | **3.1857±0.3999** |     1.0791±0.0780 |
| Joint-2D interior mean (raw) | **4.3143±0.6278** | **3.1857±0.3999** | **1.0790±0.0780** |
  uwyk_native: max |finite-grid moment - full mean| = 0.998319
  uwyk_matched: max |finite-grid moment - full mean| = 0.99857
  joint: max |finite-grid moment - full mean| = 7.95504e-08

|              | contrast                                          |               dNLL |             dKLrev |              dPEHE |
| ------------ | ------------------------------------------------- | -----------------: | -----------------: | -----------------: |
| **HEADLINE** | model gap as run (uwyk_native -> joint)           | **-0.2101±0.0196** | **-5.1406±0.9796** | **-1.1662±0.1629** |
| bridge       | resolution handicap (uwyk_native -> uwyk_matched) |     +0.0015±0.0003 |     +0.2781±0.0648 |     +0.0004±0.0002 |

_negative = joint better; the bridge row should be ~0, which is what licenses reading the headline as a model gap and not a resolution artefact._

### ACIC

Truth: generator Gaussian, raw sigma=1. Training-residual sigma (diagnostic, raw units): mean=0.9951, range=[0.9800, 1.0119].

realizations=10, ~481 queries each, anc=v3a, |tau*|>3: 0.00%

| method               |                nll |                l2 |            kl_fwd |            kl_rev |          mass |
| -------------------- | -----------------: | ----------------: | ----------------: | ----------------: | ------------: |
| UWYK (x)indep K=1000 |     -0.4253±0.1200 |     1.4460±0.0673 |     0.8606±0.0824 |     7.6526±0.9644 | 1.0000±0.0000 |
| UWYK (x)indep J=32   |     -0.4146±0.1168 |     1.4599±0.0655 |     0.8711±0.0799 |     7.7857±0.9658 | 1.0000±0.0000 |
| Joint-2D J=32        | **-0.4862±0.1496** | **1.3138±0.1228** | **0.8011±0.1166** | **6.0122±1.2115** | 1.0000±0.0000 |

Point errors in original outcome units, from the same predictions. Full-density means except the interior row; CATE L1 is per-query MAE, ATE error is unnormalised.

| mean estimator               |         sqrt PEHE |           CATE L1 |     ATE abs error |
| ---------------------------- | ----------------: | ----------------: | ----------------: |
| UWYK (x)indep K=1000         | **2.6996±0.4419** |     1.9516±0.3321 |     0.5677±0.1565 |
| UWYK (x)indep J=32           |     2.6997±0.4420 |     1.9517±0.3322 |     0.5672±0.1568 |
| Joint-2D J=32                |     2.7840±0.5051 | **1.9171±0.3554** | **0.4155±0.1125** |
| Joint-2D interior mean (raw) |     2.7840±0.5051 | **1.9171±0.3554** | **0.4155±0.1125** |
  uwyk_native: max |finite-grid moment - full mean| = 0.399156
  uwyk_matched: max |finite-grid moment - full mean| = 0.399538
  joint: max |finite-grid moment - full mean| = 2.77443e-08

|              | contrast                                          |               dNLL |             dKLrev |              dPEHE |
| ------------ | ------------------------------------------------- | -----------------: | -----------------: | -----------------: |
| **HEADLINE** | model gap as run (uwyk_native -> joint)           | **-0.0608±0.0680** | **-1.6404±0.6426** |     +0.0843±0.2296 |
| bridge       | resolution handicap (uwyk_native -> uwyk_matched) |     +0.0108±0.0035 |     +0.1331±0.0189 | **+0.0000±0.0001** |

_negative = joint better; the bridge row should be ~0, which is what licenses reading the headline as a model gap and not a resolution artefact._

### IHDP

Truth: generator Gaussian, raw sigma=1. Training-residual sigma (diagnostic, raw units): mean=1.0023, range=[0.9485, 1.0683].

realizations=100, ~75 queries each, anc=v3b, |tau*|>3: 0.00%

| method               |                nll |                l2 |            kl_fwd |             kl_rev |              mass |
| -------------------- | -----------------: | ----------------: | ----------------: | -----------------: | ----------------: |
| UWYK (x)indep K=1000 |      0.1598±0.0243 |     1.4097±0.1180 |     0.9697±0.0761 |     26.1624±5.8964 | **1.0000±0.0000** |
| UWYK (x)indep J=32   |      0.1612±0.0243 |     1.4101±0.1178 |     0.9712±0.0762 |     26.4405±5.9516 |     0.9999±0.0000 |
| Joint-2D J=32        | **-0.0017±0.0390** | **1.2931±0.1140** | **0.8113±0.0639** | **22.2235±5.2279** | **1.0000±0.0000** |

Point errors in original outcome units, from the same predictions. Full-density means except the interior row; CATE L1 is per-query MAE, ATE error is unnormalised.

| mean estimator               |         sqrt PEHE |           CATE L1 |     ATE abs error |
| ---------------------------- | ----------------: | ----------------: | ----------------: |
| UWYK (x)indep K=1000         |     5.4806±0.7760 |     4.3345±0.5533 |     1.8014±0.1178 |
| UWYK (x)indep J=32           |     5.4810±0.7761 |     4.3349±0.5534 |     1.8014±0.1177 |
| Joint-2D J=32                | **4.3538±0.5844** |     3.3551±0.3879 |     1.3035±0.0797 |
| Joint-2D interior mean (raw) | **4.3538±0.5844** | **3.3550±0.3879** | **1.3034±0.0797** |
  uwyk_native: max |finite-grid moment - full mean| = 0.998319
  uwyk_matched: max |finite-grid moment - full mean| = 0.99857
  joint: max |finite-grid moment - full mean| = 5.2887e-07

|              | contrast                                          |               dNLL |             dKLrev |              dPEHE |
| ------------ | ------------------------------------------------- | -----------------: | -----------------: | -----------------: |
| **HEADLINE** | model gap as run (uwyk_native -> joint)           | **-0.1615±0.0219** | **-3.9389±0.7979** | **-1.1267±0.2054** |
| bridge       | resolution handicap (uwyk_native -> uwyk_matched) |     +0.0015±0.0003 |     +0.2781±0.0648 |     +0.0004±0.0002 |

_negative = joint better; the bridge row should be ~0, which is what licenses reading the headline as a model gap and not a resolution artefact._

### ACIC

Truth: generator Gaussian, raw sigma=1. Training-residual sigma (diagnostic, raw units): mean=0.9951, range=[0.9800, 1.0119].

realizations=10, ~481 queries each, anc=v3b, |tau*|>3: 0.00%

| method               |                nll |                l2 |            kl_fwd |            kl_rev |          mass |
| -------------------- | -----------------: | ----------------: | ----------------: | ----------------: | ------------: |
| UWYK (x)indep K=1000 |     -0.4253±0.1200 |     1.4460±0.0673 |     0.8606±0.0824 |     7.6526±0.9644 | 1.0000±0.0000 |
| UWYK (x)indep J=32   |     -0.4146±0.1168 |     1.4599±0.0655 |     0.8711±0.0799 |     7.7857±0.9658 | 1.0000±0.0000 |
| Joint-2D J=32        | **-0.4574±0.1596** | **1.3392±0.1179** | **0.8292±0.1272** | **5.8824±1.2684** | 1.0000±0.0000 |

Point errors in original outcome units, from the same predictions. Full-density means except the interior row; CATE L1 is per-query MAE, ATE error is unnormalised.

| mean estimator               |         sqrt PEHE |           CATE L1 |     ATE abs error |
| ---------------------------- | ----------------: | ----------------: | ----------------: |
| UWYK (x)indep K=1000         | **2.6996±0.4419** | **1.9516±0.3321** |     0.5677±0.1565 |
| UWYK (x)indep J=32           |     2.6997±0.4420 |     1.9517±0.3322 |     0.5672±0.1568 |
| Joint-2D J=32                |     2.7929±0.5463 |     1.9562±0.4014 | **0.4121±0.0852** |
| Joint-2D interior mean (raw) |     2.7929±0.5463 |     1.9562±0.4014 | **0.4121±0.0852** |
  uwyk_native: max |finite-grid moment - full mean| = 0.399156
  uwyk_matched: max |finite-grid moment - full mean| = 0.399538
  joint: max |finite-grid moment - full mean| = 1.04639e-08

|              | contrast                                          |               dNLL |             dKLrev |              dPEHE |
| ------------ | ------------------------------------------------- | -----------------: | -----------------: | -----------------: |
| **HEADLINE** | model gap as run (uwyk_native -> joint)           | **-0.0320±0.0793** | **-1.7703±0.5955** |     +0.0933±0.2427 |
| bridge       | resolution handicap (uwyk_native -> uwyk_matched) |     +0.0108±0.0035 |     +0.1331±0.0189 | **+0.0000±0.0001** |

_negative = joint better; the bridge row should be ~0, which is what licenses reading the headline as a model gap and not a resolution artefact._

---
# MALC

### IHDP

Truth: generator Gaussian, raw sigma=1. Training-residual sigma (diagnostic, raw units): mean=1.0023, range=[0.9485, 1.0683].

realizations=100, ~75 queries each, anc=noanc, |tau*|>3: 0.00%

| method                     |               nll |                l2 |            kl_fwd |             kl_rev |              mass |
| -------------------------- | ----------------: | ----------------: | ----------------: | -----------------: | ----------------: |
| UWYK (x)indep K=1000       |     0.3407±0.0331 |     1.5410±0.1098 |     1.1633±0.0665 |     29.1809±6.3625 | **1.0000±0.0000** |
| UWYK (x)indep matched bins |     0.3441±0.0332 |     1.5427±0.1097 |     1.1773±0.0667 |     28.3922±6.2847 |     1.0001±0.0000 |
| UWYK Joint-2D              | **0.0584±0.0332** | **1.3549±0.1142** | **0.9112±0.0658** | **24.4112±5.7727** |     1.0003±0.0001 |

Point errors in original outcome units, from the same predictions. Full-density means except the interior row; CATE L1 is per-query MAE, ATE error is unnormalised.

| mean estimator             |         sqrt PEHE |           CATE L1 |     ATE abs error |
| -------------------------- | ----------------: | ----------------: | ----------------: |
| UWYK (x)indep K=1000       |     6.2816±0.7872 |     5.1661±0.5626 |     2.7169±0.1032 |
| UWYK (x)indep matched bins |     6.3010±0.7904 |     5.1737±0.5638 |     2.6976±0.0982 |
| UWYK Joint-2D              | **4.5613±0.6343** | **3.4248±0.4050** | **1.2897±0.0838** |

|              | contrast                                          |               dNLL |             dKLrev |              dPEHE |
| ------------ | ------------------------------------------------- | -----------------: | -----------------: | -----------------: |
| **HEADLINE** | model gap as run (uwyk_native -> joint)           | **-0.2823±0.0092** | **-4.7698±0.8039** | **-1.7203±0.1590** |
| bridge       | resolution handicap (uwyk_native -> uwyk_matched) |     +0.0034±0.0029 |     -0.7888±0.1678 |     +0.0193±0.0154 |

_negative = destination method has lower error. The UWYK bridge measures rebinning effects; DoPFN compares native bins with its joint head, without a resolution-matched control._

### ACIC

Truth: generator Gaussian, raw sigma=1. Training-residual sigma (diagnostic, raw units): mean=0.9951, range=[0.9800, 1.0119].

realizations=10, ~481 queries each, anc=noanc, |tau*|>3: 0.00%

| method                     |                nll |                l2 |            kl_fwd |            kl_rev |              mass |
| -------------------------- | -----------------: | ----------------: | ----------------: | ----------------: | ----------------: |
| UWYK (x)indep K=1000       |     -0.2562±0.1079 |     1.5860±0.0573 |     1.0326±0.0716 |     9.8515±1.1781 | **1.0000±0.0000** |
| UWYK (x)indep matched bins |     -0.2495±0.1027 |     1.6012±0.0574 |     1.0402±0.0695 |     9.3056±1.1119 |     1.0001±0.0000 |
| UWYK Joint-2D              | **-0.4391±0.1417** | **1.4091±0.1094** | **0.9378±0.0923** | **6.1693±1.2007** |     1.0006±0.0002 |

Point errors in original outcome units, from the same predictions. Full-density means except the interior row; CATE L1 is per-query MAE, ATE error is unnormalised.

| mean estimator             |         sqrt PEHE |           CATE L1 |     ATE abs error |
| -------------------------- | ----------------: | ----------------: | ----------------: |
| UWYK (x)indep K=1000       |     3.3488±0.4667 |     2.5461±0.3557 |     1.3051±0.1876 |
| UWYK (x)indep matched bins |     3.3634±0.4619 |     2.5663±0.3543 |     1.2814±0.1923 |
| UWYK Joint-2D              | **2.8159±0.4911** | **1.9628±0.3447** | **0.3524±0.0836** |

|              | contrast                                          |               dNLL |             dKLrev |              dPEHE |
| ------------ | ------------------------------------------------- | -----------------: | -----------------: | -----------------: |
| **HEADLINE** | model gap as run (uwyk_native -> joint)           | **-0.1828±0.0650** | **-3.6822±0.4536** | **-0.5329±0.1421** |
| bridge       | resolution handicap (uwyk_native -> uwyk_matched) |     +0.0068±0.0087 |     -0.5459±0.2695 |     +0.0146±0.0188 |

_negative = destination method has lower error. The UWYK bridge measures rebinning effects; DoPFN compares native bins with its joint head, without a resolution-matched control._