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

# Summary Table

Table 2: Density on IHDP and ACIC.

| Method | IHDP: f_Y0 + f_Y1 ↓ | IHDP: f_τ ↓ | IHDP: f_ATE ↓ | ACIC: f_Y0 + f_Y1 ↓ | ACIC: f_τ ↓ | ACIC: f_ATE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Do-PFN | -0.6100±0.0649 | 0.1848±0.0354 | -0.1361±0.0345 | -0.2837±0.1703 | -0.0245±0.1144 | -0.1918±0.0912 |
| Do-PFN 2D | -0.2419±0.0532 | 0.4711±0.0466 | -0.1941±0.0615 | -0.1038±0.1680 | 0.2889±0.2049 | -0.1759±0.1609 |
| UWYK No-Anc | -0.7887±0.0640 | 0.3415±0.0330 | 0.0775±0.0431 | -1.2416±0.1829 | -0.2536±0.1118 | -0.5013±0.0944 |
| UWYK No-Anc 2D | -0.9272±0.0739 | 0.0103±0.0358 | -0.3798±0.0290 | -1.2508±0.2121 | -0.4638±0.1481 | -0.9971±0.1901 |
| UWYK Anc (v3a) | -1.0910±0.0494 | 0.1598±0.0243 | -0.1829±0.0325 | -1.4673±0.2005 | -0.4253±0.1200 | -0.6910±0.1024 |
| UWYK Anc 2D (v3a) | -1.0857±0.0736 | -0.0504±0.0364 | -0.5157±0.0289 | -1.3360±0.2157 | -0.4862±0.1496 | -1.0176±0.1882 |
| CausalPFN-C j1024_headrand 1D | -3.8387±0.1688 | -0.5666±0.0829 | -1.5436±0.0788 | -2.9809±0.2365 | -0.9017±0.1125 | -1.3009±0.1298 |
| CausalPFN-C j32 1D | -1.7873±0.0658 | -0.3614±0.0441 | -0.5607±0.0335 | -2.0433±0.1912 | -0.6323±0.0934 | -0.7541±0.0892 |
| CausalPFN-C botharms 1D | -3.7637±0.1664 | -0.5640±0.0842 | -1.5290±0.0790 | -2.9764±0.2475 | -0.9054±0.1191 | -1.3118±0.1405 |
| CausalPFN-C j32_random 2D | -1.7365±0.0666 | -0.3393±0.0457 | -0.6504±0.0266 | -2.0240±0.1910 | -0.6718±0.0908 | -0.8598±0.0813 |
| CausalPFN-C j32_eta0_y01 2D | -1.6910±0.0704 | -0.3248±0.0463 | -0.6344±0.0271 | -2.0666±0.1799 | -0.7026±0.0888 | -0.8750±0.0809 |

`f_X` is the NLL of density `X`, in nats, mean ± SE over realizations; lower is
better within each column. All three tiers use the **scaled** axis, but score
different targets, so their absolute NLLs should not be compared across tiers.
Converting to raw units adds `log(y_scale)` to each one-dimensional NLL and
`2 log(y_scale)` to the summed arm NLL, per realization.

- `f_Y0 + f_Y1` — summed arm NLL, formed **per realization** and only then
  averaged. The two arms are scored on the same realizations and are positively
  correlated, so adding their per-arm SEs in quadrature would understate the
  spread. Taken at the true conditional means μ_t (the dumps carry no sampled
  y0*/y1*).
- `f_τ` — NLL of p(τ | x) at the observed τ*.
- `f_ATE` — NLL of the W2-barycenter ATE density at ATE_true = mean_q(μ1 − μ0).
  `eval_density_ate.py` rebuilds the per-query p(τ | x) with the same exact
  routines and the same `TAU_CENTERS` grid the τ tier used, so `f_ATE` and `f_τ`
  sit on identical densities — the 2D heads go through `joint_tau_density`'s
  full-mixture tail quadrature, not an anti-diagonal sum or a grid convolution.
  The eval stores the metric under both truth conventions (`_bary`, the W2
  barycenter of the true per-query densities, and `_mix`, their arithmetic
  mean); for NLL they are identical to the last bit across all 1210 displayed
  method-realization cells, since −log p_est(ATE_true) reads only the estimate. The
  distinction matters for `l2` and `kl_*`, which are stored too. Estimate mass
  is 1.00000 everywhere; no inf or nan.

Swap NLL for L2 by pointing `summarize_table2.py` at the `l2*` keys; every tier
stores nll, l2, kl_fwd, kl_rev and mass.

CausalPFN-C is broken out into all five checkpoints rather than one 1D + one 2D
row, since no representative pair was nominated. Row provenance is as in
"Densities (all models)" below.


## Marginals — p(Y | do(T=t), x)

The per-arm interventional densities the models emit **directly**, one tier
upstream of p(τ). Nothing here is reconstructed from a CATE density:

- **1D heads** (Do-PFN native, UWYK, CausalPFN-1D) are two forward passes with
  T := 0 and T := 1. `eval_density_tauC.py` then *convolves* those two arms into
  p(τ); these rows read the arms themselves, before that step. They therefore
  carry no arm-independence assumption — that assumption enters only in the
  convolution.
- **2D heads** (Do-PFN 2D, UWYK Joint-2D, CausalPFN-2D) emit one joint over
  (y0, y1). p(τ) integrates p(y0, y0 + τ) over y0, with exact interior terms
  and tail quadrature; these rows integrate the other arm out of the same
  joint instead, in closed form
  (`density_common.joint_marginals`, exact — checked against brute-force 2D
  quadrature, whose gap falls 4× per 4× refinement).

Grid `Y_MARG`: [-2, 2], 8001 nodes, step 0.0005 in scaled units — the same step
as the τ grid and anchored at 0. It aligns with the UWYK grids (1D 0.002 → 4
steps, joint J=32 0.0625 → 125 steps); Do-PFN's adaptive borders and the
affinely transformed Do-PFN/CausalPFN bins need not align. Widened from ±1.5
because UWYK's half-Gaussian tail scales run ≈0.43–0.49.

- `nll` = −log p_est(μ_t), at the true conditional mean, evaluated on the
  density object rather than read off the grid. (The τ tier takes NLL at a
  sampled τ*; the dumps carry no sampled y0*/y1*.)
- `l2` = ‖p_true − p_est‖₂ against the analytic Gaussian with mean μ_t and
  standard deviation σ from the IHDP / ACIC DGPs. `mass` = ∫p_est over the
  grid, a diagnostic.
- `n` = realizations; cells are mean ± SE across them.

Computed offline from the prediction dumps `eval_density_tauC.py` already writes
(`<OUT>/predictions/`) — no model re-run. Sanity check: arm means rebuilt this
way reproduce the stored `cate_pred_uwyk_native` to 1.3e-15.
Scripts: `benchmarks/eval_graph2d/eval_density_marginals.py`,
`benchmarks/eval_graph2d/summarize_density_marginals.py`.

### IHDP — marginals

| method | n | nll_y0 | nll_y1 | l2_y0 | l2_y1 | mass_y0 | mass_y1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Do-PFN | 100 | -0.2760±0.0288 | -0.3340±0.0426 | 1.6627±0.1379 | 1.6469±0.1287 | 0.9999±0.0000 | 0.9998±0.0000 |
| Do-PFN 2D | 100 | -0.2342±0.0275 | -0.0077±0.0324 | 1.6895±0.1371 | 1.8289±0.1345 | 0.9976±0.0002 | 0.9957±0.0002 |
| UWYK No-Anc | 100 | -0.4175±0.0285 | -0.3713±0.0413 | 1.6090±0.1343 | 1.6799±0.1238 | 1.0004±0.0001 | 1.0001±0.0001 |
| UWYK No-Anc 2D | 100 | -0.3060±0.0307 | -0.6213±0.0561 | 1.6208±0.1369 | 1.4921±0.1199 | 1.0000±0.0000 | 1.0000±0.0000 |
| UWYK Anc (v3a) | 100 | -0.5686±0.0259 | -0.5224±0.0314 | 1.5103±0.1375 | 1.5819±0.1314 | 1.0008±0.0001 | 1.0001±0.0001 |
| UWYK Anc 2D (v3a) | 100 | -0.3980±0.0317 | -0.6877±0.0543 | 1.5791±0.1369 | 1.4647±0.1205 | 1.0000±0.0000 | 1.0000±0.0000 |
| CausalPFN-C j1024_headrand 1D | 100 | -1.5776±0.0706 | -2.2611±0.0995 | 0.8616±0.0829 | 1.9555±0.1136 | 1.0000±0.0000 | 0.9997±0.0002 |
| CausalPFN-C j32 1D | 100 | -0.8660±0.0311 | -0.9212±0.0353 | 1.3547±0.1362 | 1.3173±0.1354 | 1.0000±0.0000 | 1.0000±0.0000 |
| CausalPFN-C botharms 1D | 100 | -1.5674±0.0707 | -2.1963±0.0972 | 0.8449±0.0860 | 1.9687±0.1155 | 1.0000±0.0000 | 0.9997±0.0002 |
| CausalPFN-C j32_random 2D | 100 | -0.8599±0.0325 | -0.8766±0.0349 | 1.4188±0.1325 | 1.4344±0.1293 | 1.0000±0.0000 | 1.0000±0.0000 |
| CausalPFN-C j32_eta0_y01 2D | 100 | -0.8418±0.0339 | -0.8492±0.0370 | 1.4359±0.1317 | 1.4375±0.1299 | 1.0000±0.0000 | 1.0000±0.0000 |

### ACIC — marginals

| method | n | nll_y0 | nll_y1 | l2_y0 | l2_y1 | mass_y0 | mass_y1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Do-PFN | 10 | -0.1833±0.0893 | -0.1005±0.0867 | 2.1070±0.0570 | 2.1358±0.0571 | 0.9999±0.0000 | 0.9999±0.0000 |
| Do-PFN 2D | 10 | -0.0719±0.0921 | -0.0319±0.0833 | 2.1484±0.0597 | 2.1604±0.0603 | 0.9888±0.0034 | 0.9853±0.0041 |
| UWYK No-Anc | 10 | -0.8055±0.0948 | -0.4361±0.1219 | 1.7402±0.1189 | 1.9944±0.0728 | 0.9999±0.0003 | 1.0002±0.0003 |
| UWYK No-Anc 2D | 10 | -0.6955±0.1017 | -0.5553±0.1214 | 1.8252±0.1045 | 1.9031±0.0843 | 1.0000±0.0000 | 1.0000±0.0000 |
| UWYK Anc (v3a) | 10 | -0.8656±0.0941 | -0.6016±0.1363 | 1.6958±0.1137 | 1.8897±0.0807 | 0.9999±0.0003 | 1.0001±0.0003 |
| UWYK Anc 2D (v3a) | 10 | -0.7563±0.1021 | -0.5797±0.1279 | 1.7828±0.1088 | 1.8876±0.0860 | 1.0000±0.0000 | 1.0000±0.0000 |
| CausalPFN-C j1024_headrand 1D | 10 | -1.6438±0.0961 | -1.3372±0.1468 | 1.0907±0.0730 | 1.3614±0.0637 | 1.0000±0.0000 | 1.0000±0.0000 |
| CausalPFN-C j32 1D | 10 | -1.0948±0.0760 | -0.9485±0.1190 | 1.6121±0.0603 | 1.7071±0.0558 | 0.9999±0.0001 | 0.9999±0.0001 |
| CausalPFN-C botharms 1D | 10 | -1.6392±0.1010 | -1.3372±0.1536 | 1.0892±0.0715 | 1.3599±0.0687 | 1.0000±0.0000 | 1.0000±0.0000 |
| CausalPFN-C j32_random 2D | 10 | -1.0658±0.0855 | -0.9582±0.1082 | 1.6305±0.0615 | 1.7088±0.0473 | 0.9999±0.0001 | 1.0000±0.0001 |
| CausalPFN-C j32_eta0_y01 2D | 10 | -1.0678±0.0843 | -0.9988±0.0972 | 1.6298±0.0613 | 1.6876±0.0472 | 0.9999±0.0001 | 0.9999±0.0001 |


## Densities (all models) — p(τ | x)

The same eleven models in one table, so the marginal tier above and the CATE
tier line up row for row. Recomputed from the complete per-realization NPZs;
the older per-run tables below retain their earlier coverage and may differ.
`mass` is carried along here as a diagnostic.

- `nll` = −log p(τ*) at the observed τ*, `l2` = ‖p_true − p_est‖₂ on
  `TAU_CENTERS` ([-3, 3], 12001 nodes), `mass` = ∫p_est dτ.
- `n` = realizations; mean ± SE across them.

### IHDP — CATE density

| method | n | nll | l2 | mass |
| --- | ---: | ---: | ---: | ---: |
| Do-PFN | 100 | 0.1848±0.0354 | 1.4193±0.1118 | 0.9998±0.0000 |
| Do-PFN 2D | 100 | 0.4711±0.0466 | 1.5737±0.1133 | 0.9981±0.0002 |
| UWYK No-Anc | 100 | 0.3415±0.0330 | 1.5399±0.1098 | 1.0000±0.0000 |
| UWYK No-Anc 2D | 100 | 0.0103±0.0358 | 1.2964±0.1144 | 1.0000±0.0000 |
| UWYK Anc (v3a) | 100 | 0.1598±0.0243 | 1.4097±0.1180 | 1.0000±0.0000 |
| UWYK Anc 2D (v3a) | 100 | -0.0504±0.0364 | 1.2545±0.1156 | 1.0000±0.0000 |
| CausalPFN-C j1024_headrand 1D | 100 | -0.5666±0.0829 | 0.8036±0.0597 | 1.0000±0.0000 |
| CausalPFN-C j32 1D | 100 | -0.3614±0.0441 | 1.0582±0.1159 | 1.0000±0.0000 |
| CausalPFN-C botharms 1D | 100 | -0.5640±0.0842 | 0.7959±0.0587 | 1.0000±0.0000 |
| CausalPFN-C j32_random 2D | 100 | -0.3393±0.0457 | 1.0900±0.1145 | 1.0000±0.0000 |
| CausalPFN-C j32_eta0_y01 2D | 100 | -0.3248±0.0463 | 1.1071±0.1138 | 1.0000±0.0000 |

### ACIC — CATE density

| method | n | nll | l2 | mass |
| --- | ---: | ---: | ---: | ---: |
| Do-PFN | 10 | -0.0245±0.1144 | 1.7014±0.0453 | 0.9998±0.0000 |
| Do-PFN 2D | 10 | 0.2889±0.2049 | 1.9126±0.0650 | 0.9896±0.0031 |
| UWYK No-Anc | 10 | -0.2536±0.1118 | 1.5791±0.0566 | 1.0000±0.0000 |
| UWYK No-Anc 2D | 10 | -0.4638±0.1481 | 1.3377±0.1164 | 1.0000±0.0000 |
| UWYK Anc (v3a) | 10 | -0.4253±0.1200 | 1.4460±0.0673 | 1.0000±0.0000 |
| UWYK Anc 2D (v3a) | 10 | -0.4862±0.1496 | 1.3138±0.1228 | 1.0000±0.0000 |
| CausalPFN-C j1024_headrand 1D | 10 | -0.9017±0.1125 | 0.9430±0.0625 | 1.0000±0.0000 |
| CausalPFN-C j32 1D | 10 | -0.6323±0.0934 | 1.3444±0.0462 | 1.0000±0.0000 |
| CausalPFN-C botharms 1D | 10 | -0.9054±0.1191 | 0.9332±0.0653 | 1.0000±0.0000 |
| CausalPFN-C j32_random 2D | 10 | -0.6718±0.0908 | 1.3104±0.0439 | 1.0000±0.0000 |
| CausalPFN-C j32_eta0_y01 2D | 10 | -0.7026±0.0888 | 1.2883±0.0450 | 1.0000±0.0000 |

**Row provenance.** "Do-PFN" is the library (native) head and "Do-PFN 2D" is
`repro_joint2d`, both from `results_density_tauC/dopfn_refresh`. Its marginal
and ATE scores are recomputed into separate `dopfn_refresh` directories;
the old `60508900` predictions and `dopfn` derived scores are superseded.
UWYK No-Anc uses `5312884`, UWYK Anc uses the **v3a** anchor `5312882`, and
CausalPFN uses the repaired `5571187` shard. Every displayed row uses IHDP
r000–r099 and ACIC r000–r009, with matching query truth, scaling and grids.

Recheck all three tiers and all seven tables with
`python benchmarks/eval_graph2d/refresh_density_tables.py` (NumPy/SciPy required).
Add `--fill-missing --workers 4 --write` to compute missing marginal/ATE scores
from the saved predictions and update the tables; no model inference is needed.


## ATE density — p_ATE(τ)

Third tier: the W2 barycenter over queries of the model's own p(τ | x_q).
`eval_density_ate.py` rebuilds those per-query densities with the **same exact
routines and the same `TAU_CENTERS` grid** the CATE tier uses — the 2D heads go
through `joint_tau_density`'s full-mixture tail quadrature at the dump's own
n_y0, not an anti-diagonal sum of the interior and not a grid convolution of the
marginals — so `f_ATE` and `f_τ` rest on identical densities.

- `nll` = −log p_est(ATE_true), ATE_true = mean_q(μ1 − μ0) on the scaled axis.
- `ate_err` = |E[p_est] − ATE_true|. `mass` = ∫p_est dτ (1.00000 throughout).
- `n` = realizations; mean ± SE across them.

**Two truth conventions, and they are not the same object.** `truth_bary` is the
W2 barycenter of the *true* per-query densities — the same operator as the
estimate, so it is the apples-to-apples comparison and the default shown here.
`truth_mix` is their arithmetic mean, the convention named in
`realcause_eval/eval_ate_density_metrics.py`. For N(μ_q, 2σ²) the mixture has
width ≈ √(2σ² + var(μ_q)) while the barycenter has width √2·σ, so they coincide
only under a homogeneous CATE.

This matters: `nll` and `ate_err` are **identical** under both (they read only
the estimate — verified equal to the last bit across all 1210 displayed
method-realization cells), but `l2` and `kl_rev` move a lot. IHDP Do-PFN
`kl_rev` is 181.2 against `bary` and 2.67 against `mix`; its `l2` is 1.33
against `bary` and 0.46 against `mix`. The large `kl_rev` under `bary` is the
expected blow-up of KL(est ‖ truth) where the
narrow barycenter truth is near zero — read it as a shape-mismatch diagnostic,
not as a score to rank on. Pass `--truth mix` to
`summarize_density_ate.py` for the other convention.

### IHDP — ATE density

| method | n | nll | l2 | kl_rev | mass | ate_err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Do-PFN | 100 | -0.1361±0.0345 | 1.3282±0.1130 | 181.1842±17.6737 | 1.0000±0.0000 | 0.1429±0.0104 |
| Do-PFN 2D | 100 | -0.1941±0.0615 | 1.4182±0.1103 | 154.8850±16.2952 | 1.0000±0.0000 | 0.1554±0.0123 |
| UWYK No-Anc | 100 | 0.0775±0.0431 | 1.4713±0.1044 | 230.1013±14.0457 | 1.0000±0.0000 | 0.2529±0.0146 |
| UWYK No-Anc 2D | 100 | -0.3798±0.0290 | 1.1799±0.1175 | 182.2747±15.8101 | 1.0000±0.0000 | 0.1281±0.0103 |
| UWYK Anc (v3a) | 100 | -0.1829±0.0325 | 1.3105±0.1127 | 180.7021±15.8714 | 1.0000±0.0000 | 0.1384±0.0071 |
| UWYK Anc 2D (v3a) | 100 | -0.5157±0.0289 | 1.0988±0.1190 | 155.7422±15.5828 | 1.0000±0.0000 | 0.1028±0.0088 |
| CausalPFN-C j1024_headrand 1D | 100 | -1.5436±0.0788 | 0.5405±0.0494 | 2.3426±0.9875 | 1.0000±0.0000 | 0.0101±0.0010 |
| CausalPFN-C j32 1D | 100 | -0.5607±0.0335 | 1.0350±0.1167 | 96.2148±15.4849 | 1.0000±0.0000 | 0.0266±0.0028 |
| CausalPFN-C botharms 1D | 100 | -1.5290±0.0790 | 0.5582±0.0493 | 2.4936±1.0740 | 1.0000±0.0000 | 0.0191±0.0020 |
| CausalPFN-C j32_random 2D | 100 | -0.6504±0.0266 | 0.9668±0.1205 | 91.5238±15.6013 | 1.0000±0.0000 | 0.0197±0.0018 |
| CausalPFN-C j32_eta0_y01 2D | 100 | -0.6344±0.0271 | 0.9720±0.1209 | 93.0207±15.8078 | 1.0000±0.0000 | 0.0169±0.0016 |

### ACIC — ATE density

| method | n | nll | l2 | kl_rev | mass | ate_err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Do-PFN | 10 | -0.1918±0.0912 | 1.6575±0.0509 | 313.5267±14.5518 | 1.0000±0.0000 | 0.1174±0.0187 |
| Do-PFN 2D | 10 | -0.1759±0.1609 | 1.9023±0.0882 | 166.9746±24.7882 | 1.0000±0.0000 | 0.1175±0.0193 |
| UWYK No-Anc | 10 | -0.5013±0.0944 | 1.4813±0.0643 | 222.4741±19.1474 | 1.0000±0.0000 | 0.0649±0.0109 |
| UWYK No-Anc 2D | 10 | -0.9971±0.1901 | 1.0539±0.1521 | 131.0416±29.3798 | 1.0000±0.0000 | 0.0181±0.0046 |
| UWYK Anc (v3a) | 10 | -0.6910±0.1024 | 1.3363±0.0710 | 172.1913±18.2576 | 1.0000±0.0000 | 0.0284±0.0083 |
| UWYK Anc 2D (v3a) | 10 | -1.0176±0.1882 | 1.0228±0.1626 | 117.0352±26.7298 | 1.0000±0.0000 | 0.0195±0.0052 |
| CausalPFN-C j1024_headrand 1D | 10 | -1.3009±0.1298 | 0.7308±0.0999 | 36.1852±10.5732 | 1.0000±0.0000 | 0.0164±0.0048 |
| CausalPFN-C j32 1D | 10 | -0.7541±0.0892 | 1.3019±0.0489 | 134.8748±14.3535 | 1.0000±0.0000 | 0.0085±0.0023 |
| CausalPFN-C botharms 1D | 10 | -1.3118±0.1405 | 0.7047±0.1039 | 35.8029±11.8692 | 1.0000±0.0000 | 0.0127±0.0049 |
| CausalPFN-C j32_random 2D | 10 | -0.8598±0.0813 | 1.2292±0.0524 | 103.0231±11.4622 | 1.0000±0.0000 | 0.0325±0.0067 |
| CausalPFN-C j32_eta0_y01 2D | 10 | -0.8750±0.0809 | 1.2036±0.0524 | 97.7151±10.1792 | 1.0000±0.0000 | 0.0115±0.0031 |


# UWYK — IHDP, all runs

The live UWYK runs on IHDP, pooled here for readability. Per-run detail — the full metric set
(kl_fwd, kl_rev, mass), the resolution-matched control, the interior-mean rows and the
HEADLINE/bridge contrast tables — is in the commented-out `# Raw` block and the live
`# MALC` block below. The HTML-commented blocks for anc=anc and the earlier v3a pass are
excluded here too; they are superseded.

- **raw** = the `# Raw` runs, anchors noanc / v3a / v3b. **MALC** = the `# MALC` runs,
  noanc only. So v3a and v3b have no MALC counterpart.
- realizations=100, ~75 queries each, |tau*|>3: 0.00%. Training-residual sigma mean=1.0023, range=[0.9485, 1.0683].

Rows are grouped by **variant** (the blocks between rules); each block carries the two main
models. **bold** = best in that column across the **whole table**; $\underline{underline}$
= best within its own block. A block whose winner is also the table winner shows the bold
only, so each column carries one bold (or a tied set) and at most one underline per block.
Rows that tie on a value are marked alike.

## Density

<!-- previous grouping: by method family, and including the resolution-matched
control row (`matched bins` / `J=32`). Superseded by the table below.

| variant      | method                     |                          nll |                          l2 |
| ------------ | -------------------------- | ---------------------------: | --------------------------: |
| raw · noanc  | UWYK (x)indep K=1000       |                0.3415±0.0330 | $\underline{1.5399±0.1098}$ |
| raw · v3a    | UWYK (x)indep K=1000       |            **0.1598±0.0243** |           **1.4097±0.1180** |
| raw · v3b    | UWYK (x)indep K=1000       |            **0.1598±0.0243** |           **1.4097±0.1180** |
| MALC · noanc | UWYK (x)indep K=1000       |  $\underline{0.3407±0.0331}$ |               1.5410±0.1098 |
| ------------ | -------------------------- | ---------------------------: | --------------------------: |
| raw · noanc  | UWYK (x)indep matched bins |  $\underline{0.3426±0.0329}$ | $\underline{1.5394±0.1097}$ |
| raw · v3a    | UWYK (x)indep J=32         |            **0.1612±0.0243** |           **1.4101±0.1178** |
| raw · v3b    | UWYK (x)indep J=32         |            **0.1612±0.0243** |           **1.4101±0.1178** |
| MALC · noanc | UWYK (x)indep matched bins |                0.3441±0.0332 |               1.5427±0.1097 |
| ------------ | -------------------------- | ---------------------------: | --------------------------: |
| raw · noanc  | UWYK Joint-2D              |                0.0103±0.0358 |               1.2964±0.1144 |
| raw · v3a    | Joint-2D J=32              |           **-0.0504±0.0364** |           **1.2545±0.1156** |
| raw · v3b    | Joint-2D J=32              | $\underline{-0.0017±0.0390}$ | $\underline{1.2931±0.1140}$ |
| MALC · noanc | UWYK Joint-2D              |                0.0584±0.0332 |               1.3549±0.1142 |

-->

| variant      | method               |                          nll |                          l2 |
| ------------ | -------------------- | ---------------------------: | --------------------------: |
| raw · noanc  | UWYK (x)indep K=1000 |                0.3415±0.0330 |               1.5399±0.1098 |
| raw · noanc  | UWYK Joint-2D        |  $\underline{0.0103±0.0358}$ | $\underline{1.2964±0.1144}$ |
| ------------ | -------------------- | ---------------------------: | --------------------------: |
| MALC · noanc | UWYK (x)indep K=1000 |                0.3407±0.0331 |               1.5410±0.1098 |
| MALC · noanc | UWYK Joint-2D        |  $\underline{0.0584±0.0332}$ | $\underline{1.3549±0.1142}$ |
| ------------ | -------------------- | ---------------------------: | --------------------------: |
| raw · v3a    | UWYK (x)indep K=1000 |                0.1598±0.0243 |               1.4097±0.1180 |
| raw · v3a    | Joint-2D J=32        |           **-0.0504±0.0364** |           **1.2545±0.1156** |
| ------------ | -------------------- | ---------------------------: | --------------------------: |
| raw · v3b    | UWYK (x)indep K=1000 |                0.1598±0.0243 |               1.4097±0.1180 |
| raw · v3b    | Joint-2D J=32        | $\underline{-0.0017±0.0390}$ | $\underline{1.2931±0.1140}$ |

## Point estimates

Original outcome units, from the same predictions, using full-density means; CATE L1 is
per-query MAE, ATE error is unnormalised.

<!-- previous grouping: by method family, and including the resolution-matched
control row (`matched bins` / `J=32`). Superseded by the table below.

| variant      | mean estimator             |                   sqrt PEHE |                     CATE L1 |               ATE abs error |
| ------------ | -------------------------- | --------------------------: | --------------------------: | --------------------------: |
| raw · noanc  | UWYK (x)indep K=1000       | $\underline{6.2789±0.7908}$ |               5.1695±0.5638 |               2.7218±0.1030 |
| raw · v3a    | UWYK (x)indep K=1000       |           **5.4806±0.7760** |           **4.3345±0.5533** |           **1.8014±0.1178** |
| raw · v3b    | UWYK (x)indep K=1000       |           **5.4806±0.7760** |           **4.3345±0.5533** |           **1.8014±0.1178** |
| MALC · noanc | UWYK (x)indep K=1000       |               6.2816±0.7872 | $\underline{5.1661±0.5626}$ | $\underline{2.7169±0.1032}$ |
| ------------ | -------------------------- | --------------------------: | --------------------------: | --------------------------: |
| raw · noanc  | UWYK (x)indep matched bins | $\underline{6.2802±0.7910}$ | $\underline{5.1709±0.5640}$ |               2.7221±0.1030 |
| raw · v3a    | UWYK (x)indep J=32         |           **5.4810±0.7761** |           **4.3349±0.5534** |           **1.8014±0.1177** |
| raw · v3b    | UWYK (x)indep J=32         |           **5.4810±0.7761** |           **4.3349±0.5534** |           **1.8014±0.1177** |
| MALC · noanc | UWYK (x)indep matched bins |               6.3010±0.7904 |               5.1737±0.5638 | $\underline{2.6976±0.0982}$ |
| ------------ | -------------------------- | --------------------------: | --------------------------: | --------------------------: |
| raw · noanc  | UWYK Joint-2D              |               4.5190±0.6318 |               3.4015±0.4025 | $\underline{1.2799±0.0793}$ |
| raw · v3a    | Joint-2D J=32              |           **4.3144±0.6278** |           **3.1857±0.3999** |           **1.0791±0.0780** |
| raw · v3b    | Joint-2D J=32              | $\underline{4.3538±0.5844}$ | $\underline{3.3551±0.3879}$ |               1.3035±0.0797 |
| MALC · noanc | UWYK Joint-2D              |               4.5613±0.6343 |               3.4248±0.4050 |               1.2897±0.0838 |

-->

| variant      | mean estimator       |                   sqrt PEHE |                     CATE L1 |               ATE abs error |
| ------------ | -------------------- | --------------------------: | --------------------------: | --------------------------: |
| raw · noanc  | UWYK (x)indep K=1000 |               6.2789±0.7908 |               5.1695±0.5638 |               2.7218±0.1030 |
| raw · noanc  | UWYK Joint-2D        | $\underline{4.5190±0.6318}$ | $\underline{3.4015±0.4025}$ | $\underline{1.2799±0.0793}$ |
| ------------ | -------------------- | --------------------------: | --------------------------: | --------------------------: |
| MALC · noanc | UWYK (x)indep K=1000 |               6.2816±0.7872 |               5.1661±0.5626 |               2.7169±0.1032 |
| MALC · noanc | UWYK Joint-2D        | $\underline{4.5613±0.6343}$ | $\underline{3.4248±0.4050}$ | $\underline{1.2897±0.0838}$ |
| ------------ | -------------------- | --------------------------: | --------------------------: | --------------------------: |
| raw · v3a    | UWYK (x)indep K=1000 |               5.4806±0.7760 |               4.3345±0.5533 |               1.8014±0.1178 |
| raw · v3a    | Joint-2D J=32        |           **4.3144±0.6278** |           **3.1857±0.3999** |           **1.0791±0.0780** |
| ------------ | -------------------- | --------------------------: | --------------------------: | --------------------------: |
| raw · v3b    | UWYK (x)indep K=1000 |               5.4806±0.7760 |               4.3345±0.5533 |               1.8014±0.1178 |
| raw · v3b    | Joint-2D J=32        | $\underline{4.3538±0.5844}$ | $\underline{3.3551±0.3879}$ | $\underline{1.3035±0.0797}$ |

**Caveats:**

- **v3a and v3b are identical for `UWYK (x)indep K=1000`** — every nll, l2, sqrt PEHE,
  CATE and ATE value matches to all printed digits, in both datasets (the same holds for
  the matched-bins control, no longer shown). The anchor variant changes only the joint
  model, so wherever those two rows win they win together and are marked alike.
- The **resolution-matched control** (`UWYK (x)indep matched bins` in the noanc runs,
  `UWYK (x)indep J=32` in v3a/v3b) is dropped from these tables — it is the bridge
  contrast's control, not a model. It remains in the commented-out previous grouping above
  each table, and in the per-run sections.
- The joint head is printed as `UWYK Joint-2D` in the noanc runs and `Joint-2D J=32` in
  v3a/v3b. Labels are kept exactly as each run printed them.
- MALC exists only at noanc, so a MALC-vs-raw read is only licensed between the first two
  blocks.
- `Joint-2D interior mean (raw)` rows are dropped here; they appear in the `# Raw` runs
  only, never in `# MALC`.
- MALC rows carry mass slightly above 1.0 (1.0001-1.0006) where the raw rows sit at
  1.0000; see the per-run density tables for the mass column.

---

# UWYK — ACIC, all runs

The live UWYK runs on ACIC, pooled here for readability. Per-run detail — the full metric set
(kl_fwd, kl_rev, mass), the resolution-matched control, the interior-mean rows and the
HEADLINE/bridge contrast tables — is in the commented-out `# Raw` block and the live
`# MALC` block below. The HTML-commented blocks for anc=anc and the earlier v3a pass are
excluded here too; they are superseded.

- **raw** = the `# Raw` runs, anchors noanc / v3a / v3b. **MALC** = the `# MALC` runs,
  noanc only. So v3a and v3b have no MALC counterpart.
- realizations=10, ~481 queries each, |tau*|>3: 0.00%. Training-residual sigma mean=0.9951, range=[0.9800, 1.0119].

Rows are grouped by **variant** (the blocks between rules); each block carries the two main
models. **bold** = best in that column across the **whole table**; $\underline{underline}$
= best within its own block. A block whose winner is also the table winner shows the bold
only, so each column carries one bold (or a tied set) and at most one underline per block.
Rows that tie on a value are marked alike.

## Density

<!-- previous grouping: by method family, and including the resolution-matched
control row (`matched bins` / `J=32`). Superseded by the table below.

| variant      | method                     |                          nll |                          l2 |
| ------------ | -------------------------- | ---------------------------: | --------------------------: |
| raw · noanc  | UWYK (x)indep K=1000       |               -0.2536±0.1118 | $\underline{1.5791±0.0566}$ |
| raw · v3a    | UWYK (x)indep K=1000       |           **-0.4253±0.1200** |           **1.4460±0.0673** |
| raw · v3b    | UWYK (x)indep K=1000       |           **-0.4253±0.1200** |           **1.4460±0.0673** |
| MALC · noanc | UWYK (x)indep K=1000       | $\underline{-0.2562±0.1079}$ |               1.5860±0.0573 |
| ------------ | -------------------------- | ---------------------------: | --------------------------: |
| raw · noanc  | UWYK (x)indep matched bins |               -0.2478±0.1092 | $\underline{1.5843±0.0564}$ |
| raw · v3a    | UWYK (x)indep J=32         |           **-0.4146±0.1168** |           **1.4599±0.0655** |
| raw · v3b    | UWYK (x)indep J=32         |           **-0.4146±0.1168** |           **1.4599±0.0655** |
| MALC · noanc | UWYK (x)indep matched bins | $\underline{-0.2495±0.1027}$ |               1.6012±0.0574 |
| ------------ | -------------------------- | ---------------------------: | --------------------------: |
| raw · noanc  | UWYK Joint-2D              | $\underline{-0.4638±0.1481}$ | $\underline{1.3377±0.1164}$ |
| raw · v3a    | Joint-2D J=32              |           **-0.4862±0.1496** |           **1.3138±0.1228** |
| raw · v3b    | Joint-2D J=32              |               -0.4574±0.1596 |               1.3392±0.1179 |
| MALC · noanc | UWYK Joint-2D              |               -0.4391±0.1417 |               1.4091±0.1094 |

-->

| variant      | method               |                          nll |                          l2 |
| ------------ | -------------------- | ---------------------------: | --------------------------: |
| raw · noanc  | UWYK (x)indep K=1000 |               -0.2536±0.1118 |               1.5791±0.0566 |
| raw · noanc  | UWYK Joint-2D        | $\underline{-0.4638±0.1481}$ | $\underline{1.3377±0.1164}$ |
| ------------ | -------------------- | ---------------------------: | --------------------------: |
| MALC · noanc | UWYK (x)indep K=1000 |               -0.2562±0.1079 |               1.5860±0.0573 |
| MALC · noanc | UWYK Joint-2D        | $\underline{-0.4391±0.1417}$ | $\underline{1.4091±0.1094}$ |
| ------------ | -------------------- | ---------------------------: | --------------------------: |
| raw · v3a    | UWYK (x)indep K=1000 |               -0.4253±0.1200 |               1.4460±0.0673 |
| raw · v3a    | Joint-2D J=32        |           **-0.4862±0.1496** |           **1.3138±0.1228** |
| ------------ | -------------------- | ---------------------------: | --------------------------: |
| raw · v3b    | UWYK (x)indep K=1000 |               -0.4253±0.1200 |               1.4460±0.0673 |
| raw · v3b    | Joint-2D J=32        | $\underline{-0.4574±0.1596}$ | $\underline{1.3392±0.1179}$ |

## Point estimates

Original outcome units, from the same predictions, using full-density means; CATE L1 is
per-query MAE, ATE error is unnormalised.

<!-- previous grouping: by method family, and including the resolution-matched
control row (`matched bins` / `J=32`). Superseded by the table below.

| variant      | mean estimator             |                   sqrt PEHE |                     CATE L1 |               ATE abs error |
| ------------ | -------------------------- | --------------------------: | --------------------------: | --------------------------: |
| raw · noanc  | UWYK (x)indep K=1000       | $\underline{3.3019±0.4730}$ | $\underline{2.5166±0.3596}$ | $\underline{1.2980±0.1865}$ |
| raw · v3a    | UWYK (x)indep K=1000       |           **2.6996±0.4419** |           **1.9516±0.3321** |           **0.5677±0.1565** |
| raw · v3b    | UWYK (x)indep K=1000       |           **2.6996±0.4419** |           **1.9516±0.3321** |           **0.5677±0.1565** |
| MALC · noanc | UWYK (x)indep K=1000       |               3.3488±0.4667 |               2.5461±0.3557 |               1.3051±0.1876 |
| ------------ | -------------------------- | --------------------------: | --------------------------: | --------------------------: |
| raw · noanc  | UWYK (x)indep matched bins | $\underline{3.3022±0.4730}$ | $\underline{2.5170±0.3596}$ |               1.2984±0.1866 |
| raw · v3a    | UWYK (x)indep J=32         |           **2.6997±0.4420** |           **1.9517±0.3322** |           **0.5672±0.1568** |
| raw · v3b    | UWYK (x)indep J=32         |           **2.6997±0.4420** |           **1.9517±0.3322** |           **0.5672±0.1568** |
| MALC · noanc | UWYK (x)indep matched bins |               3.3634±0.4619 |               2.5663±0.3543 | $\underline{1.2814±0.1923}$ |
| ------------ | -------------------------- | --------------------------: | --------------------------: | --------------------------: |
| raw · noanc  | UWYK Joint-2D              |           **2.7751±0.5009** | $\underline{1.9218±0.3516}$ | $\underline{0.3581±0.0837}$ |
| raw · v3a    | Joint-2D J=32              | $\underline{2.7840±0.5051}$ |           **1.9171±0.3554** |               0.4155±0.1125 |
| raw · v3b    | Joint-2D J=32              |               2.7929±0.5463 |               1.9562±0.4014 |               0.4121±0.0852 |
| MALC · noanc | UWYK Joint-2D              |               2.8159±0.4911 |               1.9628±0.3447 |           **0.3524±0.0836** |

-->

| variant      | mean estimator       |                   sqrt PEHE |                     CATE L1 |               ATE abs error |
| ------------ | -------------------- | --------------------------: | --------------------------: | --------------------------: |
| raw · noanc  | UWYK (x)indep K=1000 |               3.3019±0.4730 |               2.5166±0.3596 |               1.2980±0.1865 |
| raw · noanc  | UWYK Joint-2D        | $\underline{2.7751±0.5009}$ | $\underline{1.9218±0.3516}$ | $\underline{0.3581±0.0837}$ |
| ------------ | -------------------- | --------------------------: | --------------------------: | --------------------------: |
| MALC · noanc | UWYK (x)indep K=1000 |               3.3488±0.4667 |               2.5461±0.3557 |               1.3051±0.1876 |
| MALC · noanc | UWYK Joint-2D        | $\underline{2.8159±0.4911}$ | $\underline{1.9628±0.3447}$ |           **0.3524±0.0836** |
| ------------ | -------------------- | --------------------------: | --------------------------: | --------------------------: |
| raw · v3a    | UWYK (x)indep K=1000 |           **2.6996±0.4419** |               1.9516±0.3321 |               0.5677±0.1565 |
| raw · v3a    | Joint-2D J=32        |               2.7840±0.5051 |           **1.9171±0.3554** | $\underline{0.4155±0.1125}$ |
| ------------ | -------------------- | --------------------------: | --------------------------: | --------------------------: |
| raw · v3b    | UWYK (x)indep K=1000 |           **2.6996±0.4419** | $\underline{1.9516±0.3321}$ |               0.5677±0.1565 |
| raw · v3b    | Joint-2D J=32        |               2.7929±0.5463 |               1.9562±0.4014 | $\underline{0.4121±0.0852}$ |

**Caveats:**

- **v3a and v3b are identical for `UWYK (x)indep K=1000`** — every nll, l2, sqrt PEHE,
  CATE and ATE value matches to all printed digits, in both datasets (the same holds for
  the matched-bins control, no longer shown). The anchor variant changes only the joint
  model, so wherever those two rows win they win together and are marked alike.
- The **resolution-matched control** (`UWYK (x)indep matched bins` in the noanc runs,
  `UWYK (x)indep J=32` in v3a/v3b) is dropped from these tables — it is the bridge
  contrast's control, not a model. It remains in the commented-out previous grouping above
  each table, and in the per-run sections.
- The joint head is printed as `UWYK Joint-2D` in the noanc runs and `Joint-2D J=32` in
  v3a/v3b. Labels are kept exactly as each run printed them.
- MALC exists only at noanc, so a MALC-vs-raw read is only licensed between the first two
  blocks.
- `Joint-2D interior mean (raw)` rows are dropped here; they appear in the `# Raw` runs
  only, never in `# MALC`.
- MALC rows carry mass slightly above 1.0 (1.0001-1.0006) where the raw rows sit at
  1.0000; see the per-run density tables for the mass column.

---

# DoPFN — IHDP, all runs

The new DoPFN reproduction on IHDP, pooled here for readability. Per-run detail — the full
metric set (kl_fwd, kl_rev, mass), the interior-mean rows, the contrast tables and the
tail notes — is in the commented-out per-run blocks below (`# Do-PFN Old`,
`# Updated DoPFN Reprodcution results`, `# Dopfn MALC reproduction`).

- **new (raw)** = `# Updated DoPFN Reprodcution results`, **new (MALC)** = `# Dopfn MALC
  reproduction`. realizations=90, ~75 queries each, graph=none. Training-residual sigma
  mean=1.0029, range=[0.9550, 1.0683]. Checkpoints:
  `Required_checkpoints/new/dopfn_repro_{1d_J10,1d_J100,joint2d}_step150000.pt`;
  `native` is the library model in both runs.
- The **stale** run (`# Do-PFN Old`, realizations=100, ~75 queries each) is dropped from these tables — it is a different
  query set. It is kept in the commented-out previous grouping above each table.

Rows are grouped by **run** (the blocks between rules), with `repro_1d_J100` pulled out
into its own raw-vs-MALC pair. **bold** = best in that column across the **whole table**;
$\underline{underline}$ = best within its own block. A block whose winner is also the
table winner shows the bold only, so each column carries one bold (or a tied set) and at
most one underline per block. Rows that tie on a value are marked alike.

## Density

<!-- previous grouping: by method family, and including the stale run
(`# Do-PFN Old`, a different query set). Superseded by the table below.

| run        | method                       |                         nll |                          l2 |
| ---------- | ---------------------------- | --------------------------: | --------------------------: |
| stale      | DoPFN (x)indep native        |           **0.1849±0.0354** |               1.4193±0.1118 |
| new (raw)  | DoPFN (x)indep native        | $\underline{0.2115±0.0343}$ |           **1.3222±0.1080** |
| new (MALC) | DoPFN (x)indep native        |               0.2179±0.0341 | $\underline{1.3254±0.1081}$ |
| ---------- | ---------------------------- | --------------------------: | --------------------------: |
| new (raw)  | DoPFN repro_1d_J10 (x)indep  |           **0.6903±0.0540** |           **1.4401±0.1024** |
| new (MALC) | DoPFN repro_1d_J10 (x)indep  | $\underline{0.6917±0.0543}$ | $\underline{1.4406±0.1024}$ |
| ---------- | ---------------------------- | --------------------------: | --------------------------: |
| new (raw)  | DoPFN repro_1d_J100 (x)indep | $\underline{0.1121±0.0247}$ |           **1.2533±0.1141** |
| new (MALC) | DoPFN repro_1d_J100 (x)indep |           **0.1120±0.0246** | $\underline{1.2575±0.1140}$ |
| ---------- | ---------------------------- | --------------------------: | --------------------------: |
| stale      | DoPFN Joint-2D               |               0.5781±0.0475 |               1.6428±0.1104 |
| new (raw)  | DoPFN repro_joint2d Joint-2D | $\underline{0.4802±0.0449}$ | $\underline{1.4690±0.1099}$ |
| new (MALC) | DoPFN repro_joint2d (x)indep |           **0.3514±0.0347** |           **1.3874±0.1122** |

-->

| run        | method                       |                         nll |                          l2 |
| ---------- | ---------------------------- | --------------------------: | --------------------------: |
| new (raw)  | DoPFN (x)indep native        | $\underline{0.2115±0.0343}$ | $\underline{1.3222±0.1080}$ |
| new (raw)  | DoPFN repro_1d_J10 (x)indep  |               0.6903±0.0540 |               1.4401±0.1024 |
| new (raw)  | DoPFN repro_joint2d Joint-2D |               0.4802±0.0449 |               1.4690±0.1099 |
| ---------- | ---------------------------- | --------------------------: | --------------------------: |
| new (MALC) | DoPFN (x)indep native        | $\underline{0.2179±0.0341}$ | $\underline{1.3254±0.1081}$ |
| new (MALC) | DoPFN repro_1d_J10 (x)indep  |               0.6917±0.0543 |               1.4406±0.1024 |
| new (MALC) | DoPFN repro_joint2d (x)indep |               0.3514±0.0347 |               1.3874±0.1122 |
| ---------- | ---------------------------- | --------------------------: | --------------------------: |
| new (raw)  | DoPFN repro_1d_J100 (x)indep |               0.1121±0.0247 |           **1.2533±0.1141** |
| new (MALC) | DoPFN repro_1d_J100 (x)indep |           **0.1120±0.0246** |               1.2575±0.1140 |

## Point estimates

Original outcome units, from the same predictions, using full-density means; CATE L1 is
per-query MAE, ATE error is unnormalised.

<!-- previous grouping: by method family, and including the stale run
(`# Do-PFN Old`, a different query set). Superseded by the table below.

| run        | mean estimator               |                   sqrt PEHE |                     CATE L1 |               ATE abs error |
| ---------- | ---------------------------- | --------------------------: | --------------------------: | --------------------------: |
| stale      | DoPFN (x)indep native        |               6.6013±0.9953 |               4.4945±0.6109 |               2.4695±0.4023 |
| new (raw)  | DoPFN (x)indep native        |           **6.0065±1.0287** |           **4.1687±0.6372** |           **2.2837±0.4223** |
| new (MALC) | DoPFN (x)indep native        | $\underline{6.0133±1.0283}$ | $\underline{4.1739±0.6379}$ | $\underline{2.2864±0.4233}$ |
| ---------- | ---------------------------- | --------------------------: | --------------------------: | --------------------------: |
| new (raw)  | DoPFN repro_1d_J10 (x)indep  | $\underline{9.2522±0.6143}$ | $\underline{7.6938±0.4089}$ | $\underline{6.5640±0.3247}$ |
| new (MALC) | DoPFN repro_1d_J10 (x)indep  |           **9.2498±0.6144** |           **7.6893±0.4089** |           **6.5626±0.3252** |
| ---------- | ---------------------------- | --------------------------: | --------------------------: | --------------------------: |
| new (raw)  | DoPFN repro_1d_J100 (x)indep |           **4.3980±0.7484** | $\underline{3.3470±0.5329}$ | $\underline{0.8957±0.0953}$ |
| new (MALC) | DoPFN repro_1d_J100 (x)indep | $\underline{4.3993±0.7485}$ |           **3.3460±0.5324** |           **0.8914±0.0954** |
| ---------- | ---------------------------- | --------------------------: | --------------------------: | --------------------------: |
| stale      | DoPFN Joint-2D               |               5.7038±0.8368 |               4.4645±0.5963 |           **1.5492±0.1466** |
| new (raw)  | DoPFN repro_joint2d Joint-2D |           **5.4772±0.8300** |           **4.3394±0.5797** | $\underline{1.8038±0.1371}$ |
| new (MALC) | DoPFN repro_joint2d (x)indep | $\underline{5.4801±0.8303}$ | $\underline{4.3408±0.5794}$ |               1.8041±0.1364 |

-->

| run        | mean estimator               |                   sqrt PEHE |                     CATE L1 |               ATE abs error |
| ---------- | ---------------------------- | --------------------------: | --------------------------: | --------------------------: |
| new (raw)  | DoPFN (x)indep native        |               6.0065±1.0287 | $\underline{4.1687±0.6372}$ |               2.2837±0.4223 |
| new (raw)  | DoPFN repro_1d_J10 (x)indep  |               9.2522±0.6143 |               7.6938±0.4089 |               6.5640±0.3247 |
| new (raw)  | DoPFN repro_joint2d Joint-2D | $\underline{5.4772±0.8300}$ |               4.3394±0.5797 | $\underline{1.8038±0.1371}$ |
| ---------- | ---------------------------- | --------------------------: | --------------------------: | --------------------------: |
| new (MALC) | DoPFN (x)indep native        |               6.0133±1.0283 | $\underline{4.1739±0.6379}$ |               2.2864±0.4233 |
| new (MALC) | DoPFN repro_1d_J10 (x)indep  |               9.2498±0.6144 |               7.6893±0.4089 |               6.5626±0.3252 |
| new (MALC) | DoPFN repro_joint2d (x)indep | $\underline{5.4801±0.8303}$ |               4.3408±0.5794 | $\underline{1.8041±0.1364}$ |
| ---------- | ---------------------------- | --------------------------: | --------------------------: | --------------------------: |
| new (raw)  | DoPFN repro_1d_J100 (x)indep |           **4.3980±0.7484** |               3.3470±0.5329 |               0.8957±0.0953 |
| new (MALC) | DoPFN repro_1d_J100 (x)indep |               4.3993±0.7485 |           **3.3460±0.5324** |           **0.8914±0.0954** |

**Caveats:**

- The finite tau grid omits distant tail mass for all four methods in the new raw run (the
  MALC section prints no tail note). NLL is at the observed tau and point errors use exact
  full-density means; l2 (and kl/mass) are finite-grid quantities.
- `dopfn_repro_1d_J10` has p(tau) mass ~0.816 on IHDP in both new runs (0.8158 raw, 0.8157
  MALC) — >1% off 1.0, so the tau grid is clipping real density and its numbers should not
  be read as calibrated. Widen `TAU_EDGES`.
- The joint2d row is labelled `Joint-2D` in the new raw run and `(x)indep` in the new MALC
  run, as printed by each run.
- `interior mean (raw)` rows are dropped here; they remain in the per-run blocks.

---

# DoPFN — ACIC, all runs

The new DoPFN reproduction on ACIC, pooled here for readability. Per-run detail — the full
metric set (kl_fwd, kl_rev, mass), the interior-mean rows, the contrast tables and the
tail notes — is in the commented-out per-run blocks below (`# Do-PFN Old`,
`# Updated DoPFN Reprodcution results`, `# Dopfn MALC reproduction`).

- **new (raw)** = `# Updated DoPFN Reprodcution results`, **new (MALC)** = `# Dopfn MALC
  reproduction`. realizations=8, ~481 queries each, graph=none. Training-residual sigma
  mean=0.9963, range=[0.9806, 1.0119]. Checkpoints:
  `Required_checkpoints/new/dopfn_repro_{1d_J10,1d_J100,joint2d}_step150000.pt`;
  `native` is the library model in both runs.
- The **stale** run (`# Do-PFN Old`, realizations=10, ~481 queries each) is dropped from these tables — it is a different
  query set. It is kept in the commented-out previous grouping above each table.

Rows are grouped by **run** (the blocks between rules), with `repro_1d_J100` pulled out
into its own raw-vs-MALC pair. **bold** = best in that column across the **whole table**;
$\underline{underline}$ = best within its own block. A block whose winner is also the
table winner shows the bold only, so each column carries one bold (or a tied set) and at
most one underline per block. Rows that tie on a value are marked alike.

## Density

<!-- previous grouping: by method family, and including the stale run
(`# Do-PFN Old`, a different query set). Superseded by the table below.

| run        | method                       |                          nll |                          l2 |
| ---------- | ---------------------------- | ---------------------------: | --------------------------: |
| stale      | DoPFN (x)indep native        |               -0.0245±0.1144 |           **1.7014±0.0452** |
| new (raw)  | DoPFN (x)indep native        |           **-0.0374±0.1436** | $\underline{1.7047±0.0549}$ |
| new (MALC) | DoPFN (x)indep native        | $\underline{-0.0255±0.1401}$ |               1.7151±0.0584 |
| ---------- | ---------------------------- | ---------------------------: | --------------------------: |
| new (raw)  | DoPFN repro_1d_J10 (x)indep  |  $\underline{0.1911±0.1491}$ |           **1.7408±0.0557** |
| new (MALC) | DoPFN repro_1d_J10 (x)indep  |            **0.1902±0.1497** |           **1.7408±0.0558** |
| ---------- | ---------------------------- | ---------------------------: | --------------------------: |
| new (raw)  | DoPFN repro_1d_J100 (x)indep |           **-0.0085±0.1400** |           **1.7155±0.0550** |
| new (MALC) | DoPFN repro_1d_J100 (x)indep | $\underline{-0.0032±0.1382}$ | $\underline{1.7212±0.0569}$ |
| ---------- | ---------------------------- | ---------------------------: | --------------------------: |
| stale      | DoPFN Joint-2D               |           **-0.2285±0.2289** |           **1.5219±0.0903** |
| new (raw)  | DoPFN repro_joint2d Joint-2D |                0.2644±0.2574 |               1.8911±0.0684 |
| new (MALC) | DoPFN repro_joint2d (x)indep | $\underline{-0.0454±0.1929}$ | $\underline{1.6979±0.0543}$ |

-->

| run        | method                       |                          nll |                          l2 |
| ---------- | ---------------------------- | ---------------------------: | --------------------------: |
| new (raw)  | DoPFN (x)indep native        | $\underline{-0.0374±0.1436}$ | $\underline{1.7047±0.0549}$ |
| new (raw)  | DoPFN repro_1d_J10 (x)indep  |                0.1911±0.1491 |               1.7408±0.0557 |
| new (raw)  | DoPFN repro_joint2d Joint-2D |                0.2644±0.2574 |               1.8911±0.0684 |
| ---------- | ---------------------------- | ---------------------------: | --------------------------: |
| new (MALC) | DoPFN (x)indep native        |               -0.0255±0.1401 |               1.7151±0.0584 |
| new (MALC) | DoPFN repro_1d_J10 (x)indep  |                0.1902±0.1497 |               1.7408±0.0558 |
| new (MALC) | DoPFN repro_joint2d (x)indep |           **-0.0454±0.1929** |           **1.6979±0.0543** |
| ---------- | ---------------------------- | ---------------------------: | --------------------------: |
| new (raw)  | DoPFN repro_1d_J100 (x)indep | $\underline{-0.0085±0.1400}$ | $\underline{1.7155±0.0550}$ |
| new (MALC) | DoPFN repro_1d_J100 (x)indep |               -0.0032±0.1382 |               1.7212±0.0569 |

## Point estimates

Original outcome units, from the same predictions, using full-density means; CATE L1 is
per-query MAE, ATE error is unnormalised.

<!-- previous grouping: by method family, and including the stale run
(`# Do-PFN Old`, a different query set). Superseded by the table below.

| run        | mean estimator               |                   sqrt PEHE |                     CATE L1 |               ATE abs error |
| ---------- | ---------------------------- | --------------------------: | --------------------------: | --------------------------: |
| stale      | DoPFN (x)indep native        |           **4.1553±0.5434** |               3.3656±0.4353 |           **2.4312±0.3214** |
| new (raw)  | DoPFN (x)indep native        | $\underline{4.1863±0.6864}$ |           **3.3368±0.5497** | $\underline{2.4508±0.3163}$ |
| new (MALC) | DoPFN (x)indep native        |               4.1972±0.6844 | $\underline{3.3441±0.5474}$ |               2.4533±0.3157 |
| ---------- | ---------------------------- | --------------------------: | --------------------------: | --------------------------: |
| new (raw)  | DoPFN repro_1d_J10 (x)indep  |           **4.7559±0.6123** | $\underline{3.7696±0.5274}$ |           **1.6296±0.5955** |
| new (MALC) | DoPFN repro_1d_J10 (x)indep  |           **4.7559±0.6132** |           **3.7678±0.5280** | $\underline{1.6302±0.5954}$ |
| ---------- | ---------------------------- | --------------------------: | --------------------------: | --------------------------: |
| new (raw)  | DoPFN repro_1d_J100 (x)indep |           **3.8339±0.7193** |           **2.9316±0.5915** | $\underline{1.8055±0.3326}$ |
| new (MALC) | DoPFN repro_1d_J100 (x)indep | $\underline{3.8483±0.7190}$ | $\underline{2.9421±0.5909}$ |           **1.8048±0.3352** |
| ---------- | ---------------------------- | --------------------------: | --------------------------: | --------------------------: |
| stale      | DoPFN Joint-2D               |           **3.2313±0.5948** |           **2.4505±0.4441** |           **0.8903±0.1348** |
| new (raw)  | DoPFN repro_joint2d Joint-2D |               4.1702±0.5990 |               3.2844±0.4916 |               2.1617±0.3401 |
| new (MALC) | DoPFN repro_joint2d (x)indep | $\underline{4.1692±0.5992}$ | $\underline{3.2821±0.4913}$ | $\underline{2.1608±0.3397}$ |

-->

| run        | mean estimator               |                   sqrt PEHE |                     CATE L1 |               ATE abs error |
| ---------- | ---------------------------- | --------------------------: | --------------------------: | --------------------------: |
| new (raw)  | DoPFN (x)indep native        |               4.1863±0.6864 |               3.3368±0.5497 |               2.4508±0.3163 |
| new (raw)  | DoPFN repro_1d_J10 (x)indep  |               4.7559±0.6123 |               3.7696±0.5274 |           **1.6296±0.5955** |
| new (raw)  | DoPFN repro_joint2d Joint-2D | $\underline{4.1702±0.5990}$ | $\underline{3.2844±0.4916}$ |               2.1617±0.3401 |
| ---------- | ---------------------------- | --------------------------: | --------------------------: | --------------------------: |
| new (MALC) | DoPFN (x)indep native        |               4.1972±0.6844 |               3.3441±0.5474 |               2.4533±0.3157 |
| new (MALC) | DoPFN repro_1d_J10 (x)indep  |               4.7559±0.6132 |               3.7678±0.5280 | $\underline{1.6302±0.5954}$ |
| new (MALC) | DoPFN repro_joint2d (x)indep | $\underline{4.1692±0.5992}$ | $\underline{3.2821±0.4913}$ |               2.1608±0.3397 |
| ---------- | ---------------------------- | --------------------------: | --------------------------: | --------------------------: |
| new (raw)  | DoPFN repro_1d_J100 (x)indep |           **3.8339±0.7193** |           **2.9316±0.5915** |               1.8055±0.3326 |
| new (MALC) | DoPFN repro_1d_J100 (x)indep |               3.8483±0.7190 |               2.9421±0.5909 | $\underline{1.8048±0.3352}$ |

**Caveats:**

- The finite tau grid omits distant tail mass for all four methods in the new raw run (the
  MALC section prints no tail note). NLL is at the observed tau and point errors use exact
  full-density means; l2 (and kl/mass) are finite-grid quantities.
- p(tau) mass off 1.0 by >1% on ACIC for `dopfn_repro_1d_J10` in both new runs (0.9313 in
  each) and, in the raw run only, `dopfn_repro_joint2d` (0.9899). Under MALC the joint2d
  mass is 0.9902, just inside the 1% threshold, so that run warns for J10 alone. Where the
  grid clips, the numbers are not calibrated — widen `TAU_EDGES`.
- Two exact ties on ACIC: l2 for `repro_1d_J10` is 1.7408 in each run, and its sqrt PEHE
  is 4.7559 in each (SEs differ — ±0.6123 raw, ±0.6132 MALC). Neither wins its block under
  this scheme, so neither carries a mark.
- The joint2d row is labelled `Joint-2D` in the new raw run and `(x)indep` in the new MALC
  run, as printed by each run.
- `interior mean (raw)` rows are dropped here; they remain in the per-run blocks.

---

# CausalPFN

- `causalpfn_j32_random_2d`: /project/6105522/lukez/CFM-for-Decision-Makers/Required_checkpoints/new_checkpoints/cpfn2d_j32_random_step_50000.pt
- `causalpfn_j32_eta0_y01_2d`: /project/6105522/lukez/CFM-for-Decision-Makers/Required_checkpoints/new_checkpoints/cpfn2d_j32_eta0_y01_step50000.pt
- `causalpfn_j1024_headrand_1d`: /project/6105522/lukez/CFM-for-Decision-Makers/Required_checkpoints/new_checkpoints/cpfn1d_j1024_headrand_step_50000.pt
- `causalpfn_j32_1d`: /project/6105522/lukez/CFM-for-Decision-Makers/Required_checkpoints/new_checkpoints/cpfn1d_j32_step50000.pt
- `causalpfn_botharms_1d`: /project/6105522/lukez/CFM-for-Decision-Makers/Required_checkpoints/new_checkpoints/cpfn1d_botharms_step50000.pt

### IHDP — CausalPFN

Truth: generator Gaussian, raw sigma=1. Training-residual sigma (diagnostic, raw units): mean=1.0025, range=[0.9485, 1.0680].

realizations=90, ~75 queries each, graph=none, |tau*|>3: 0.00%


| method                               |                nll |                l2 |            kl_fwd |            kl_rev |          mass |
| ------------------------------------ | -----------------: | ----------------: | ----------------: | ----------------: | ------------: |
| CausalPFN j32_random_2d Joint-2D     |     -0.3649±0.0478 |     1.1127±0.1209 |     0.4839±0.0559 |     7.1073±2.1882 | 1.0000±0.0000 |
| CausalPFN j32_eta0_y01_2d Joint-2D   |     -0.3511±0.0484 |     1.1281±0.1203 |     0.4965±0.0560 |     7.3102±2.2431 | 1.0000±0.0000 |
| CausalPFN j1024_headrand_1d (x)indep |     -0.6009±0.0867 |     0.8198±0.0641 | **0.2313±0.0130** | **0.4934±0.1693** | 1.0000±0.0000 |
| CausalPFN j32_1d (x)indep            |     -0.3847±0.0463 |     1.0816±0.1224 |     0.4558±0.0574 |     7.0317±2.1881 | 1.0000±0.0000 |
| CausalPFN botharms_1d (x)indep       | **-0.6012±0.0880** | **0.8116±0.0627** |     0.2338±0.0127 |     0.5112±0.1979 | 1.0000±0.0000 |


| mean estimator                                         |         sqrt PEHE |           CATE L1 |     ATE abs error |
| ------------------------------------------------------ | ----------------: | ----------------: | ----------------: |
| CausalPFN j32_random_2d Joint-2D                       |     1.1716±0.1031 |     0.8330±0.0506 |     0.2080±0.0157 |
| CausalPFN j32_eta0_y01_2d Joint-2D                     |     1.1525±0.0840 |     0.8314±0.0394 |     0.2309±0.0248 |
| CausalPFN j1024_headrand_1d (x)indep                   |     0.7608±0.1236 |     0.4800±0.0442 | **0.1385±0.0209** |
| CausalPFN j32_1d (x)indep                              |     0.7652±0.1078 |     0.5069±0.0393 |     0.2349±0.0178 |
| CausalPFN botharms_1d (x)indep                         | **0.7192±0.1075** | **0.4732±0.0423** |     0.1929±0.0201 |
| CausalPFN j32_random_2d Joint-2D interior mean (raw)   |     1.1718±0.1032 |     0.8332±0.0507 |     0.2079±0.0157 |
| CausalPFN j32_eta0_y01_2d Joint-2D interior mean (raw) |     1.1527±0.0841 |     0.8315±0.0394 |     0.2308±0.0248 |


### ACIC — CausalPFN

Truth: generator Gaussian, raw sigma=1. Training-residual sigma (diagnostic, raw units): mean=0.9951, range=[0.9800, 1.0119].

realizations=10, ~481 queries each, graph=none, |tau*|>3: 0.00%

| method                               |                nll |                l2 |            kl_fwd |            kl_rev |          mass |
| ------------------------------------ | -----------------: | ----------------: | ----------------: | ----------------: | ------------: |
| CausalPFN j32_random_2d Joint-2D     |     -0.6718±0.0908 |     1.3104±0.0439 |     0.6124±0.0461 |     2.8739±0.4178 | 1.0000±0.0000 |
| CausalPFN j32_eta0_y01_2d Joint-2D   |     -0.7026±0.0888 |     1.2883±0.0450 |     0.5834±0.0439 |     2.6312±0.3788 | 1.0000±0.0000 |
| CausalPFN j1024_headrand_1d (x)indep |     -0.9017±0.1125 |     0.9430±0.0625 |     0.3917±0.0696 | **1.9576±0.5605** | 1.0000±0.0000 |
| CausalPFN j32_1d (x)indep            |     -0.6323±0.0934 |     1.3444±0.0462 |     0.6536±0.0507 |     3.6875±0.7491 | 1.0000±0.0000 |
| CausalPFN botharms_1d (x)indep       | **-0.9054±0.1191** | **0.9332±0.0653** | **0.3880±0.0750** |     1.9935±0.6799 | 1.0000±0.0000 |

<!-- Finite support (causalpfn_j1024_headrand_1d): zero density at tau* in 0.00% of queries (mean over realizations); NLL retains +inf. Grid KL uses the shared numerical density floor.

Finite support (causalpfn_j32_1d): zero density at tau* in 0.00% of queries (mean over realizations); NLL retains +inf. Grid KL uses the shared numerical density floor.

Finite support (causalpfn_botharms_1d): zero density at tau* in 0.00% of queries (mean over realizations); NLL retains +inf. Grid KL uses the shared numerical density floor.

Point errors in original outcome units, from the same predictions. Full-density means except the interior row; CATE L1 is per-query MAE, ATE error is unnormalised. -->

| mean estimator                                         |         sqrt PEHE |           CATE L1 |     ATE abs error |
| ------------------------------------------------------ | ----------------: | ----------------: | ----------------: |
| CausalPFN j32_random_2d Joint-2D                       |     1.8774±0.2549 |     1.2980±0.1605 |     0.6679±0.1164 |
| CausalPFN j32_eta0_y01_2d Joint-2D                     |     1.6544±0.2438 |     1.1225±0.1531 |     0.2375±0.0607 |
| CausalPFN j1024_headrand_1d (x)indep                   |     1.7308±0.2771 |     1.1583±0.1703 |     0.3246±0.0758 |
| CausalPFN j32_1d (x)indep                              | **1.5959±0.2943** | **1.0778±0.1854** | **0.1886±0.0579** |
| CausalPFN botharms_1d (x)indep                         |     1.7295±0.3138 |     1.1484±0.1815 |     0.2479±0.0782 |
| CausalPFN j32_random_2d Joint-2D interior mean (raw)   |     1.8748±0.2546 |     1.2975±0.1605 |     0.6674±0.1163 |
| CausalPFN j32_eta0_y01_2d Joint-2D interior mean (raw) |     1.6533±0.2439 |     1.1223±0.1531 |     0.2371±0.0605 |
  <!-- causalpfn_j32_random_2d: max |finite-grid moment - full mean| = 0.0926288
  causalpfn_j32_eta0_y01_2d: max |finite-grid moment - full mean| = 0.0798311
  causalpfn_j1024_headrand_1d: max |finite-grid moment - full mean| = 0.0178889
  causalpfn_j32_1d: max |finite-grid moment - full mean| = 0.2561
  causalpfn_botharms_1d: max |finite-grid moment - full mean| = 0.0723109 -->

<!-- |        | contrast                                                                    |               dNLL |             dKLrev |              dPEHE |
| ------ | --------------------------------------------------------------------------- | -----------------: | -----------------: | -----------------: |
| as run | model gap as run (causalpfn_j1024_headrand_1d -> causalpfn_j32_random_2d)   |     +0.2298±0.0405 |     +0.9163±0.3176 |     +0.1467±0.1344 |
| as run | model gap as run (causalpfn_j1024_headrand_1d -> causalpfn_j32_eta0_y01_2d) |     +0.1991±0.0411 |     +0.6736±0.2769 | **-0.0763±0.1026** |
| as run | model gap as run (causalpfn_j32_1d -> causalpfn_j32_random_2d)              |     -0.0395±0.0281 |     -0.8136±0.5169 |     +0.2815±0.1875 |
| as run | model gap as run (causalpfn_j32_1d -> causalpfn_j32_eta0_y01_2d)            | **-0.0703±0.0228** | **-1.0563±0.4715** |     +0.0585±0.1500 |
| as run | model gap as run (causalpfn_botharms_1d -> causalpfn_j32_random_2d)         |     +0.2336±0.0454 |     +0.8804±0.4497 |     +0.1479±0.1644 |
| as run | model gap as run (causalpfn_botharms_1d -> causalpfn_j32_eta0_y01_2d)       |     +0.2028±0.0467 |     +0.6377±0.4082 |     -0.0751±0.1373 | -->

_negative = destination method has lower error._
_CausalPFN rows differ in head resolution and in what they were trained on; these are as-run comparisons._



---

<!-- # Raw

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



---

# Do-PFN Old
### IHDP — DoPFN

Truth: generator Gaussian, raw sigma=1. Training-residual sigma (diagnostic, raw units): mean=1.0023, range=[0.9485, 1.0683].

realizations=100, ~75 queries each, graph=none, |tau*|>3: 0.00%

| method                |               nll |                l2 |            kl_fwd |             kl_rev |              mass |
| --------------------- | ----------------: | ----------------: | ----------------: | -----------------: | ----------------: |
| DoPFN (x)indep native | **0.1849±0.0354** | **1.4193±0.1118** | **0.9942±0.0704** |     29.3664±6.5775 |     0.9998±0.0000 |
| DoPFN Joint-2D        |     0.5781±0.0475 |     1.6428±0.1104 |     1.3866±0.0749 | **25.2715±5.9922** | **1.0000±0.0000** |

Point errors in original outcome units, from the same predictions. Full-density means except the interior row; CATE L1 is per-query MAE, ATE error is unnormalised.

| mean estimator                     |         sqrt PEHE |           CATE L1 |     ATE abs error |
| ---------------------------------- | ----------------: | ----------------: | ----------------: |
| DoPFN (x)indep native              |     6.6013±0.9953 |     4.4945±0.6109 |     2.4695±0.4023 |
| DoPFN Joint-2D                     | **5.7038±0.8368** | **4.4645±0.5963** | **1.5492±0.1466** |
| DoPFN Joint-2D interior mean (raw) |     5.7064±0.8374 |     4.4726±0.5969 |     1.5647±0.1461 |
  dopfn_native: max |finite-grid moment - full mean| = 50.3259
  dopfn_joint: max |finite-grid moment - full mean| = 0.000571078

**TAIL NOTE:** The finite tau grid omits distant tail mass for ['dopfn_native']. NLL is evaluated at the observed tau and point errors use exact full-density means; L2/KL/mass are finite-grid quantities. In particular, KL_rev is not the full-support reverse KL when omitted tail mass lies far from the truth.

|              | contrast                                       |           dNLL |         dKLrev |          dPEHE |
| ------------ | ---------------------------------------------- | -------------: | -------------: | -------------: |
| **HEADLINE** | model gap as run (dopfn_native -> dopfn_joint) | +0.3932±0.0336 | -4.0949±1.0072 | -0.8975±0.2663 |

_negative = destination method has lower error._
_Native DoPFN and its joint head use different resolutions; this is an as-run comparison._

### ACIC — DoPFN

Truth: generator Gaussian, raw sigma=1. Training-residual sigma (diagnostic, raw units): mean=0.9951, range=[0.9800, 1.0119].

realizations=10, ~481 queries each, graph=none, |tau*|>3: 0.00%

| method                |                nll |                l2 |            kl_fwd |            kl_rev |              mass |
| --------------------- | -----------------: | ----------------: | ----------------: | ----------------: | ----------------: |
| DoPFN (x)indep native |     -0.0245±0.1144 |     1.7014±0.0452 |     1.2603±0.0704 |    14.9594±1.6459 |     0.9998±0.0000 |
| DoPFN Joint-2D        | **-0.2285±0.2289** | **1.5219±0.0903** | **1.0554±0.1944** | **6.1542±1.3511** | **1.0000±0.0000** |

Point errors in original outcome units, from the same predictions. Full-density means except the interior row; CATE L1 is per-query MAE, ATE error is unnormalised.

| mean estimator                     |         sqrt PEHE |           CATE L1 |     ATE abs error |
| ---------------------------------- | ----------------: | ----------------: | ----------------: |
| DoPFN (x)indep native              |     4.1553±0.5434 |     3.3656±0.4353 |     2.4312±0.3214 |
| DoPFN Joint-2D                     | **3.2313±0.5948** | **2.4505±0.4441** | **0.8903±0.1348** |
| DoPFN Joint-2D interior mean (raw) |     3.2328±0.6041 |     2.4534±0.4550 |     0.9367±0.1365 |
  dopfn_native: max |finite-grid moment - full mean| = 10.1372
  dopfn_joint: max |finite-grid moment - full mean| = 0.000368183

**TAIL NOTE:** The finite tau grid omits distant tail mass for ['dopfn_native']. NLL is evaluated at the observed tau and point errors use exact full-density means; L2/KL/mass are finite-grid quantities. In particular, KL_rev is not the full-support reverse KL when omitted tail mass lies far from the truth.

|              | contrast                                       |           dNLL |         dKLrev |          dPEHE |
| ------------ | ---------------------------------------------- | -------------: | -------------: | -------------: |
| **HEADLINE** | model gap as run (dopfn_native -> dopfn_joint) | -0.2040±0.1471 | -8.8052±1.1319 | -0.9240±0.2902 |

_negative = destination method has lower error._
_Native DoPFN and its joint head use different resolutions; this is an as-run comparison._

---
# Updated DoPFN Reprodcution results


### IHDP — DoPFN

Truth: generator Gaussian, raw sigma=1. Training-residual sigma (diagnostic, raw units): mean=1.0029, range=[0.9550, 1.0683].

realizations=90, ~75 queries each, graph=none, |tau*|>3: 0.00%

- `dopfn_native`: library
- `dopfn_repro_1d_J10`: /home/lukez/CFM-for-Decision-Makers/Required_checkpoints/new/dopfn_repro_1d_J10_step150000.pt
- `dopfn_repro_1d_J100`: /home/lukez/CFM-for-Decision-Makers/Required_checkpoints/new/dopfn_repro_1d_J100_step150000.pt
- `dopfn_repro_joint2d`: /home/lukez/CFM-for-Decision-Makers/Required_checkpoints/new/dopfn_repro_joint2d_step150000.pt

| method                       |               nll |                l2 |            kl_fwd |             kl_rev |              mass |
| ---------------------------- | ----------------: | ----------------: | ----------------: | -----------------: | ----------------: |
| DoPFN (x)indep native        |     0.2115±0.0343 |     1.3222±0.1080 |     0.9473±0.0717 |     26.0474±6.7933 | **0.9998±0.0000** |
| DoPFN repro_1d_J10 (x)indep  |     0.6903±0.0540 |     1.4401±0.1024 |     1.4202±0.0562 |     41.4002±7.2440 |     0.8158±0.0126 |
| DoPFN repro_1d_J100 (x)indep | **0.1121±0.0247** | **1.2533±0.1141** | **0.8481±0.0785** |     25.2750±6.7573 |     0.9944±0.0003 |
| DoPFN repro_joint2d Joint-2D |     0.4802±0.0449 |     1.4690±0.1099 |     1.2194±0.0821 | **24.9072±6.5440** |     0.9981±0.0002 |

Point errors in original outcome units, from the same predictions. Full-density means except the interior row; CATE L1 is per-query MAE, ATE error is unnormalised.

| mean estimator                                   |         sqrt PEHE |           CATE L1 |     ATE abs error |
| ------------------------------------------------ | ----------------: | ----------------: | ----------------: |
| DoPFN (x)indep native                            |     6.0065±1.0287 |     4.1687±0.6372 |     2.2837±0.4223 |
| DoPFN repro_1d_J10 (x)indep                      |     9.2522±0.6143 |     7.6938±0.4089 |     6.5640±0.3247 |
| DoPFN repro_1d_J100 (x)indep                     | **4.3980±0.7484** | **3.3470±0.5329** | **0.8957±0.0953** |
| DoPFN repro_joint2d Joint-2D                     |     5.4772±0.8300 |     4.3394±0.5797 |     1.8038±0.1371 |
| DoPFN repro_joint2d Joint-2D interior mean (raw) |     5.5764±0.8413 |     4.4459±0.5906 |     1.9153±0.1374 |
  dopfn_native: max |finite-grid moment - full mean| = 50.3087
  dopfn_repro_1d_J10: max |finite-grid moment - full mean| = 61.2402
  dopfn_repro_1d_J100: max |finite-grid moment - full mean| = 40.6119
  dopfn_repro_joint2d: max |finite-grid moment - full mean| = 4.72033

**TAIL NOTE:** The finite tau grid omits distant tail mass for ['dopfn_native', 'dopfn_repro_1d_J10', 'dopfn_repro_1d_J100', 'dopfn_repro_joint2d']. NLL is evaluated at the observed tau and point errors use exact full-density means; L2/KL/mass are finite-grid quantities. In particular, KL_rev is not the full-support reverse KL when omitted tail mass lies far from the truth.

|        | contrast                                               |               dNLL |              dKLrev |              dPEHE |
| ------ | ------------------------------------------------------ | -----------------: | ------------------: | -----------------: |
| as run | model gap (dopfn_native -> dopfn_repro_joint2d)        |     +0.2687±0.0423 |      -1.1402±0.8909 |     -0.5293±0.2618 |
| as run | model gap (dopfn_repro_1d_J10 -> dopfn_repro_joint2d)  | **-0.2101±0.0530** | **-16.4930±1.9556** | **-3.7750±0.4891** |
| as run | model gap (dopfn_repro_1d_J100 -> dopfn_repro_joint2d) |     +0.3681±0.0345 |      -0.3678±0.6039 |     +1.0792±0.1160 |

_negative = destination method has lower error._
_DoPFN rows differ in head resolution (and native DoPFN in preprocessing); these are as-run comparisons._

**WARNING:** p(tau) mass off 1.0 by >1% for ['dopfn_repro_1d_J10'] -- the tau grid is clipping real density; widen TAU_EDGES.

### ACIC — DoPFN

Truth: generator Gaussian, raw sigma=1. Training-residual sigma (diagnostic, raw units): mean=0.9963, range=[0.9806, 1.0119].

realizations=8, ~481 queries each, graph=none, |tau*|>3: 0.00%

- `dopfn_native`: library
- `dopfn_repro_1d_J10`: /home/lukez/CFM-for-Decision-Makers/Required_checkpoints/new/dopfn_repro_1d_J10_step150000.pt
- `dopfn_repro_1d_J100`: /home/lukez/CFM-for-Decision-Makers/Required_checkpoints/new/dopfn_repro_1d_J100_step150000.pt
- `dopfn_repro_joint2d`: /home/lukez/CFM-for-Decision-Makers/Required_checkpoints/new/dopfn_repro_joint2d_step150000.pt

| method                       |                nll |                l2 |            kl_fwd |             kl_rev |              mass |
| ---------------------------- | -----------------: | ----------------: | ----------------: | -----------------: | ----------------: |
| DoPFN (x)indep native        | **-0.0374±0.1436** | **1.7047±0.0549** | **1.2606±0.0892** |     15.1882±2.0775 | **0.9998±0.0001** |
| DoPFN repro_1d_J10 (x)indep  |      0.1911±0.1491 |     1.7408±0.0557 |     1.4890±0.0944 |     59.4220±4.0391 |     0.9313±0.0090 |
| DoPFN repro_1d_J100 (x)indep |     -0.0085±0.1400 |     1.7155±0.0550 |     1.2885±0.0869 |     19.6245±2.1997 |     0.9966±0.0013 |
| DoPFN repro_joint2d Joint-2D |      0.2644±0.2574 |     1.8911±0.0684 |     1.5676±0.2006 | **14.5573±2.1428** |     0.9899±0.0038 |

Point errors in original outcome units, from the same predictions. Full-density means except the interior row; CATE L1 is per-query MAE, ATE error is unnormalised.

| mean estimator                                   |         sqrt PEHE |           CATE L1 |     ATE abs error |
| ------------------------------------------------ | ----------------: | ----------------: | ----------------: |
| DoPFN (x)indep native                            |     4.1863±0.6864 |     3.3368±0.5497 |     2.4508±0.3163 |
| DoPFN repro_1d_J10 (x)indep                      |     4.7559±0.6123 |     3.7696±0.5274 | **1.6296±0.5955** |
| DoPFN repro_1d_J100 (x)indep                     | **3.8339±0.7193** | **2.9316±0.5915** |     1.8055±0.3326 |
| DoPFN repro_joint2d Joint-2D                     |     4.1702±0.5990 |     3.2844±0.4916 |     2.1617±0.3401 |
| DoPFN repro_joint2d Joint-2D interior mean (raw) |     4.4226±0.6205 |     3.6012±0.5073 |     2.7921±0.2948 |
  dopfn_native: max |finite-grid moment - full mean| = 10.1069
  dopfn_repro_1d_J10: max |finite-grid moment - full mean| = 12.3885
  dopfn_repro_1d_J100: max |finite-grid moment - full mean| = 1.84353
  dopfn_repro_joint2d: max |finite-grid moment - full mean| = 11.5334

**TAIL NOTE:** The finite tau grid omits distant tail mass for ['dopfn_native', 'dopfn_repro_1d_J10', 'dopfn_repro_1d_J100', 'dopfn_repro_joint2d']. NLL is evaluated at the observed tau and point errors use exact full-density means; L2/KL/mass are finite-grid quantities. In particular, KL_rev is not the full-support reverse KL when omitted tail mass lies far from the truth.

|        | contrast                                               |               dNLL |              dKLrev |              dPEHE |
| ------ | ------------------------------------------------------ | -----------------: | ------------------: | -----------------: |
| as run | model gap (dopfn_native -> dopfn_repro_joint2d)        |     +0.3018±0.1548 |      -0.6308±2.3675 |     -0.0161±0.2388 |
| as run | model gap (dopfn_repro_1d_J10 -> dopfn_repro_joint2d)  | **+0.0733±0.1448** | **-44.8646±4.2062** | **-0.5857±0.4640** |
| as run | model gap (dopfn_repro_1d_J100 -> dopfn_repro_joint2d) |     +0.2729±0.1515 |      -5.0672±2.4751 |     +0.3363±0.2267 |

_negative = destination method has lower error._
_DoPFN rows differ in head resolution (and native DoPFN in preprocessing); these are as-run comparisons._

**WARNING:** p(tau) mass off 1.0 by >1% for ['dopfn_repro_1d_J10', 'dopfn_repro_joint2d'] -- the tau grid is clipping real density; widen TAU_EDGES.


# Dopfn MALC reproduction

### IHDP — DoPFN

Truth: generator Gaussian, raw sigma=1. Training-residual sigma (diagnostic, raw units): mean=1.0029, range=[0.9550, 1.0683].

realizations=90, ~75 queries each, graph=none, |tau*|>3: 0.00%


| method                       |               nll |                l2 |            kl_fwd |             kl_rev |              mass |
| ---------------------------- | ----------------: | ----------------: | ----------------: | -----------------: | ----------------: |
| DoPFN (x)indep native        |     0.2179±0.0341 |     1.3254±0.1081 |     0.9673±0.0716 |     25.7897±6.8165 | **0.9999±0.0000** |
| DoPFN repro_1d_J10 (x)indep  |     0.6917±0.0543 |     1.4406±0.1024 |     1.4220±0.0560 |     41.2970±7.2206 |     0.8157±0.0126 |
| DoPFN repro_1d_J100 (x)indep | **0.1120±0.0246** | **1.2575±0.1140** | **0.8494±0.0787** | **25.0180±6.7366** |     0.9944±0.0003 |
| DoPFN repro_joint2d (x)indep |     0.3514±0.0347 |     1.3874±0.1122 |     1.0936±0.0792 |     26.0094±6.8208 |     0.9981±0.0002 |

Point errors in original outcome units, from the same predictions. Full-density means except the interior row; CATE L1 is per-query MAE, ATE error is unnormalised.

| mean estimator               |         sqrt PEHE |           CATE L1 |     ATE abs error |
| ---------------------------- | ----------------: | ----------------: | ----------------: |
| DoPFN (x)indep native        |     6.0133±1.0283 |     4.1739±0.6379 |     2.2864±0.4233 |
| DoPFN repro_1d_J10 (x)indep  |     9.2498±0.6144 |     7.6893±0.4089 |     6.5626±0.3252 |
| DoPFN repro_1d_J100 (x)indep | **4.3993±0.7485** | **3.3460±0.5324** | **0.8914±0.0954** |
| DoPFN repro_joint2d (x)indep |     5.4801±0.8303 |     4.3408±0.5794 |     1.8041±0.1364 |

**WARNING:** p(tau) mass off 1.0 by >1% for ['dopfn_repro_1d_J10'] -- the tau grid is clipping real density; widen TAU_EDGES.

### ACIC — DoPFN

Truth: generator Gaussian, raw sigma=1. Training-residual sigma (diagnostic, raw units): mean=0.9963, range=[0.9806, 1.0119].

realizations=8, ~481 queries each, graph=none, |tau*|>3: 0.00%


| method                       |                nll |                l2 |            kl_fwd |             kl_rev |              mass |
| ---------------------------- | -----------------: | ----------------: | ----------------: | -----------------: | ----------------: |
| DoPFN (x)indep native        |     -0.0255±0.1401 |     1.7151±0.0584 |     1.2753±0.0895 | **14.4787±2.0263** | **0.9999±0.0000** |
| DoPFN repro_1d_J10 (x)indep  |      0.1902±0.1497 |     1.7408±0.0558 |     1.4876±0.0949 |     59.3346±4.0367 |     0.9313±0.0090 |
| DoPFN repro_1d_J100 (x)indep |     -0.0032±0.1382 |     1.7212±0.0569 |     1.2936±0.0862 |     19.1105±2.0893 |     0.9966±0.0013 |
| DoPFN repro_joint2d (x)indep | **-0.0454±0.1929** | **1.6979±0.0543** | **1.2544±0.1416** |     16.2514±2.0455 |     0.9902±0.0038 |

Point errors in original outcome units, from the same predictions. Full-density means except the interior row; CATE L1 is per-query MAE, ATE error is unnormalised.

| mean estimator               |         sqrt PEHE |           CATE L1 |     ATE abs error |
| ---------------------------- | ----------------: | ----------------: | ----------------: |
| DoPFN (x)indep native        |     4.1972±0.6844 |     3.3441±0.5474 |     2.4533±0.3157 |
| DoPFN repro_1d_J10 (x)indep  |     4.7559±0.6132 |     3.7678±0.5280 | **1.6302±0.5954** |
| DoPFN repro_1d_J100 (x)indep | **3.8483±0.7190** | **2.9421±0.5909** |     1.8048±0.3352 |
| DoPFN repro_joint2d (x)indep |     4.1692±0.5992 |     3.2821±0.4913 |     2.1608±0.3397 |

**WARNING:** p(tau) mass off 1.0 by >1% for ['dopfn_repro_1d_J10'] -- the tau grid is clipping real density; widen TAU_EDGES. -->