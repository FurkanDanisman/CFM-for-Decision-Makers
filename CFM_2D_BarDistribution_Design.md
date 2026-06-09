# CFM 2D Bar Distribution Design

> **Status note (2026-06-09):** This document was first drafted to follow the
> design in `CFM_for_Decision_Makers.pdf`, which placed MALC (the 2D bivariate
> log-concave density estimator, formerly "DensOLog") **inside the training
> loop** to provide a smooth density at the exact continuous target point for
> the inner-inner region. That intention turned out to be infeasible at UWYK
> training scale (50K steps × ~8K queries/step = 4×10⁸ MALC fits; CLARABEL is
> ~13s per fit and not GPU-vectorizable). The design has been revised:
>
> - **Training**: histogram density (`p_mat[j0, j1] / bin_width²`) for the
>   inner-inner region — fully differentiable, ms-scale on GPU, identical in
>   structure to UWYK's 1D BarDistribution loss extended to 2D.
> - **Inference**: MALC smooths the K×K p_mat into a continuous log-concave
>   density. Only at this step does MALC run.
>
> The three-region structure (inner-inner / mixed / both-outside), the tail
> design, and the 9 mixture weights are unchanged from the PDF. The original
> PDF describes the prior intent; this file describes the revised one. Where
> the two disagree, this file is authoritative for the current implementation.

## Starting Point — UWYK 1D BarDistribution

UWYK's 1D BarDistribution outputs K+4 values per query point:
- K+2 logits: [left_tail, bar_0, ..., bar_{K-1}, right_tail]
- 2 raw tail scale params: sL_raw, sR_raw → positive via softplus

The density is:
- Y < -1: left half-Gaussian `2 × p_L × N(y; -1, sL)`
- Y ∈ [-1,1]: histogram density `p_k / bin_width` (uniform inside each bar);
  at inference can be smoothed by 1D MALC/logconDens
- Y > +1: right half-Gaussian `2 × p_R × N(y; 1, sR)`

Loss = `-average_log_prob` (proper log-density, not plain cross-entropy over
discrete bins — the `log(bin_width)` correction matters).

Tail scales:
```
sL = base_s_left × (softplus(sL_raw) + floor)
sR = base_s_right × (softplus(sR_raw) + floor)
```
where base_s_left = base_s_right = bin width. Bin width is just the initialization anchor — the model learns the actual scale through backprop.

---

## Why Tails Are Necessary

Y_obs is min-max scaled to [-1,1] using training context min/max. Y_intv (or Y_do0, Y_do1) uses the same transform but can land outside [-1,1] because the interventional distribution differs from the observational one. Without tails, out-of-range values get density 0 → log(0) = -∞ → training breaks.

The histogram inside [-1,1] has compact support — density is exactly 0 outside
the data range. This creates a hard discontinuity at the boundary. Tails solve
this by creating smooth decay beyond the boundary.

Why Gaussian tails (not t-distribution, not Cauchy):
- sL is learned — the Gaussian adapts its width to match how far out values land
- Mean remains finite and usable
- Gaussian log-density is cheap to compute
- t-SNE's argument for heavy tails is specific to the crowding problem — does not apply here

---

## The Data Generation Fix

Current `generate_paired_samples.py` previously clamped Y_do0 and Y_do1 to
[-1,1]. This is wrong — it hides the out-of-range values that tails are
designed to handle. The clamp has been removed. Y_do0 and Y_do1 are scaled
with the same transform as Y_obs but NOT clamped.

---

## 2D Extension — The Core Challenge

Y_do0 and Y_do1 are paired potential outcomes for the same individual. They share the same exogenous noise. They are NOT independent — treating them as a product of two 1D densities is wrong.

### Methods Considered and Rejected

| Method | Reason Rejected |
|--------|----------------|
| Product of independent 1D densities | Y_do0 and Y_do1 are not independent |
| Zero weight for out-of-range (histogram gives 0) | Hard discontinuity at boundary — exactly what tails solve |
| Skip mixed-case training points | Wastes training signal |
| Gaussian copula over two 1D distributions | Removes the purpose of 2D joint density entirely |
| Single shared covariance for all tail regions | Too restrictive — left and right tails may differ |
| Low-rank factorization of K² grid | Diagonal correlation (typical for paired potential outcomes sharing exogenous noise) is the worst case for low-rank approximation — would smear the diagonal into rectangular blobs |
| K=1000 per axis (K²=1M cells) | Output head ~64M params, sparse signal, won't fit on a single GPU |
| MALC inside the training loop | CLARABEL is ~13s/fit, not differentiable through autograd; at UWYK scale (50K steps × 8K queries) this is ~5+ years of compute even fully CPU-parallelized |

### Chosen approach

**Per axis K = 100 bins** (J²=10,000 inner cells). Plain softmax over the K²
cells gives a tractable output_dim (10,013 = K² + 9 + 4), a 641K-param output
head with d_model=64, and ~80 MB of logits memory per microbatch at UWYK
batch sizes. MALC is reserved for **inference** — it interprets the coarse
K×K p_mat as a continuous log-concave density at any continuous (y0, y1).

---

## Final Design — Three-Case Structure

The 2D space is divided into three regions based on where (Y_do0, Y_do1) falls:

### Case 1 — Both inside [-1,1] × [-1,1]
```
Training:  density = p_mat[j0, j1] / bin_width²     (histogram, differentiable)
Inference: density = dmalc_2d(MALC_fit(p_mat), y0, y1)   (continuous, smooth)
```
The coarse K×K p_mat carries the full joint correlation captured
non-parametrically. At inference, MALC smooths it into a piecewise-linear
log-concave density. This is the common case.

### Case 2 — Mixed (one inside, one outside)
```
f(y0, y1) = half_Gaussian(y_outside) × f_inner(y_inside | y_outside = boundary)
```

The conditional at the boundary, computed from the discrete p_mat:
```
f_inner(y1 | y0 = -1) ≈ p_mat[0, j1] / (Σ_j p_mat[0, j]) / bin_width
```

Evaluate p_mat at the boundary row (j0=0 or j0=J-1) and normalize over the
in-range axis. At inference MALC can produce the smoother analytical version
of the boundary conditional; during training the discrete approximation
suffices.

**Accepted approximation:** p(y1 | y0 outside) ≈ p(y1 | y0 = boundary). This is self-correcting — the half-Gaussian assigns negligible probability to very extreme y0, so the approximation only matters near the boundary where it is most accurate.

### Case 3 — Both outside [-1,1]
```
f(y0, y1) = 2D half-Gaussian(y0, y1 ; σ_L0, σ_R0, σ_L1, σ_R1, ρ)
```

Full bivariate Gaussian, captures correlation in the tails. Normalized by the
appropriate quadrant probability from Sheppard's theorem:

- Same-direction corners (L0L1, R0R1): `1/4 + arcsin(ρ)/(2π)`
- Opposite-direction corners (L0R1, R0L1): `1/4 − arcsin(ρ)/(2π)`

---

## Tail Parameters

Four scale parameters (directly mirroring 1D UWYK per axis):
```
σ_L0 — left tail scale for Y_do0
σ_R0 — right tail scale for Y_do0
σ_L1 — left tail scale for Y_do1
σ_R1 — right tail scale for Y_do1
```

All transformed via: `σ = bin_width × (softplus(σ_raw) + floor)`

One correlation parameter:
```
ρ = correlation(Y_do0, Y_do1) derived from p_mat
```

ρ is NOT a separately learned parameter. It is computed from the K×K bin
probabilities — the Pearson correlation implied by the joint discrete
distribution. This guarantees consistency across all three regions. The same
correlation structure that p_mat captures inside is used in the tails.

---

## Why ρ Is Derived, Not Learned

Three different correlation representations exist across the three regions:
- Inner-inner: non-parametric (K×K p_mat; smoothed by MALC at inference) — very expressive
- Mixed: boundary conditional (row/column of p_mat) — semi-parametric
- Both-outside: bivariate Gaussian with ρ — fully parametric

If ρ were learned freely, the model could represent inconsistent correlations in different regions. Deriving ρ from p_mat guarantees consistency — the same underlying correlation (driven by shared exogenous noise) is used everywhere. One fewer parameter to learn, internally consistent model.

The assumption: correlation in the tail is the same as in the inner region. This is reasonable since both come from the same shared exogenous noise.

---

## Mixture Weights

Nine regions in total (3×3 grid):
- 1 inner-inner
- 4 mixed (L0-inner, R0-inner, inner-L1, inner-R1)
- 4 both-outside (L0L1, L0R1, R0L1, R0R1)

Each region has a mixture weight. Weights sum to 1 via softmax. Output by the model per query point — learned through backpropagation.

---

## Final per-query output (K=100)

```
output_dim = K² + 9 + 4 = 10,013
  [0 : K²]            inner bin logits  → softmax → p_mat (K×K)
  [K² : K²+9]         region logits     → softmax → w_region (9 weights)
  [K²+9 : K²+13]      tail raw          → softplus → σ_L0, σ_R0, σ_L1, σ_R1
```

Output head (with d_model = 64): 641K params. Per-microbatch logits memory
at UWYK scale (B=8, M=250): ~80 MB. Tractable.

---

## What Needs to Change in the Codebase

1. **`generate_paired_samples.py`** — Remove the clamp on Y_do0 and Y_do1. Keep the scale transform, drop the hard clamp. *(Done.)*

2. **`losses/BarDistribution2D.py`** — Implements:
   - Histogram density for inner-inner case (training)
   - Boundary conditional for mixed case (from p_mat rows/columns)
   - Bivariate Gaussian with (σ_L0, σ_R0, σ_L1, σ_R1, ρ) for both-outside case
   - 9 mixture weights
   - ρ derived from p_mat
   - `fit_malc_inner` + `eval_density_2d` helpers for inference-time MALC smoothing.

3. **`train_cfm_small.py`** — Uses `neg_log_prob_2d` and `output_dim = total_params(J)` where `J=100`.

4. **`models/InterventionalPFN.py`** — Output dim matches `total_params(J) = 10,013` at K=100.

---

## Divergence from CFM_for_Decision_Makers.pdf

The PDF (Section 4, paired potential-outcome head + DensOLog refinement)
implies MALC/DensOLog runs as part of the training-time density. We have
deviated from that plan for tractability reasons: MALC fits a non-trivial
convex program per query and cannot be vectorized across queries on the GPU.
At UWYK training scale this is infeasible, even with aggressive CPU
parallelism and stripped-down MALC variants.

The current design preserves every architectural commitment in the PDF
(paired joint head, 3-region structure, tails, derived ρ, shared exogenous
noise coupling, 9 mixture weights) — only the **role of MALC** has changed:
inference-time smoother rather than training-time density. The discrete K×K
histogram during training is a faithful coarse approximation; MALC interprets
it as a continuous log-concave density only when needed for downstream
queries (CATE distribution, highest-density region, etc.).
