# Optimal-Transport Aggregation of CATE Distributions into an ATE Distribution

## 1. Problem statement

We have $N$ units. For each unit $i \in \{1, \dots, N\}$ a foundational prior network has produced a 2D joint density of the potential outcomes $(Y_i(0), Y_i(1))$, which we have fit with MALC_2D:

$$
p_i(y_0, y_1) \quad \text{on the same 2D grid for all } i.
$$

The conditional average treatment effect (CATE) for unit $i$ is the random variable

$$
D_i \;=\; Y_i(1) \;-\; Y_i(0), \qquad D_i \sim f_i(d).
$$

The goal is to aggregate $\{f_1, \dots, f_N\}$ into a single **population ATE distribution** $F_{\mathrm{ATE}}(d)$ that respects the **modal / causal-world structure** of the inputs — i.e. when each $f_i$ is a mixture over $K$ latent "causal worlds", $F_{\mathrm{ATE}}$ should also have $K$ modes, with each mode being the within-world average of the corresponding mode across units.

## 2. Notation

| Symbol | Meaning |
|---|---|
| $i = 1, \dots, N$ | unit (patient) index |
| $k = 1, \dots, K$ | causal-world (mode) index |
| $p_i(y_0, y_1)$ | 2D joint density from MALC_2D |
| $f_i(d)$ | 1D CATE density for unit $i$ |
| $\pi_{i,k}$ | mixing weight of world $k$ for unit $i$, $\sum_k \pi_{i,k} = 1$ |
| $g_{i,k}(d)$ | within-world density of unit $i$ for world $k$ (log-concave) |
| $G_{i,k}(d)$ | CDF of $g_{i,k}$ |
| $Q_{i,k}(\tau) = G_{i,k}^{-1}(\tau)$ | quantile function of $g_{i,k}$ |
| $g_k^\star(d)$ | per-world 1D 2-Wasserstein barycenter |
| $w_k$ | aggregate weight of world $k$, $\sum_k w_k = 1$ |
| $F_{\mathrm{ATE}}(d)$ | final ATE distribution |
| $W_2$ | 2-Wasserstein distance |

## 3. Deriving the 1D CATE density from the 2D joint

For each unit $i$, change variables from $(Y_0, Y_1)$ to $(Y_0, D)$ with $D = Y_1 - Y_0$. The Jacobian determinant is $1$, so the joint density of $(Y_0, D)$ is

$$
\tilde p_i(y_0, d) \;=\; p_i(y_0, y_0 + d).
$$

Marginalising out $y_0$ gives the CATE density

$$
\boxed{\,f_i(d) \;=\; \int_{\mathbb{R}} p_i\!\big(y_0,\; y_0 + d\big)\, dy_0\,}
$$

Numerically, we sample $y_0$ on a fine 1D grid covering the range of $p_i$ and approximate the integral by a Riemann sum along anti-diagonal lines $y_1 = y_0 + d$.

## 4. Per-unit decomposition with MALC_BM

For a chosen $K$, fit each unit's CATE density with MALC_BM:

$$
f_i(d) \;=\; \sum_{k=1}^{K} \pi_{i,k}\, g_{i,k}(d), \qquad \pi_{i,k}\ge 0,\;\; \sum_{k=1}^{K}\pi_{i,k} = 1,
$$

where each $g_{i,k}$ is a **log-concave 1D density** (the MALC_BM per-component fit). After fitting, sort components within each unit by mode location so that the index $k$ refers to the same world across all $i$:

$$
\operatorname{mode}(g_{i,1}) < \operatorname{mode}(g_{i,2}) < \cdots < \operatorname{mode}(g_{i,K}).
$$

## 5. 1D 2-Wasserstein barycenter — closed form

Given $M$ probability measures on $\mathbb{R}$ with densities $\nu_1, \dots, \nu_M$ and weights $\alpha_1, \dots, \alpha_M$ ($\alpha_m \ge 0$, $\sum_m \alpha_m = 1$), the **2-Wasserstein barycenter** is

$$
\nu^\star \;=\; \arg\min_{\mu} \;\sum_{m=1}^{M} \alpha_m\, W_2^2(\mu,\, \nu_m).
$$

In one dimension this has a closed form via quantile functions. If $Q_m(\tau) = F_m^{-1}(\tau)$ is the quantile function of $\nu_m$, then the barycenter has quantile function

$$
\boxed{\,Q^\star(\tau) \;=\; \sum_{m=1}^{M} \alpha_m\, Q_m(\tau), \qquad \tau \in (0, 1).\,}
$$

The barycenter density $\nu^\star$ is recovered from $Q^\star$ by inversion: $\nu^\star$ is the pushforward of the uniform measure on $(0,1)$ by $Q^\star$.

This is a special case of Agueh–Carlier (2011) and has the property that if all $\nu_m$ are translations of a common shape $\nu_0$ — $\nu_m(x) = \nu_0(x - c_m)$ — then $\nu^\star(x) = \nu_0\!\left(x - \sum_m \alpha_m c_m\right)$, i.e. the barycenter is the canonical shape at the mean translation. Linear (mixture) averaging would *blur* the shape across translations.

## 6. Per-world barycenter

For each world $k$, define the **renormalised within-world weights**

$$
\alpha_{i,k} \;=\; \frac{\pi_{i,k}}{\sum_{j=1}^{N} \pi_{j,k}}, \qquad i = 1, \dots, N,
$$

so $\sum_i \alpha_{i,k} = 1$. The per-world 1D Wasserstein barycenter is

$$
\boxed{\,Q_k^\star(\tau) \;=\; \sum_{i=1}^{N} \alpha_{i,k}\, Q_{i,k}(\tau), \qquad g_k^\star(d) \;=\; \big(Q_k^\star\big)_\# \mathrm{Unif}(0,1).\,}
$$

In words: at every probability level $\tau$, the world-$k$ barycenter quantile is the $\pi$-weighted average of the world-$k$ quantiles of each unit.

## 7. Aggregate ATE distribution

The aggregate weight of world $k$ across the sample is

$$
w_k \;=\; \frac{1}{N}\sum_{i=1}^{N} \pi_{i,k}, \qquad \sum_{k=1}^{K} w_k = 1.
$$

The final ATE distribution is

$$
\boxed{\,F_{\mathrm{ATE}}(d) \;=\; \sum_{k=1}^{K} w_k\, g_k^\star(d).\,}
$$

This is a $K$-mode density: world $k$ contributes a single mode, located at the within-world average mode, with mass $w_k$ equal to the empirical mean of the per-unit world-$k$ weights.

## 8. Comparison with the naive linear mixture

The pointwise (Monte-Carlo / linear-mixture) aggregator is

$$
F_{\mathrm{lin}}(d) \;=\; \frac{1}{N}\sum_{i=1}^{N} f_i(d).
$$

Substituting the MALC_BM decomposition of each $f_i$,

$$
F_{\mathrm{lin}}(d) \;=\; \frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K} \pi_{i,k}\, g_{i,k}(d) \;=\; \sum_{k=1}^{K} w_k\, h_k(d),
$$

where the within-world **linear mixture** is

$$
h_k(d) \;=\; \sum_{i=1}^{N} \alpha_{i,k}\, g_{i,k}(d).
$$

The aggregate world weights $w_k$ are identical; the difference is purely within-world:

$$
F_{\mathrm{ATE}}(d) - F_{\mathrm{lin}}(d) \;=\; \sum_{k=1}^{K} w_k \big[\, g_k^\star(d) - h_k(d) \,\big].
$$

If all $g_{i,k}$ coincide ($g_{i,k} = g_k$ for every $i$), then $g_k^\star = h_k = g_k$ and the two aggregators agree. Otherwise — i.e. when there is within-world heterogeneity in shape or location — $h_k$ is a Wasserstein-smeared version of $g_k^\star$ and is wider than $g_k^\star$ (variance is additive in the linear mixture, while the Wasserstein barycenter averages quantiles which is variance-preserving under translation and shrinks within-world variance under shape mismatch).

## 9. Why we cannot skip the decomposition step

A natural question: can we just compute the 1D Wasserstein barycenter of the $\{f_i\}$ directly, without the MALC_BM decomposition?

**No, because the quantile-averaging barycenter does not preserve mixing weights when those weights differ across inputs.** Concretely, consider two bimodal CATE densities with the *same* mode locations but different weights:

$$
f_1(d) = 0.5\,\mathcal{N}(-2, \sigma^2) + 0.5\,\mathcal{N}(+2, \sigma^2), \quad
f_2(d) = 0.3\,\mathcal{N}(-2, \sigma^2) + 0.7\,\mathcal{N}(+2, \sigma^2).
$$

The "correct" average is $F_{\mathrm{target}} = 0.4\,\mathcal{N}(-2, \sigma^2) + 0.6\,\mathcal{N}(+2, \sigma^2)$. But the quantile-averaged barycenter $Q^\star(\tau) = \tfrac12 Q_1(\tau) + \tfrac12 Q_2(\tau)$ behaves as

| $\tau$ range | $Q_1$ | $Q_2$ | $Q^\star$ | comment |
|---|---|---|---|---|
| $(0, 0.3)$ | $\approx -2$ | $\approx -2$ | $\approx -2$ | ✓ left mode |
| $(0.3, 0.5)$ | $\approx -2$ | $\approx +2$ | $\approx 0$ | ✗ phantom mass at 0 |
| $(0.5, 1)$ | $\approx +2$ | $\approx +2$ | $\approx +2$ | ✓ right mode |

The interval $\tau \in (0.3, 0.5)$ where one quantile axis "switches modes" before the other contributes a spurious central mode with mass $0.2$. The decomposition step circumvents this by aligning the world axis explicitly: world-1 mass is averaged with world-1 mass, world-2 mass with world-2 mass, independently of any quantile mismatch.

## 10. Algorithm summary

```
Inputs:
    N units, each with 2D MALC_2D fit  p_i(y_0, y_1)
    common 1D grid for d
    K (number of causal worlds)

For each unit i:
    1. Compute CATE density:
           f_i(d) = ∫ p_i(y_0, y_0 + d) dy_0
    2. Decompose with MALC_BM at fixed K:
           f_i(d) = Σ_k π_{i,k} g_{i,k}(d)
       Sort components by mode location.

For each world k = 1..K:
    3. Compute renormalised weights  α_{i,k} = π_{i,k} / Σ_j π_{j,k}.
    4. Compute the 1D Wasserstein barycenter:
           Q_k*(τ) = Σ_i α_{i,k} Q_{i,k}(τ)
       (numerically: average quantile functions evaluated on a uniform τ-grid)
    5. Recover the barycenter density g_k* from Q_k*.

Combine:
    6. Aggregate world weights  w_k = (1/N) Σ_i π_{i,k}.
    7. Final ATE distribution
           F_ATE(d) = Σ_k w_k g_k*(d).

Validation:
    - Sample-size ablation: repeat for N' < N, compare F_ATE^{(N')} to F_ATE^{(N)} or to truth.
    - Ground-truth simulation: known DGP for (π_{i,k}, g_{i,k}); report L2(F_ATE, F_true)
      and L2(F_lin, F_true).
```

## 11. References

- Agueh, M., Carlier, G. (2011). *Barycenters in the Wasserstein space.* SIAM J. Math. Anal.
- Balabdaoui, F., Jankowski, H., Rufibach, K., Pavlides, M. (2013). *Asymptotics of the discrete log-concave maximum likelihood estimator and related applications.* JRSS B. *(used for `logConDiscrMLE`)*
- Cule, M., Samworth, R., Stewart, M. (2010). *Maximum likelihood estimation of a multidimensional log-concave density.* JRSS B. *(used for the 2D MLE)*
- Dümbgen, L., Rufibach, K. (2009). *Maximum likelihood estimation of a log-concave density: basic properties and uniform consistency.* Bernoulli. *(used for the 1D continuous MLE)*
