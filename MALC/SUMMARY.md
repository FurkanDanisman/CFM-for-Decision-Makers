# MALC_2D (Python) — Summary

Pure-Python implementation of MALC_2D: a mixture of 2D log-concave densities
fit to binned probability data.

## What it does

Input: a matrix `p_mat` of joint bin probabilities over a rectangular grid
defined by breakpoints `grid_x` and `grid_y` (sums to 1; raw observations are
*not* available — only the binned probabilities).

Output: a mixture of K log-concave components

```
f(x, y) = Σ_{k=1..K}  π_k · f_k(x, y)
```

where each `f_k` is a 2D log-concave density (Cule–Samworth–Stewart MLE) and K
is selected by BIC.

## Algorithm (per-component fit, 4 steps)

1. **Marginal EM mean correction** per dimension — recovers the true latent
   mean μ̂ from the binned proportions via a truncated-normal score
   fixed-point iteration.
2. **Beta jitter calibration** — within-bin displacement modeled as
   `δ · Beta(α+β, α−β)` with β chosen so the Beta mean matches μ̂.
3. **Synthetic point generation** — sample B bins from the joint `p_mat`
   (preserves 2D correlation), add independent Beta jitter in x and y.
4. **2D log-concave MLE** on the B synthetic points (CSS estimator) — a
   piecewise-linear concave function on the convex hull of the points.

## Mixture EM

- **Init**: peak-detection on `p_mat` + non-maximum suppression to find
  distinct modes, then **k-means++-style deterministic augmentation** when K
  exceeds the number of distinct modes.
- **E-step**: integrate each `f_k` over each bin, form responsibilities γ_jk.
- **M-step**: refit each component on the responsibility-weighted `p_mat * γ_k`;
  update π_k.
- **Convergence**: patience-based (stop after 5 non-improving iterations) —
  needed because the Monte-Carlo M-step injects noise into log-likelihood.
- **K=1 special case**: no EM, single fit.

## K selection

Two-stage BIC pipeline. BIC formula:

```
BIC(K) = −2 · loglik · n_eff + (3K−1) · log(n_eff),   n_eff = 1 / Σ p_mat²
```

**Final configuration: `B_select = B_fit = 500`, `max_K = 5`.**

Earlier work explored `B_select=100, B_fit=300` (the original two-stage idea),
but the BIC scan at B=100 was unreliable — small convex hulls per component
left non-negligible `p_mat` mass in bins with predicted density 0, dominating
the loglik and making K selection sensitive to RNG. At B=500 BIC selection
is stable; the extra cost is paid back by reliability.

## Implementation choices

### 2D log-concave MLE (`mlelcd_2d`)

The R package LogConcDEAD uses Shor's r-algorithm to solve the CSS MLE. We
solve the same convex program directly:

- **Triangulation**: Delaunay (scipy.spatial.Delaunay).
- **Variables**: `y_i = log f(x_i)` at each input point.
- **Concavity**: a linear inequality at every interior edge — the apex of one
  triangle must lie at or below the linear extrapolation of the neighbouring
  triangle.
- **Integral** `∫_T exp(linear interp)` on each triangle: Dunavant order-5
  quadrature (7 points, positive weights, exact for degree-5 polynomials).
- **Solve**: build the sparse exponential-cone program directly and hand it to
  CLARABEL — no CVXPY DSL layer (the DSL canonicalization dominates wall-time
  for problems of this size).

Validated against R `LogConcDEAD::mlelcd` on the same 100-point input:
density values agree to ~1e-3 at interior points; differences at convex-hull
vertices are larger but don't affect density (those are points where the
function extends to −∞).

### Performance optimisations

1. Direct CLARABEL call (no CVXPY DSL).
2. ProcessPoolExecutor for Stage 1 BIC scan — K=1..max_K fits in parallel.
3. Looser solver tolerance during Stage 1 (1e-5 vs 1e-8 for final fit) —
   BIC ordering doesn't need high accuracy.
4. Smaller default `bin_n_eval=24` for the E-step density integration grid.

Combined effect: ~5–6× faster than a CVXPY+serial implementation.

## Files (correct final structure)

The Python port lives in `python/`. The two files that hold the final
implementation:

| File | Role |
|---|---|
| `python/log_concave_2d_fast.py` | 2D log-concave MLE + density evaluation (direct CLARABEL exp-cone). The single source of truth for `mlelcd_2d_fast` and `dlcd_2d_fast`. |
| `python/malc_2d.py` | Full MALC_2D pipeline: per-component fit, mixture EM, BIC, two-stage K selection (with parallel Stage 1). Defaults: `B_select=B_fit=500`, `max_K=5`. |

Supporting / illustrative files (also in `python/`):

| File | Role |
|---|---|
| `python/benchmark.py` | Runs the original 3-scenario R benchmark in Python. |
| `python/examples.py` | Four worked examples mirroring `MALC_2D_Examples.R`. |
| `python/b_sweep.py` | B-sweep across 4 distribution families. |
| `python/b_sweep_full.py` | B-sweep × n_eval comparison. |
| `python/compare_with_r.py` | Side-by-side Python ↔ R contour plots. |
| `python/log_concave_2d.py` | Original CVXPY-based MLE, kept for reference. Not used at runtime. |

Original R code is in `MALC_2D_Algorithm.R`, `MALC_2D_Examples.R`,
`MALC_2D_Benchmark.R`, `MALC_2D_Notes.md` (sibling to `python/`).

## Final benchmark results

Using the new defaults (B=500, max_K=5, parallel Stage 1):

| Scenario | True K | Selected K | L2 | Time (s) |
|---|---|---|---|---|
| Trimodal Normal | 3 | 5 | 0.081 | 16.4 |
| Beta(8,2)/(2,8) Mixture | 2 | **2 ✓** | 0.290 | 13.7 |
| Gamma(5)/Gamma(10) Mixture | 2 | 5 | 0.045 | 16.0 |
| t(df=3) Mixture | 2 | 4 | 0.054 | 16.4 |

Notes:
- L2 is the L2 error against the true density on an 80×80 evaluation grid.
- Where K_selected > K_true, this is the **BIC over-splitting** known
  limitation — the log-concave shape can only approximate each mode, so
  splitting one mode into multiple components fits binned data better in
  log-likelihood terms even when the true mixture has fewer components.
- L2 is the more reliable performance metric. Across the four scenarios it
  ranges from 0.045 (Gamma) to 0.290 (Beta — bounded support hurts).
- For the Trimodal case the **true K=3 fit gives L2 ≈ 0.05** when forced,
  whereas BIC's K=5 choice produces L2 ≈ 0.08. If you care more about the
  density shape than the K count, you can pass `K=K_known` directly to
  `MALC_2D(...)`.

## Known limitations

1. **BIC over-splits** at finite B — the safer failure mode (over-splitting
   produces a suboptimal but structurally correct density estimate, vs
   under-splitting which misses entire modes).
2. **Each component must be log-concave (unimodal).** A truly bimodal
   component cannot be represented as a single `f_k`. The mixture handles
   multimodality across components, not within.
3. **Boundary-concentrated distributions** (e.g., Beta with mass piled at 0
   or 1) are hard for the log-concave shape — the estimator can't represent
   the sharp boundary spike, leading to higher L2.
4. **The Monte-Carlo M-step** means log-likelihood is not monotone. Patience
   convergence mitigates this but EM may return at a slightly suboptimal
   state if the patience budget is exhausted by noise.
5. **No support for components with disconnected support.**

## How to use

```python
import numpy as np
from malc_2d import MALC_2D, dmalc_2d, plot_malc_2d

grid_x = np.arange(-5, 5 + 1e-9, 0.5)
grid_y = np.arange(-5, 5 + 1e-9, 0.5)
# p_mat: (n_y, n_x) bin probabilities summing to 1

# Default: B_select=B_fit=500, max_K=5, parallel Stage 1
fit = MALC_2D(p_mat, grid_x, grid_y, seed=20180621)

print(f"Selected K = {fit.K}")
print(f"Mixing weights: {fit.pi}")

# Evaluate the mixture density at arbitrary 2D points
pts = np.array([[0.0, 0.0], [1.0, 1.0]])
density = dmalc_2d(fit, pts)

# Contour plot
plot_malc_2d(fit, true_pdf=my_pdf, main="My fit")
```

If K is known, skip the BIC scan:

```python
fit = MALC_2D(p_mat, grid_x, grid_y, K=3, seed=20180621)
```

See `example.py` in this folder for a runnable demonstration.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy clarabel matplotlib
```

Python 3.10+. CLARABEL is the only solver dependency (CVXPY is not required).
