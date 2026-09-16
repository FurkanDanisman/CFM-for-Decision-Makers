# paper_appendix

`appendix_density.tex` — the derivation behind every calibration number:
how each model's predictive becomes a distribution over
$\tau = Y^{(1)} - Y^{(0)}$, and how that distribution is scored.

Written in parameters, not measurements. Bin counts are $K$, grids are
$\Delta$, levels are $\alpha$, normalisation constants are $\alpha$ or
$\sigma_t$ — nothing in it changes when the numbers do.

## Using it

Drop into the ICLR main file:

```latex
\appendix
\input{appendix_density}
```

It defines no preamble and opens at `\section`. It needs `amsmath`,
`booktabs`, and `\argmin` (`\DeclareMathOperator*{\argmin}{arg\,min}`).

Or compile alone:

```bash
cd Reproduce/paper_appendix && pdflatex main_standalone
```

## Contents

| § | what |
|---|---|
| Notation | bar distributions, bins, representative points, the lattice condition |
| 2D heads | anti-diagonal projection of the joint — no independence assumption |
| 1D heads | independence convolution, comonotonic coupling, and why they differ by $\rho$ |
| Standardisation | why per-arm scaling breaks $\tau$ and pooled does not |
| Non-uniform bins | CDF re-expression; why both arms need one shared, untruncated grid |
| Model-specific | what each method emits; full-support tails; point/density consistency |
| CATE $\to$ ATE | the 1-D Wasserstein barycenter, and why not a mixture |
| Metrics | coverage, length, $\mathrm{IS}_\alpha$, CRPS/WIS, the sd-ratio units check |

## Two claims worth knowing are load-bearing

Both are stated in the appendix and both were found empirically, not assumed:

- **Coupling.** Eq. (independence vs comonotonic) — a 1D head cannot
  determine the law of $\tau$ from two marginals. The gap is governed by
  $\rho$ through $\operatorname{Var}(\tau) = \sigma_0^2 + \sigma_1^2 -
  2\rho\sigma_0\sigma_1$, which is the quantitative case for a 2D head.
- **Representative points.** A method's density must use the same bin
  representatives as its own point estimate, or PEHE and the interval scores
  describe different estimators and can rank methods oppositely. The residual
  is given in closed form and vanishes on a uniform grid.
