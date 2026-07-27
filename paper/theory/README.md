# paper/theory/

LaTeX source for the two theoretical sections.

| File | Purpose |
|---|---|
| `variance_reduction.tex` | Theorem 1: joint estimator strictly dominates two-marginal in variance whenever ρ > 0 (matches `plot_variance_reduction.py`) |
| `malc_multimodal.tex` | Extending log-concave MLE to a K-component MALC mixture — EM steps, bootstrap-based K selection, consistency proposition |
| `bar_distribution_2d.tex` | Methodology section for the 2D BarDistribution head: three-region density, tail Gaussians with Sheppard-normalised orthant probabilities, correlation derived from the histogram, and the negative log-density loss |

## References that should live in your `.bib`

```
@article{cule2010maximum,
  author  = {Cule, Madeleine and Samworth, Richard and Stewart, Michael},
  title   = {Maximum likelihood estimation of a multi-dimensional log-concave density},
  journal = {Journal of the Royal Statistical Society: Series B},
  volume  = {72}, number = {5}, pages = {545--607}, year = {2010}
}

@article{samworth2018recent,
  author  = {Samworth, Richard J.},
  title   = {Recent progress in log-concave density estimation},
  journal = {Statistical Science},
  volume  = {33}, number = {4}, pages = {493--509}, year = {2018}
}

@article{doss2016global,
  author  = {Doss, Charles R. and Wellner, Jon A.},
  title   = {Global rates of convergence of the {MLE}s of log-concave and $s$-concave densities},
  journal = {Annals of Statistics},
  volume  = {44}, number = {3}, pages = {954--981}, year = {2016}
}

@article{sheppard1899application,
  author  = {Sheppard, W. F.},
  title   = {On the application of the theory of error to cases of normal distribution and normal correlation},
  journal = {Philosophical Transactions of the Royal Society of London A},
  volume  = {192}, pages = {101--167}, year = {1899}
}
```

## Usage

Include all three sections directly from your main tex file:

```tex
\input{paper/theory/bar_distribution_2d.tex}
\input{paper/theory/malc_multimodal.tex}
\input{paper/theory/variance_reduction.tex}
```

A natural order is BarDistribution → MALC → variance reduction, since MALC
appears as an inference-time step referenced in the BarDistribution section
and the variance-reduction theorem uses the joint model that both establish.

Cross-reference from the body:
- `Theorem~\ref{thm:variance-ratio}`
- `Proposition~\ref{prop:malc-consistency}`
- `Figure~\ref{fig:variance-reduction}` — populate this with `benchmarks/plots/joint_vs_marginals/variance_reduction.png`
- `Section~\ref{sec:mixture-scm}` — referenced by the MALC "Limitation" paragraph; point at wherever your mixture-SCM diagnostic figure sits
