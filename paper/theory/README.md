# paper/theory/

LaTeX source for the two theoretical sections.

| File | Purpose |
|---|---|
| `variance_reduction.tex` | Theorem 1: joint estimator strictly dominates two-marginal in variance whenever ρ > 0 (matches `plot_variance_reduction.py`) |
| `malc_multimodal.tex` | Extending log-concave MLE to a K-component MALC mixture — EM steps, bootstrap-based K selection, consistency proposition |

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
```

## Usage

Include both directly from your main tex file:

```tex
\input{paper/theory/variance_reduction.tex}
\input{paper/theory/malc_multimodal.tex}
```

Cross-reference from the body:
- `Theorem~\ref{thm:variance-ratio}`
- `Proposition~\ref{prop:malc-consistency}`
- `Figure~\ref{fig:variance-reduction}` — populate this with `benchmarks/plots/joint_vs_marginals/variance_reduction.png`
- `Section~\ref{sec:mixture-scm}` — referenced by the MALC "Limitation" paragraph; point at wherever your mixture-SCM diagnostic figure sits
