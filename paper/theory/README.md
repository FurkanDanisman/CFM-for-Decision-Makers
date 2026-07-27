# paper/theory/

Consolidated LaTeX source for the theoretical section of the paper.

| File | Purpose |
|---|---|
| `theory.tex` | Single self-contained methodology + theory section covering (1) the 2D BarDistribution head, (2) multimodal density recovery with MALC, (3) population aggregation via the 2-Wasserstein barycenter, and (4) the joint-vs-marginal variance-reduction theorem. |

## What is (and isn't) claimed

- **Formal statements** (theorem / proposition / corollary) are limited to
  results that either follow from moment calculations, are stated with a
  citation, or are direct computations from the code.
- The variance-reduction theorem uses only finite second moments, so it
  applies unchanged to any log-concave / exponential-family / mixture pair
  with finite variance. Bivariate normality is used only to describe the
  Monte-Carlo simulation.
- The 1D 2-Wasserstein barycenter closed form is cited to Agueh & Carlier
  (2011).
- Single-component MALC and its theoretical properties are cited to
  Danisman, Jankowski & de Souza (2026). The mixture extension (K components,
  bin-integrated E-step, mode-based initialisation, BIC selection) is
  described as implemented in the code — no additional asymptotic claim
  is made about the mixture procedure.
- Sheppard's identity (orthant probabilities of a centred bivariate normal)
  is cited to Sheppard (1899).

## References that should live in your `.bib`

```
@article{danisman2026malc,
  author  = {Danisman, Furkan and Jankowski, Hanna and de Souza, Camila P. E.},
  title   = {Bandwidth-free nonparametric density estimation for grouped data},
  journal = {arXiv preprint arXiv:2607.13182},
  year    = {2026}
}

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

@article{aguehcarlier2011barycenters,
  author  = {Agueh, Martial and Carlier, Guillaume},
  title   = {Barycenters in the {W}asserstein space},
  journal = {SIAM Journal on Mathematical Analysis},
  volume  = {43}, number = {2}, pages = {904--924}, year = {2011}
}

@article{sheppard1899application,
  author  = {Sheppard, W. F.},
  title   = {On the application of the theory of error to cases of normal distribution and normal correlation},
  journal = {Philosophical Transactions of the Royal Society of London A},
  volume  = {192}, pages = {101--167}, year = {1899}
}
```

## Usage

Include from your main tex file with a single `\input`:

```tex
\input{paper/theory/theory.tex}
```

Or, if the theoretical section will sit under its own top-level heading:

```tex
\section{Theory}
\input{paper/theory/theory.tex}
```

Cross-reference from the main body:

- `Theorem~\ref{thm:variance-ratio}` — the variance-reduction result
- `Corollary~\ref{cor:consistency-clt}` — consistency + CLT statement
- `Proposition~\ref{prop:agueh-carlier}` — the 1D Wasserstein barycenter closed form
- `Figure~\ref{fig:variance-reduction}` — populate with
  `benchmarks/plots/joint_vs_marginals/variance_reduction.png`
- `Section~\ref{sec:mixture-scm}` — populate with your mixture-SCM diagnostic
