# benchmarks/plots — layout

Scripts live at this top level; each writes into the subfolder that matches
its topic, so the flat root is only ever `.py` files.

```
plots/
├── plot_joint_marginals.py       → context_sweep/    (SCM-prior sweep)
├── plot_mixture_scm.py           → mixture_scm/      (hand-crafted mixture)
├── plot_mask_example.py          → malc_examples/    (MALC diagonal masking)
├── plot_raw_malc_example.py      → malc_examples/    (raw vs MALC-smoothed)
├── plot_variance_reduction.py    → joint_vs_marginals/  (Monte-Carlo: joint variance vs ρ)
├── plot_te_demonstration.py      → TE_demonstration/    (same marginals, 3 joints, 3 τ shapes)
├── find_multimodal_queries.py    (utility: prints QUERY_IDXS= lines for plot_joint_marginals)
│
├── context_sweep/                per-query TE / joint / marginals across N,
│                                 sampled from the default UWYK SCM prior
├── mixture_scm/                  diagnostic 3-cluster mixture where q5 (centroid)
│                                 has p(Z|X)=(1/3,1/3,1/3) → tri-modal marginals
├── joint_vs_marginals/           didactic figure for the paper's core structural
│   └── variance_reduction.png    argument: at ρ=0.9 the joint estimator has
│                                 ~20× lower variance on τ̂ than the two-marginal
│                                 estimator, purely from exploiting the within-
│                                 unit correlation between potential outcomes
├── TE_demonstration/             three-panel demo — Y_do0 ~ 50/50 at ±5,
│   └── same_marginals_three_joints.png   Y_do1 ~ 50/50 at ±8. The same marginals
│                                 admit independent / comonotonic / countermonotonic
│                                 joints, giving p(τ) supports {±3,±13} / {±3} /
│                                 {±13}. All three have E[τ] = 0 — a marginal-only
│                                 model cannot distinguish them.
├── malc_examples/                MASK_example.png, RAW_MALC_example.png
└── multimodal_search/            iterations of the seed-7 multimodal search
                                  (scm7_mm, scm7_mm2, scm7_dtau — keep for now)
```

## How to reproduce

```bash
source .venv/bin/activate

# Joint-vs-marginals explainers (no external deps, ~5 s each)
python benchmarks/plots/plot_variance_reduction.py
python benchmarks/plots/plot_te_demonstration.py

# Hand-crafted mixture SCM (needs the trained checkpoint + UWYK src for imports)
python benchmarks/plots/plot_mixture_scm.py

# SCM-prior sweep — pick queries first, then plot
UWYK_SRC=/path/to/g4cfm/src PYTHONHASHSEED=0 \
    SCM_SEEDS=0-29 python benchmarks/plots/find_multimodal_queries.py
UWYK_SRC=/path/to/g4cfm/src PYTHONHASHSEED=0 \
    SCM_SEED=7 QUERY_IDXS=241,130,129,308,333,113 OUT_PREFIX=scm7 \
    python benchmarks/plots/plot_joint_marginals.py
```

## Notes

- `joint_vs_marginals/variance_reduction.png` is pure numpy/matplotlib — no
  model or checkpoint involved. It illustrates the structural argument for
  the paper: joint modelling of (Y_do0, Y_do1) lowers the variance of τ̂ vs
  two-marginal modelling, and this holds under any convex loss (including
  the squared error that PEHE penalises). Mode-vs-mean would only reappear
  as an argument if we could evidence systematic model miscalibration.
- `mixture_scm/` figures do use the checkpoint. The 3-cluster arrangement
  puts cluster centres on an equilateral triangle in (X₁, X₂), so the
  centroid query has p(Z|X) = (1/3, 1/3, 1/3) — proving the model can
  represent tri-modal marginal outcomes when the SCM demands it.
- `multimodal_search/` holds three iterations of the same seed-7 search
  (raw, higher smoothing, distinct-τ scoring). The `scm7_dtau_*` files are
  the final ranking; older iterations are kept for reproducibility.
