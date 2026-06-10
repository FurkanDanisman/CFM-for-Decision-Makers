# CFM for Decision Makers

A 2D paired potential outcome head for Causal Foundation Models (CFMs).
Extends the Prior-Fitted Network architecture from
[*Use What You Know* (UWYK)](https://github.com/ArikReuter/Graphs4CausalFoundationModels)
to predict the **joint** distribution of `(Y_do(0), Y_do(1))` for a query `x`
given observational context `D = {(X_obs, T_obs, Y_obs)}` — without
fabricating the spurious `(ψ_a's treated, ψ_b's control)` pairings that
result from convolving two independent marginal predictives.

> **TL;DR.** Y_do(0) and Y_do(1) share exogenous noise → they are correlated
> potential outcomes. Modeling them with independent marginals destroys that
> coupling. This repo provides a single 2D-bar-distribution head that
> captures the joint, a coarse `100×100` grid during training, and a smooth
> log-concave density (MALC) at inference.

---

## Documentation

| File | Contents |
|---|---|
| `README.md` (this file) | Project goals, design overview, how to run |
| [`docs/CFM_2D_BarDistribution_Design.md`](docs/CFM_2D_BarDistribution_Design.md) | Full design of the 2D BarDistribution head and divergence from the original PDF |
| [`docs/prior_summary.md`](docs/prior_summary.md) | SCM prior + proposal annotations |
| [`docs/PROBLEM_TRAINING.md`](docs/PROBLEM_TRAINING.md) | **Read this before touching the training pipeline.** Active bug log + forbidden patterns. |
| [`docs/Luke.md`](docs/Luke.md) | Hand-off guide for reproducing the training (or the bug) on a new cluster |
| `CFM_for_Decision_Makers.pdf` | Original proposal |

## Table of contents

- [Problem](#problem)
- [Design at a glance](#design-at-a-glance)
- [The 9-region structure](#the-9-region-structure)
- [Why MALC is inference-only](#why-malc-is-inference-only)
- [Pipeline](#pipeline)
- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Running](#running)
- [Status & limitations](#status--limitations)
- [References](#references)

---

## Problem

For a decision-maker, the right deliverable from a causal model is not the
posterior mean of an outcome but the **distribution of the treatment effect**
`τ(x) = Y_do(1) − Y_do(0)`. Current CFMs (Do-PFN, UWYK, MACE-TNP, CausalPFN)
output a marginal predictive `q_θ(Y | D_obs, x, do(t))` per intervention.
CATE is then obtained by subtracting two posterior means.

This discards the coupling between `Y_do(1)` and `Y_do(0)`. Within a single
SCM ψ they share the same exogenous noise draw `ε`:

```
p(Y_do(1), Y_do(0) | x, ψ) = ∫ p(Y_do(1)|x,ψ,ε) p(Y_do(0)|x,ψ,ε) p(ε) dε
```

So the posterior predictive joint is **not** the product of marginals.
Convolving the two marginals fabricates pairs that no individual SCM in the
posterior actually produces.

We replace the two independent 1D heads with one **2D paired head**.

---

## Design at a glance

| Component | Choice |
|---|---|
| Grid resolution | K = 100 per axis → K² = 10,000 inner cells, `bin_width = 0.02` |
| Output dim per query | K² + 9 + 4 = **10,013** logits |
| Tail support | Half-Gaussian per axis, four learnable scales `σ_L0, σ_R0, σ_L1, σ_R1` |
| Correlation | ρ **derived** from `p_mat` (Pearson on the discrete joint), not learned |
| Mixture weights | 9 region weights (inner, 4 mixed, 4 corner), softmax to sum to 1 |
| Training inner-inner density | Histogram: `p_mat[j0, j1] / bin_width²` — differentiable on GPU |
| Inference inner-inner density | MALC: smooth 2D log-concave density via Cule–Samworth–Stewart MLE |
| Backbone | Transformer PFN (context self-attention + cross-attention from query) |

The 2D extension preserves UWYK's per-axis structure: `K` bars plus left/right
half-Gaussian tails, just lifted to a 2D grid plus 4 corner regions.

---

## The 9-region structure

The 2D space `ℝ²` is split based on where `(Y_do(0), Y_do(1))` falls relative
to the inner box `[-1, 1]²`:

```
                   Y_do(1) < -1     Y_do(1) ∈ [-1,1]     Y_do(1) > 1

Y_do(0) > 1        R0-L1             R0-inner             R0-R1
Y_do(0) ∈ [-1,1]   inner-L1          inner-inner          inner-R1
Y_do(0) < -1       L0-L1             L0-inner             L0-R1
```

Densities per region:

| Region | Density |
|---|---|
| inner-inner       | training: `p_mat[j0,j1] / bw²`     ·     inference: `dmalc_2d(fit, y0, y1)` |
| L0-/R0-inner      | `2 · N(y0; ±1, σ) × p_mat[boundary_row, j1] / Σ p_mat[boundary_row, :] / bw` |
| inner-L1/R1       | mirror of the above on the y1 axis |
| Same-dir corners (L0L1, R0R1) | `Φ_ρ(y0,y1) / (¼ + arcsin(ρ)/2π)` — Sheppard normalization |
| Opp-dir corners (L0R1, R0L1)  | `Φ_ρ(y0,y1) / (¼ − arcsin(ρ)/2π)` |

`Φ_ρ` is the bivariate Gaussian centered at the appropriate corner with
scales `(σ_L0, σ_L1)` / `(σ_R0, σ_R1)` and correlation `ρ` derived from
`p_mat`. ρ is shared across all 4 corner regions for self-consistency.

Per-point loss = `−log(w_region · density)`. Batch loss = mean.

Everything is fully differentiable on GPU (softmax + gather + elementary
ops). No solver runs in the forward pass.

---

## Why MALC is inference-only

The original intent (`CFM_for_Decision_Makers.pdf`, Section 4.3) was to run
MALC inside the training loop so the inner-inner density at the exact
continuous target `(y0, y1)` would be the smooth log-concave value, not the
piecewise-constant histogram value.

At UWYK training scale this is infeasible:

```
50,000 steps × ~8,000 queries/step ≈ 4 × 10⁸ MALC fits
```

MALC fits use the CLARABEL convex solver internally (~13 s per fit, not
GPU-vectorizable, not differentiable through autograd). Even with aggressive
CPU parallelism and stripped-down MALC variants we estimated ~4 days to
several years of compute for a single training run. The histogram baseline
takes hours.

**Resolution.** Training uses the histogram density (same structure as
UWYK's 1D BarDistribution, just lifted to 2D). MALC runs only at inference,
once per (query, context), to convert the coarse `100 × 100` grid into a
continuous log-concave density. This is the same role MALC plays in the 1D
UWYK head (logconDens inside `[-1, 1]`), just generalized to 2D.

See [`docs/CFM_2D_BarDistribution_Design.md`](docs/CFM_2D_BarDistribution_Design.md)
for the full rationale and the divergence from the PDF.

---

## Pipeline

### 1. Data generation — `generate_paired_samples.py`

Sample an SCM ψ, choose binary treatment `T ∈ {0,1}` and outcome `Y`, then
generate **paired** potential outcomes from the same exogenous noise draw
ε. For each query point:

- `X_intv` (covariates, standardized)
- `Y_do0 = Y | do(T=0), ε` — scaled to `[-1, 1]` using observational
  train min/max but **not** clamped (tails handle out-of-range)
- `Y_do1 = Y | do(T=1), ε`

Output `.pt` files in `outputs_paired/`.

### 2. Model — `models/InterventionalPFN.py`

A Transformer PFN:

```
Context tokens: linear((X_obs, T_obs, Y_obs))         → (B, N, d_model)
Query tokens:   linear((X_intv, null_T, null_Y))      → (B, M, d_model)
self_attn × depth over context
cross_attn × depth from query to context
output_head: linear(d_model → 10,013)                 → (B, M, 10,013)
```

`null_T` and `null_Y` are learned parameters that fill the slots
of treatment and outcome in the query token (the query has no T, no Y).

### 3. Loss — `losses/BarDistribution2D.py`

Unpacks the 10,013 outputs, computes ρ from p_mat, routes each target
through the 9-region structure, returns the mean `−log_prob`.

### 4. Training — `train_cfm_small.py`

Demo scale: 200 steps, B=1, M=30 (for sanity-checking on a laptop).
Production scale (UWYK config): 50K steps, B=8, M ~ U[1,500].

### 5. Inference — `fit_malc_inner` + `eval_density_2d`

After training, hand `p_mat` to MALC to obtain a smooth log-concave density;
evaluate at any continuous `(y0, y1)` through the same 9-region routing.

---

## Repository layout

```
.
├── README.md
├── CFM_for_Decision_Makers.pdf        # original proposal
│
├── docs/                              # all design / problem / handoff documents
│   ├── CFM_2D_BarDistribution_Design.md  # full design, including divergence from PDF
│   ├── prior_summary.md                  # SCM prior + proposal annotations
│   ├── PROBLEM_TRAINING.md               # active training-pipeline bug log + forbidden patterns
│   └── Luke.md                           # hand-off guide for someone reproducing on a new cluster
│
├── generate_paired_samples.py         # data generation (paired Y_do0, Y_do1)
├── generate_5_samples.py              # earlier single-outcome generator
├── train_cfm.py                       # production training entry point (UWYK-scale)
├── train_cfm_small.py                 # smoke / demo training entry point
├── smoke.py                           # tiny CPU/GPU import + forward+backward check
│
├── submit_smoke.sh                    # SLURM: 200-step demo on 1 H100 (~5 min)
├── submit_scaledup.sh                 # SLURM: 500-step UWYK-config smoke (~30 min)
├── submit_train.sh                    # SLURM: full 50K-step training, 3 × 20h chained chunks
│
├── models/
│   └── InterventionalPFN.py           # Transformer PFN (UWYK body verbatim)
├── losses/
│   └── BarDistribution2D.py           # 9-region 2D loss + MALC inference helpers
├── data/
│   └── PairedInterventionalDataset.py # streaming SCM dataset (paired potential outcomes)
│
├── MALC/                              # MALC 2D log-concave density estimator
│   ├── SUMMARY.md
│   ├── malc_2d.py
│   ├── log_concave_2d_fast.py
│   ├── example.py
│   └── example_*.png                  # illustrative density plots
│
├── outputs_paired/                    # generated paired-outcome samples (used by train_cfm_small.py)
│   ├── sample_0.pt … sample_4.pt
│   └── sample_0/ … sample_4/          # CSV exports
├── outputs/                           # earlier (non-paired) generator output, kept for reference
│
└── debugging/                         # historical debug notes
    ├── DEBUGGING_SUMMARY.md
    └── debug_cate{,2,3}.py
```

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy scipy clarabel matplotlib
```

Python 3.10+. Tested on macOS (MPS backend) with `torch >= 2.0`.

`generate_paired_samples.py` additionally depends on the UWYK source
(`/tmp/g4cfm/src` in our setup, or the official repo
[ArikReuter/Graphs4CausalFoundationModels](https://github.com/ArikReuter/Graphs4CausalFoundationModels)).
The included `outputs_paired/` files are pre-generated so training can
run without it.

---

## Running

### Quick training sanity check

```bash
python3 train_cfm_small.py
```

This trains for 200 steps on samples 0–3 and evaluates on sample 4.
Expected initial loss ≈ `log 9 + 2·log 2 ≈ 3.58` (uniform-softmax baseline);
loss drops below the baseline as the model learns to concentrate mass at
the paired outcome locations.

### Density at a continuous point (inference path)

```python
from losses.BarDistribution2D import (
    total_params, make_edges, fit_malc_inner, eval_density_2d
)

# 1. Run the trained model, get logits for one query
logits = model(X_obs, T_obs, Y_obs, X_intv)['predictions'][0, query_idx]  # (10013,)

# 2. Unpack
import torch.nn.functional as F
import torch, math
J = 100
p_mat   = F.softmax(logits[:J*J], dim=-1).reshape(J, J).cpu().numpy()
w_reg   = F.softmax(logits[J*J : J*J + 9], dim=-1).cpu().numpy()
sL0, sR0, sL1, sR1 = [
    0.02 * (F.softplus(logits[J*J + 9 + i]).item() + 1e-3) for i in range(4)
]

# 3. Smooth p_mat with MALC (runs CLARABEL once, ~13s)
import numpy as np
edges  = np.linspace(-1, 1, J + 1)
fit    = fit_malc_inner(p_mat, edges, edges)

# 4. Evaluate density at arbitrary continuous points
import numpy as np
y0_pts = np.array([0.2, -0.5, 1.3])
y1_pts = np.array([0.1,  0.0, -0.7])
rho    = ...  # _compute_rho on p_mat (see losses/BarDistribution2D.py)
dens   = eval_density_2d(fit, y0_pts, y1_pts, edges, sL0, sR0, sL1, sR1, rho, w_reg)
```

---

## Status & limitations

This is an **active research prototype**, not a polished library.

- **Trained model**: not included. The repo provides the architecture, loss,
  and a small-scale training script. Production-scale training (UWYK-scale)
  is not run here.
- **Data**: 5 pre-generated SCM tasks are included. Larger prior coverage
  requires the UWYK source.
- **MALC at inference is slow** (~13s per fit). Acceptable for downstream
  analysis but not for tight inference loops. Faster MALC variants are
  possible (smaller B, fix K=1 mixture component, looser tolerance).
- **Coarse grid at training**: K=100 → bin_width = 0.02. At training time
  the inner density is piecewise constant inside each bin. MALC fixes this
  at inference but not during gradient computation.
- **Boundary conditional approximation** in mixed regions
  (`p(y_in | y_out ≈ boundary)`): uses the discrete boundary row/column of
  p_mat. Self-correcting because the half-Gaussian assigns negligible
  weight to extreme y_out.
- **Single SCM per task**: paired potential outcomes are coupled within an
  SCM but the model is not constrained to be coherent across SCMs. The
  R-PFN minimax extension (see `docs/prior_summary.md` and project memory)
  addresses this.

---

## References

- *Use What You Know: Causal Foundation Models with Partial Graphs*
  ([arXiv:2602.14972](https://arxiv.org/abs/2602.14972)) —
  [code](https://github.com/ArikReuter/Graphs4CausalFoundationModels).
  Source of the 1D BarDistribution we extend.
- *Do-PFN: Pre-Trained Transformers for Causal Inference* —
  [code](https://github.com/jr2021/Do-PFN). Closely related PFN baseline.
- *Maximum Likelihood Estimation of a Multivariate Log-Concave Density*
  (Cule, Samworth, Stewart 2010). The 2D MLE MALC implements.
- `CFM_for_Decision_Makers.pdf` (in this repo). Original proposal; this
  implementation departs from it on the MALC-in-training decision.

---

## License

Code: choose a license before publishing further (suggest MIT or Apache-2.0).
MALC code in `MALC/` is the author's own working implementation.
