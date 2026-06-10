# Project Summary: CFM for Decision Makers + Prior Notes

Source repo: https://github.com/ArikReuter/Graphs4CausalFoundationModels  
Config file: `experiments/complexmech/configs/complexmech_gcn_softatt.yaml`  
Entry point: `src/priordata_processing/Datasets/InterventionalDataset.py`  
SCM sampler: `src/priors/causal_prior/scm/SCMSampler.py`

---

## Output Format (one `dataset[i]` call)

Each item is a tuple of 7 tensors:

| Tensor | Shape | Description |
|---|---|---|
| `X_obs` | `(n_train, 50)` | Observational covariates, zero-padded to 50 features, feature-standardized |
| `T_obs` | `(n_train, 1)` | Treatment in observational distribution, standardized (mean≈0) |
| `Y_obs` | `(n_train, 1)` | Outcome in observational distribution, scaled to `[-1, 1]` |
| `X_intv` | `(n_train, 50)` | Covariates under interventional distribution (X is not intervened on; matches X_obs rows after resampling) |
| `T_intv` | `(n_train, 1)` | Treatment resampled from interventional distribution |
| `Y_intv` | `(n_train, 1)` | Outcome under do(T=T_intv) — the prediction target |
| `anc_matrix` | `(L+2, L+2)` | Partial ancestor matrix over `[T, Y, X_0, ..., X_{L-1}]` where L = real (non-padded) features |

`n_train` ∈ [1, 1000] sampled per-dataset.  
The ancestor matrix is `(L+2)×(L+2)`, NOT `52×52` fixed — it grows with the number of real SCM nodes kept after dropout. In the 5 samples run, all produced `52×52` because no feature dropout was used (`dropout_prob=0.0`) and the graphs had 50 real features + T + Y.

### Ancestor matrix values
- `-1.0`: i is definitely NOT an ancestor of j  
- `0.0`: relationship unknown (randomly hidden by `hide_fraction_matrix ~ Uniform(0,1)`)  
- `1.0`: i IS an ancestor of j (transitive closure of the DAG)

Node ordering in the matrix: index 0 = T, index 1 = Y, index 2..L+1 = X[:,0]..X[:,L-1].

### Interventional distribution type
`interventional_distribution_type = "resample"`: T_intv is resampled from the marginal of T (not a fixed do-value). This means the model learns `E[Y | T=t, X=x]` under resampled T, not a fixed scalar intervention. Y_intv is the outcome from the SCM under that T_intv draw.

---

## Prior: SCM Distribution

### Graph
- **Nodes**: `discrete_uniform(2, 51)` — so 2–50 nodes total
- **Edge probability**: `Beta(2, 3)` — sparse-to-medium graphs (mean ~0.4)
- **Graph type**: Erdős-Rényi DAG (topological order fixed, edges drawn with prob p)

### Mechanisms (per node)
- **Type**: MLP with probability ~0.9, XGBoost with ~0.1
- **MLP hidden layers**: 0 layers with prob 0.875, 1 with 0.1, 2 with 0.025, 3 with 0.01
- **MLP hidden dim**: mostly 1 (prob 0.7), then 2 (0.2), then 4–32 (small probs)
- **MLP activation mode**: pre/post/mixed_in each ~30%
- **MLP batch norm**: 50/50
- **Nonlinearities**: `"tabicl"` (TabICL nonlinearity set)
- **XGBoost training samples**: mostly 200–500

### Noise
- **Both exogenous and endogenous**: `MixedDistRandomStd` — mixture of Normal, Laplace, StudentT with equal weights `[0.33, 0.33, 0.34]`
- **Exogenous std**: drawn from `GammaMeanStd(mean~LogNormal(1.0,1.0), std~Uniform(0.1,0.4))` — heavy-tailed, large noise
- **Endogenous std**: drawn from `GammaMeanStd(mean~LogNormal(-3.0,0.6), std~Uniform(0.0,0.5))` — small noise (mean of lognormal at -3 → median ~0.05)
- **Endogenous p_zero = 0.0**: no noise sparsity
- **Exogenous mechanisms enabled**: root nodes also pass through a mechanism (not just raw noise)

Key implication: exogenous variables have large noise, endogenous variables have small additive noise — structure dominates endogenous signal.

### Preprocessing
- Features: standardized (zero mean, unit variance)
- Target Y: scaled to `[-1, 1]`
- Outliers removed at 99th quantile
- No Yeo-Johnson transform
- No feature dropout (`dropout_prob=0.0`)
- `remove_outliers=True, outlier_quantile=0.99`

---

## How to Replicate (minimum viable code)

```python
import sys
sys.path.insert(0, '/tmp/g4cfm/src')  # or wherever the repo lives

from priordata_processing.Datasets.InterventionalDataset import InterventionalDataset

dataset = InterventionalDataset(
    scm_config=scm_config,          # see generate_5_samples.py
    preprocessing_config=preprocessing_config,
    dataset_config=dataset_config,
    seed=42,
)

# Each call to dataset[i] samples a fresh SCM, generates observational +
# interventional data, and returns the 7-tuple.
X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv, anc_matrix = dataset[0]
```

Full config dicts are in `/tmp/g4cfm/generate_5_samples.py`.

---

## Verified Output Stats (5 samples, seed=42)

| Sample | X_obs shape | anc_matrix shape | anc -1% | anc 0% | anc 1% |
|---|---|---|---|---|---|
| 0 | (1000,50) | (52,52) | 66.8% | 13.8% | 19.4% |
| 1 | (1000,50) | (52,52) | 47.0% | 51.2% | 1.8% |
| 2 | (1000,50) | (52,52) | 69.5% | 22.1% | 8.4% |
| 3 | (1000,50) | (52,52) | 97.9% | 1.8% | 0.3% |
| 4 | (1000,50) | (52,52) | 76.5% | 12.2% | 11.3% |

High variance in ancestor density reflects the wide graph size prior (2–50 nodes) and Uniform(0,1) hide fraction.

---

## CFM for Decision Makers — Proposal Summary (CFM_for_Decision_Makers.pdf)

### Core Argument
Current CFMs (Do-PFN, Use What You Know, MACE-TNP, CausalPFN) output `q_θ(Y | D_obs, x, do(t))` — a distribution over the *outcome* under a single intervention. CATE and ATE are then scalar estimates obtained by subtracting posterior means. The proposal argues this is the wrong deliverable: decision-makers need the distribution of the *treatment effect* τ(x) = Y^do(1) − Y^do(0), not the outcome distribution. The critical problem is that computing τ from two independent marginal predictives discards the coupling between Y^do(1) and Y^do(0), producing spurious treatment-effect values.

### Why the Joint Doesn't Factorize
Within a single SCM, both potential outcomes share the same exogenous noise draw ε:
`p(Y^do(1), Y^do(0) | x, ψ) = ∫ p(Y^do(1)|x,ψ,ε) p(Y^do(0)|x,ψ,ε) p(ε) dε`
The shared ε couples them. So the posterior predictive joint ≠ product of marginals. Convolving two independent marginals fabricates (ψ_a's treated, ψ_b's control) pairings that no single SCM in the posterior produces.

### Five Proposed Components
1. **Paired potential-outcome head**: Replace two independent K-bin heads with one J×J 2D bar distribution over (Y^do(1), Y^do(0)). Trained by a proper log-density loss (not plain cross-entropy: `log(bin_width²)` correction included). J ≪ K (currently J=100 → 10,000 cells vs 1000 cells for marginal head).
2. **DensOLog refinement (NOW: inference-only, renamed MALC)**: Pass the coarse J×J table to bivariate MALC to recover a continuous log-concave joint density. Marginal densities and CATE density derived from the smoothed MALC density. **Revised role (2026-06-09):** MALC runs **only at inference**, not in the training loop. The original PDF intent was MALC-in-the-loop, but MALC's CLARABEL solver is ~13s per fit and cannot be vectorized across queries on GPU; at UWYK scale (50K steps × ~8K queries/step) this is infeasible by orders of magnitude. Training uses the histogram density `p_mat[j0,j1] / bin_width²` for the inner-inner region, which is fully differentiable on GPU and matches UWYK's 1D log-density structure. See `CFM_2D_BarDistribution_Design.md` for the full revised design.
3. **Shared-rank ATE**: Draw u~Uniform(0,1), compute τ_i^(s) = F_i^{-1}(u^(s)) for each query i, ATE^(s) = mean over i. Empirical distribution of {ATE^(s)} is the ATE posterior. Keeps "which causal world" uncertainty correlated across queries.
4. **Highest-density region**: Report shortest (1-α) region for τ distribution (union of intervals for multimodal case).
5. **LLM-guided prior reweighting**: q(ψ) ∝ score_LLM(ψ; domain) · p(ψ). Partial graph conditioning (PAM) inherited from "Use What You Know" unchanged.

### What Changes in Training Data Generation
The paired head requires generating BOTH Y^do(1) and Y^do(0) from the **same SCM with the same ε draw** per query point. Current InterventionalDataset only generates one Y_intv. The modified pipeline must:
- Fix a query covariate x
- Draw ε once from the exogenous distribution
- Propagate ε through ψ_do(1) → get Y^do(1)
- Propagate ε through ψ_do(0) → get Y^do(0)
- Record the pair (Y^do(1), Y^do(0)) as the joint training target
- Bin it into the J×J grid → that cell index is the cross-entropy target

### Key Open Issues Identified
- **Binary vs continuous treatment**: Proposal says t ∈ {0,1} in setup (Section 3) but "Use What You Know" prior uses continuous T. Contradiction needs resolution.
- **~~DensOLog bivariate version: Does not exist yet — must be built. Entire Section 4.3 depends on it.~~** *(Resolved: 2D MALC implemented in `../MALC/malc_2d.py` and `../MALC/log_concave_2d_fast.py`. Role is inference-only — see Component 2 above. Not in the training loop.)*
- **Log-concavity assumption**: MALC recovers log-concave densities. The motivating multimodal case (mass at two separated points) is NOT log-concave — this is the exact case MALC will fail to represent as a single component. MALC_2D handles this with a K-component mixture (BIC-selected), so multimodality is supported, but each individual component must still be unimodal.
- **Shared-rank coupling**: Conservative but potentially wrong. Assumes all individuals respond in same direction across causal worlds.
- **Architecture for paired head**: Unspecified. How does the transformer accept (do(1), do(0)) simultaneously as a query? Two tokens? Concatenation?
- **Joint calibration diagnostics**: No existing tools; acknowledged as open problem.
- **LLM scoring mechanism**: Completely unspecified — no description of how LLM assigns scores to individual DAGs.

### Relation to "Use What You Know" Prior
This proposal **reuses the prior unchanged** for the SCM structure, mechanisms, noise, and partial graph conditioning. The only change to data generation is the paired potential-outcome requirement. The prior repo's `SCM.py` already supports do-interventions; generating paired outcomes from the same ε draw requires a small modification to the sampling loop.
