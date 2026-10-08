# Do-PFN (1D) on the Do-PFN case studies

This folder is self-contained. It does two things:

1. Generates the six original Do-PFN synthetic case studies, with the functions and noise you choose.
2. Runs the released Do-PFN 1D model on them and scores the CATE.

```
generation.py      case-study generator (numpy only)
run_dopfn.py       Do-PFN inference + PEHE / L1-ATE
requirements.txt
dopfn/             released Do-PFN code + weights (github.com/jr2021/Do-PFN @ 90d6743), trimmed to inference
  artifacts/       dopfn_config.pkl, dopfn_model.pkl, model_submitit_..._epoch_-1.cpkt (weights; all three needed)
  model/, scripts/, utils.py
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python generation.py --out-dir data --context-sizes 200 500 1000 --n-realizations 100
python run_dopfn.py --data-root data --cases Observed_Confounder --n 1000
```

`run_dopfn.py` runs on CPU or GPU. On CPU, N=200 takes about one second per realization.

Two changes were made to the upstream Do-PFN code, so that it runs on current torch and sklearn:

- **`dopfn/model/layer.py`:** `Optional` is now imported from `typing`. The old import from `torch.nn.modules.transformer` fails on newer torch.
- **`dopfn/scripts/transformer_prediction_interface/base.py`:** `check_array` gets `ensure_all_finite` or `force_all_finite`, whichever the installed sklearn accepts.

Nothing else in the model was changed.

---

## 1. Case studies: `generation.py`

### The six DAGs

| Case | Graph | Observed in X |
|---|---|---|
| `Observed_Confounder` | C→T, C→Y, T→Y | C |
| `Backdoor_Criterion` | C→T, C→Y, T→M→Y | C |
| `Observed_Mediator` | T→M→Y, T→Y (T ~ Bernoulli(½)) | M |
| `Observed_Mediator_and_Confounder` | C→T, C→Y, T→Y, T→M→Y | C, M |
| `Unobserved_Confounder` | U→T, U→Y, C→T, C→Y, T→Y | C (U hidden) |
| `Frontdoor_Criterion` | C→T, C→Y, T→M→Y | C |

Backdoor and Frontdoor have the same structure, as in the released Do-PFN data. Each DAG is a list of `Node`s returned by a `_<case>()` function. The `Node` kinds are:

- `root_normal`: z ~ N(0, exo_std).
- `root_bernoulli`: T ~ Bernoulli(½).
- `structural`: activation(w · parents) + ε.

### One realization (one SCM + one dataset)

`_SampledSCM` holds one sampled SCM. The following are drawn once per realization:

- `exo_std ~ U(1, 3)`: the std of the root nodes.
- `noise_std = noise_scale · 0.3 · Beta(1, 5)`: the std of the additive ε on every structural node.
- Weights `w ~ U(−1/√p, 1/√p)`, where p is the number of parents.
- One activation per structural node, picked uniformly from `ACTIVATIONS` = {square, relu, tanh, identity}.
- A structural T is binarised: T = 1 if its continuous value is above the sample median.
- All noise is drawn once per unit and reused. The observational pass and the do(T=0) and do(T=1) passes therefore see the same units.

### Output

Files are written to `<out>/<Case>/N{N}/{Case}_{r}.npz`, plus a `manifest.json`.

| key | shape | meaning |
|---|---|---|
| `X` | (N, d) | observed covariates (T excluded) |
| `T` | (N,) | binary treatment, 0/1 |
| `Y` | (N,) | observed outcome |
| `mu_0`, `mu_1` | (N,) | Y under do(T=0) and do(T=1), with Y's own noise ε_Y removed |
| `cate` | (N,) | `mu_1 − mu_0`, the ground-truth CATE per unit |
| `feature_names`, `graph_edges`, `exo_std`, `noise_std`, `seed`, … | | metadata |

Only ε_Y is removed when computing `mu_t`, so in five of the cases `cate` is the true CATE τ(x). The exception is `Unobserved_Confounder`, where `cate` still depends on the hidden U. Since U is not in X, there `cate` is a per-unit effect, not τ(x).

If an activation blows up, `generate_realization` raises `ValueError`, and `generate_sweep` resamples with a new seed. The resamples are counted in `manifest.json`.

### Original settings

These are the defaults:

```bash
python generation.py --out-dir data --context-sizes 200 500 1000 --n-realizations 100 --seed-base 0
# optional: --cases Observed_Confounder Unobserved_Confounder ...
```

Leave `--cate-shift 0` and `--noise-scale 1.0` at their defaults, because they give the original structure. `--cate-shift` adds a constant effect β·T to Y and is not part of the original case studies.

### Changing functions and noise

- **Functions:** edit the `ACTIVATIONS` dict at the top of `generation.py`. Every structural node draws uniformly from it, so to fix one function for all nodes, leave a single entry.
- **Noise level:** `--noise-scale s` multiplies `noise_std`, so the noise on T, M and Y is scaled together.
- **Noise or root distribution:** edit `_SampledSCM.__init__`.
  - The ε draw is `self._noise[...] = rng.normal(0, self.noise_std, N)`.
  - The roots are drawn in `self._root[...]`.
  - Keep each draw a length-N array that is drawn once. The interventional passes reuse it, and that reuse is what makes `cate` correct.
- **Graph:** add or edit a `_<case>()` function, then register it in `_BUILDERS` and `CASE_STUDIES`.

`cate` is computed from the SCM itself, so it stays correct after these changes, with one exception. If ε_Y is no longer additive (for example, multiplicative), then dropping ε_Y in `forward(y_noiseless=True)` no longer gives E[Y | do(t), x]. In that case, compute `mu_t` by averaging Y over many fresh ε_Y draws instead.

---

## 2. Do-PFN 1D: `run_dopfn.py`

```bash
python run_dopfn.py --data-root data --cases Observed_Confounder Observed_Mediator --n 1000 [--max-real 10]
```

The script prints PEHE and L1-ATE for each realization, and the mean over realizations for each case.

### How it calls the model

`DoPFNRegressor` loads `artifacts/` by relative path, so the script does `chdir` into `dopfn/` before building it. Build the regressor once and reuse it. The treatment always goes in column 0 of X.

```python
x_ctx = np.concatenate([T[:, None], X], 1)          # context: [T | X]
reg.fit(torch.tensor(x_ctx), torch.tensor(Y))
X0 = [0 | X];  X1 = [1 | X]                         # same query units, T forced to 0 / 1
f0 = reg.predict_full(torch.tensor(X0))             # p(y | do(T=0), x)
f1 = reg.predict_full(torch.tensor(X1))             # p(y | do(T=1), x)
cate_hat = f1['mean'] - f0['mean']
```

- `reg.predict_cate(torch.tensor(x_q))` gives the same point CATE in one call. Keep the two `predict_full` calls if you also want the distributions: each call has ensemble randomness, so separate calls would not match each other exactly.
- Each `predict_full` output is a dict with:
  - `'logits'`: shape (N_q, number of bins);
  - `'buckets'`: the softmax pmf;
  - `'criterion'`: a bar distribution whose `.borders` are in raw Y units;
  - `'mean'`, `'median'`, and `'quantile_0.10'` … `'quantile_0.90'`.
- Do-PFN is a 1D model: it predicts each arm separately. A CATE distribution needs a coupling assumption, and our usual choice is to treat the two arms as independent and convolve them.

### Evaluation conventions

- **Context:** all N rows of a realization.
- **Queries:** the covariates of the same N units (in-sample CATE).
- **Truth:** `cate` from the npz.
- **PEHE:** √mean((τ̂ − τ)²).
- **L1-ATE:** |mean τ̂ − mean τ|.
- Report each metric as the mean over the 100 realizations, for each case and each N.
