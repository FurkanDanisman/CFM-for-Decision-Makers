# UWYK_Fig3_4 — PEHE benchmark for UWYK's Figure 3 / Figure 4 priors

Makes UWYK's Figure 3 (LinGaus) and Figure 4 (ComplexMech) settings yield PEHE,
by changing as little as possible. UWYK's own YAMLs are loaded and their
`scm_config` is passed untouched to their `SCMSampler`; their `BasicProcessing`
does every preprocessing step.

## Why Fig 3 / Fig 4 give no PEHE as shipped

Verified against UWYK @ `c27fba6`:

1. `binarize_treatment_prob` defaults to `0.0` and appears in none of the 24
   LinGaus configs nor any ComplexMechIDK config — T is continuous and do-values
   come from `ScaledUniformResamplingDist`, never do(0)/do(1).
2. The interventional pass resamples the noise, so there is no matched
   `(y0_i, y1_i)` per unit.
3. **`test_feature_mask_fraction: 1.0`** in every one of those configs, and
   `BasicProcessing._apply_test_feature_masking` zeroes *every* non-zero test
   feature column. In Fig 3 / Fig 4 the model is handed `X_intv == 0` and
   predicts `p(y | do(t), D)` — there is no covariate for an effect to be
   heterogeneous in.

## Deviations

| | change | why |
|---|---|---|
| D1 | `binarize_treatment_prob` `0.0 → 1.0` | the sanctioned change; UWYK's own switch and their own `BinarizingMechanism` |
| D2 | exogenous + endogenous noise shared across arms | without it there are no matched potential outcomes |
| D3 | X from the observational test pass, tiled across arms | descendants of T differ per arm, so there is otherwise no single `x`; non-descendants are bit-identical, so this is a no-op when `n_descendant_features == 0` |
| D4 | `test_feature_mask_fraction` default `0.0` (UWYK: `1.0`) | at 1.0, `tau(x)` is constant and PEHE collapses to an ATE error |
| D5 | outcome pinned to the regime-checked node | UWYK checks `ensure_*_path` against a uniformly drawn node, then lets `_select_target_feature` pick the real outcome *by variance* and reassigns it (`InterventionalDataset.py:905`), so the path-regime labels are not enforced upstream |

Pass `--test-feature-mask-fraction 1.0` to reproduce UWYK's setting exactly.

## Use

Runs fine on a laptop — the whole 30-cell grid takes ~9 min on CPU. torch here
is only small-tensor SCM sampling; no GPU or cluster involved. If the system
python is PEP 668-managed:

```bash
uv venv --python 3.12 .venv-uwyk
VIRTUAL_ENV=$PWD/.venv-uwyk uv pip install torch numpy pyyaml networkx xgboost scikit-learn scipy matplotlib
git clone https://github.com/ArikReuter/Graphs4CausalFoundationModels.git /tmp/uwyk && \
    git -C /tmp/uwyk checkout c27fba6      # GIT_LFS_SKIP_SMUDGE=1 to skip checkpoints
```

```bash
export UWYK_SRC=/tmp/uwyk/src   UWYK_ROOT=/tmp/uwyk        # or $DEPLOY_ROOT/external/uwyk

# invariants first (cheap): null-effect regimes must give an exactly-zero CATE
python UWYK_Fig3_4/generate_pehe_benchmark.py --prior lingaus \
    --nodes 5 --self-test

python UWYK_Fig3_4/generate_pehe_benchmark.py --prior lingaus \
    --nodes 2 5 20 35 50 --n-realizations 100
python UWYK_Fig3_4/generate_pehe_benchmark.py --prior complexmech \
    --nodes 20 --hide-fractions 0.0 0.25 0.5 0.75 1.0 --n-realizations 100
```

Output: `data/<prior>/<n>node/<regime>/hide_<h>/r<idx>.npz` + `manifest_<prior>.json`.

## Reading the numbers

`pehe_metrics.py` gives `rmse_cate` (= UWYK's `sqrt(PEHE)`), `eps_ate`, and
`oracle_pehe_floor`.

- `tau_i` carries unit-level noise `x_i` cannot determine, so a perfect model
  still scores above `oracle_pehe_floor` (the sd of the true ITE). The floor is
  the same labels for every method, so rankings stay fair.
- `path_YT` and `path_independent_TY` have no directed T → Y path, so the true
  CATE is identically 0 and `eps_ate` is undefined (returns NaN). There,
  `rmse_cate` is a direct false-positive-effect measure.
- `n_descendant_features == 0` marks realizations where D3 is a no-op — filter
  to those for a subset that is exactly faithful to UWYK modulo binary T.

## CATE / ATE distributions

`cate_ate_distributions.py` reports, per cell, the pooled true-CATE distribution,
the per-dataset ATE distribution, and the **oracle sqrt(PEHE) floor** (mean
within-dataset sd of the true ITE).

```bash
python UWYK_Fig3_4/cate_ate_distributions.py \
    --prior lingaus complexmech --nodes 2 5 20 35 50 --n-realizations 100
```

Reads saved realizations from `--data-dir` when present, otherwise samples them
in memory. Reading saved data needs only numpy/matplotlib — no torch — so the
analysis runs on a laptop once the npz files exist. Writes
`distributions/cate_ate_stats.json`, `cate_ate_summary.md`, and per-prior
`cate_<prior>.png` / `ate_<prior>.png`.

`hide_fraction` is not an axis. The ComplexMechIDK YAMLs are byte-identical
across `hide_0.0 ... hide_1.0` apart from `hide_fraction_matrix`, which only
masks the ancestor matrix given to the model and never touches the SCM — so the
CATE and ATE distributions are identical for every hide fraction.

Read `heterogeneity_ratio` first: it is the oracle floor over mean |ATE|. Near 0
means the prior generates an effectively constant treatment effect, so PEHE on
that cell measures only ATE accuracy. **Expect this for LinGaus (Fig 3)**: with
`mlp_nonlins: id` and `mlp_num_hidden_layers: 0`, every mechanism is affine, so
with the noise held fixed across arms `tau_i` reduces to the product of path
coefficients times `(t1 - t0)` — constant across units. Fig 4's `tabicl`
nonlinearities and XGBoost mechanisms are what produce genuine heterogeneity.
The script measures this rather than assuming it.

## Results (100 realizations per cell)

Generated numbers are in `distributions/`. Two headlines:

**Fig 3 (LinGaus) has no treatment-effect heterogeneity.** Within-dataset sd of
the true ITE is 4e-8 to 6e-8 — machine zero — at every node count. Every
mechanism is affine (`mlp_nonlins: id`, 0 hidden layers), so with the noise held
fixed across arms `tau` collapses to a constant per dataset. **PEHE on Fig-3 data
is an ATE error wearing a PEHE label.** Use Fig 4 if you want to measure
heterogeneity.

**Fig 4 (ComplexMech) has real heterogeneity, plus a large exact-zero spike.**
Within-dataset ITE sd is 0.07–0.16, but 12–27% of units have an *exactly* zero
effect (the median CATE is 0 for n>=5): tree and saturating mechanisms often do
not respond at all to flipping T. From n=20 up, the oracle floor exceeds mean
|ATE| (`heterogeneity_ratio` 1.4–3.4), so within-dataset spread dominates the
average effect.

Effect sizes shrink as the graph grows in both priors — ATE sd 0.39 -> 0.13
(LinGaus n=2 -> 50), 0.68 -> 0.15 (ComplexMech).

The null regimes come out exactly 0 as required; three cells carry float residue
at 1e-10..1e-8, which is numerically zero on a target scaled to [-1, 1].

## Two traps, both fixed in-code

1. **Segfault.** torch and XGBoost each load an OpenMP runtime; together they
   corrupt the heap on macOS (`Fatal Python error:` with no traceback). Both
   modules set `OMP_NUM_THREADS=1` before importing torch. Override by exporting
   it yourself.
2. **Non-reproducibility.** `SCMSampler.sample(seed=)` seeds only the graph and
   hyperparameter draws — mechanism weight init and every noise draw read the
   *global* torch RNG, and the first XGBoost-involving realization in a process
   never matches later ones. Unfixed, identical runs differed by max|d tau| =
   2.1, the size of the signal. `generate_realization` now reseeds the global RNG
   per attempt and burns one discarded warm-up realization. `--self-test` asserts
   bitwise regeneration.

## Which cells are actually usable for PEHE

`learnable_heterogeneity.py` answers the question the distribution table cannot:
how much of the within-dataset CATE variance is *predictable from X*. Within each
dataset it splits the test units, fits a random forest `tau ~ X` on one half and
scores it on the other. `learnable = 1 - (pehe_oracle/pehe_constant)^2`. The
forest is fitted on the true `tau`, so this is an optimistic ceiling, not a
competitor.

| prior | nodes | features | learnable (median) | frac of datasets > 0.1 |
|---|---|---|---|---|
| lingaus | 2–50 | 0–48 | **0.000** | 0.00 |
| complexmech | 2 | 0 | 0.000 | 0.00 |
| complexmech | 5 | 3 | **0.781** | 0.70 |
| complexmech | 20 | 18 | **0.404** | 0.80 |
| complexmech | 35 | 33 | 0.260 | 0.60 |
| complexmech | 50 | 47 | 0.143 | 0.57 |

**LinGaus is zero at every node count, in every single dataset.** Its treatment
effect is constant within a dataset, so no method can beat a constant predictor
and PEHE is exactly an ATE error. Fig-3 cells cannot separate a CATE method from
an ATE estimator.

**ComplexMech n=2 is also zero** — a 2-node SCM is just T and Y, so there are no
covariates to condition on at all.

**ComplexMech n=5–20 is the usable range**, and n=5–20 is where heterogeneity is
both large and present in most datasets. Learnability falls off as the graph
grows (0.78 -> 0.14 from n=5 to n=50): more nodes dilute each path's contribution.

Judge these on medians. The effect distribution is heavy-tailed with a spike at
exactly zero, so means get dragged — at n=20 the mean R^2 is -0.23 while the
median is +0.40.
