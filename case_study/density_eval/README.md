# `case_study/density_eval`

Working copies of the Tier-C density eval, relocated here so the case-study
version can diverge without touching the originals.

## Frozen originals — never edited from this folder

| Original | Status |
|---|---|
| `benchmarks/eval_graph2d/density_common.py` | **read-only**, copied verbatim |
| `benchmarks/eval_graph2d/eval_density_tauC.py` | **read-only**, copied + path plumbing only |

Everything under `benchmarks/eval_graph2d/` stays as it is. Edits go here.

## What was copied

- `density_common.py` — **byte-identical** to the original. Its lazy MALC import
  resolves `../../MALC`, and since this folder sits at the same depth
  (`case_study/density_eval/` vs `benchmarks/eval_graph2d/`), that still lands on
  the repo root. No change was needed.
- `eval_density_tauC.py` — identical except for import plumbing (3 hunks):
  - added `_ORIG = <repo>/benchmarks/eval_graph2d`, with an explicit check
  - the harness `eval_graph2d_realcause.py` is loaded from `_ORIG`, not `_HERE`
  - `sys.path` gets `_ORIG` then `_HERE`, so **`_HERE` wins**: `density_common`
    resolves to the local copy while `density_truth` and the rest fall through
    to the originals

Only two files were copied. The runner's other siblings — `eval_graph2d_realcause.py`
(the harness) and `density_truth.py` — are still imported from the original
directory. If either needs to diverge too, copy it here and it will shadow the
original automatically, since `_HERE` precedes `_ORIG` on `sys.path`.

`_REPO` still resolves to the repo root, so `CKPT`, `DOPFN_JOINT_CKPT` and the
ACIC cache defaults are unchanged.

## Interval + distributional metrics — `interval_metrics.py`

Raw path only (no MALC). Consumes a density on `TAU_CENTERS` and returns:

| metric | what it answers | gameable alone? |
|---|---|---|
| `coverage_pct` | is the stated level honest | yes — widen |
| `length_mean` | is it sharp | yes — narrow |
| `winkler_mean` | both at once, `length + (2/α)·miss` | **no**, proper |
| `crps_mean` | whole-distribution, units of τ | **no**, proper |

```python
from interval_metrics import query_metrics, summarize, format_table
per_q = [query_metrics(p_tau, TAU_CENTERS, tau_true, y_scale=y_scale)
         for p_tau, tau_true in zip(densities, truths)]
print(format_table(summarize(per_q), title='graph2d v3a'))
```

`levels` are **alphas** (0.05 ⇔ a 95% interval); default `(0.50, 0.20, 0.10, 0.05)`.
`method='equal-tailed'` (default) or `'hpd'`.

Notes that matter for correctness:

- **CRPS splits the integral at τ\*.** The integrand jumps there, so a plain
  trapezoid over the whole grid is only O(dz); at dz=5e-4 that is a ~2e-4 bias,
  the same size as the model differences being resolved. Inserting τ\* as an
  exact node makes it O(dz²).
- **Censoring is tested on unnormalised mass.** `TAU_CENTERS` spans only
  [-3, 3]; truncation biases length, Winkler and CRPS *downward*, flattering a
  diffuse model. `censored_frac` and `mass_*` are reported alongside, never
  folded in — gate on them.
- **HPD returns `measure`, not just the hull.** For a multimodal density the
  region is disjoint and its hull can exceed the equal-tailed length while the
  region is shorter. Coverage uses exact region membership (`p(τ\*) ≥ level`).

Validated against closed forms in `test_interval_metrics.py` (19 tests):
Gaussian CRPS, Gaussian quantiles, the 0.2337·σ centred-CRPS constant, the
CRPS → |τ̂−τ\*| limit at rate σ/√π, Winkler properness under both over- and
under-widening, and end-to-end coverage on calibrated / overconfident /
diffuse forecasts.

## CausalPFN densities — `density_cpfn.py`

`density_common.py` has no CausalPFN class: `UWYK1D` / `DoPFN1D` / `Joint2D` all
unpack **raw head logits** (bars + half-Gaussian tail masses and scales), and the
CausalPFN evals never expose logits. What they do expose is `DENSITY_DUMP=1`:

| dump | keys |
|---|---|
| 1D `eval_causalpfn_v0_realcause.py` | `edges` (J+1), `p_y0_scaled` (N_q,J), `p_y1_scaled`, `y_shift`, `y_scale` |
| 2D `eval_cpfn2d_realcause.py` | + `p_joint_scaled` (N_q,J,J), `true_cate_per_query` |

Requires `STD_MODE=pooled` (1D) / not `per_arm` (2D) — both scripts already
assert this, because per-arm shifts do not cancel in τ.

```python
from density_cpfn import load_dump, iter_queries
dump = load_dump('.../cpfn2d_dump.npz')
for q, tau_density, extras in iter_queries(dump):
    p = tau_density(TAU_CENTERS)              # scaled axis
    m = query_metrics(p, TAU_CENTERS, true_cate[q] / dump['y_scale'],
                      y_scale=dump['y_scale'])
```

**A CausalPFN prediction is a pure histogram** — the dump carries no tail
parameters and none are recoverable. Two consequences:

- **p(τ) is exact, no quadrature.** `joint_tau_density` splits p(τ) into an
  exact interior diagonal plus quadrature over 8 outside regions; here those
  regions are empty, so only `_interior_tau` survives. `to_joint2d()` /
  `to_uwyk1d()` adapters are provided and the tests assert they give the
  identical answer through the generic path (max abs diff < 1e-9).
- **Support is compact:** p(τ)=0 outside ±(edges[-1]−edges[0]). A truth outside
  is a guaranteed miss at every level — `support_diagnostics` reports the
  fraction, and it must be read next to coverage.

**τ support can exceed `TAU_CENTERS`.** For edges spanning [−2,2] the τ support
is ±4, but `TAU_CENTERS` spans only [−3,3]. Truncation biases mass, length,
Winkler and CRPS **downward**. Check the `mass` field from `query_metrics`, or
widen the grid.

### Isolating the independence cost

`CPFN2D.independent()` returns the same prediction with the coupling discarded
(`p_mat → outer(marginals)`). Marginals — and therefore the **point CATE** — are
bit-identical; only the width changes. Measured at ρ=0.6, truth drawn from the
model's own joint:

| | cov@95% | length | Winkler | CRPS |
|---|---|---|---|---|
| joint | 95.1% | 1.588 | 1.933 | 0.2333 |
| Y0⊥Y1 | 99.6% | 2.489 | 2.515 | 0.2463 |

**1.57× longer intervals for the same point estimate**, matching the predicted
`1/√(1−ρ)` = 1.58. That is the joint's value in a number PEHE cannot see.

## Keeping the copy in sync

`eval_density_tauC.py` here is a **snapshot**, and the original does change.
Verify before trusting a run:

```bash
diff benchmarks/eval_graph2d/eval_density_tauC.py \
     case_study/density_eval/eval_density_tauC.py
```

Only three hunks should appear (`_ORIG`, the harness path, the `sys.path`
order). Anything else means the original moved forward — re-copy and re-apply
those three.

## Not yet adapted

This is a faithful relocation, not a case-study eval yet. The runner still reads
`DATASET` from the harness and expects IHDP / ACIC. Pointing it at the
case-study SCM data (`case_study/data*`, `case_study/d_variation/d*`) is the
next step and has not been done.

## Verifying the originals are untouched

```bash
diff benchmarks/eval_graph2d/density_common.py case_study/density_eval/density_common.py   # identical
diff benchmarks/eval_graph2d/eval_density_tauC.py case_study/density_eval/eval_density_tauC.py  # 3 path hunks
```
