# Do-PFN reproduction (1-D) + joint head

A faithful rebuild of Do-PFN's prior-fitting pipeline, and the controlled joint
variant built on top of it.

Plan and full findings: <https://claude.ai/code/artifact/3b5ba99e-2886-4724-93c7-1a523ad28db1>

This directory is new and self-contained. `training_dopfn_base/` and
`training/data/PairedDoPFNDataset.py` are left untouched as reference points.

## Why a rebuild rather than a wrapper

Do-PFN's training script was never published — the upstream repo's eleven
commits never contained one. The shipped `.cpkt` is a full training checkpoint
though (state dict, optimizer state, scaler state, and the complete 131-key
config), so most of the recipe is recoverable rather than guesswork.

What is *not* recoverable is the batch-assembly code. Three things are therefore
inferred, and the parity harness exists to test them against the released
weights instead of taking them on faith.

The shipped prior also does not run. `prior_data_example.py` raises before
producing a batch; see REPAIR A/B/C in [`prior.py`](prior.py).

## Layout

Two deliverables, one shared pipeline:

```bash
python training_dopfn_repro/train.py --variant dopfn_1d --steps 150000   # task 1
python training_dopfn_repro/train.py --variant joint_2d --steps 150000   # task 2
```

| File | What it does |
|---|---|
| [`prior.py`](prior.py) | Do-PFN's SCM prior, repaired, seeded, emitting both arms |
| [`batch.py`](batch.py) | Context/query splice, y-space handling, splice corruptions |
| [`model.py`](model.py) | Shared backbone init + the two heads + losses |
| [`borders.py`](borders.py) | Bucket/grid fitting, and what the grid costs |
| [`train.py`](train.py) | Training loop for both variants |
| [`parity.py`](parity.py) | Validates the reconstruction against the released weights |

Cluster submission is deliberately out of scope — `train.py` takes `--steps`
and friends and is yours to wrap.

## Environment

Do-PFN's `model/layer.py` does `from torch.nn.modules.transformer import
Optional`, which only resolves on **torch 2.1**; the model cannot be unpickled
on newer torch. The prior itself has no such constraint.

```bash
uv run --python 3.10 --with "torch==2.1.*" --with "numpy<2" --with networkx \
  --with scipy --with dill --with tqdm --with scikit-learn --with einops \
  --with "pandas<2.2" python training_dopfn_repro/parity.py --n-batches 64
```

Set `DOPFN_SRC` if the Do-PFN checkout is not at the repo root's `Do-PFN/`.

## Status

**Done and verified.**

- Prior draws match the checkpoint's recorded distributions over 120 draws:
  `num_features` mean 3.55 (spec 3.5), `num_unobserved` 2.02 (2.0), `exo_std`
  1.90 (2.0), `noise_std` 0.0528 (0.05). No retries, no non-finite batches.
- Throughput 9 ms per (2200 × 4) batch — about 1.2 s per 128-step epoch, so the
  prior will not bottleneck training.
- `y_int` reproduces arm-selection exactly (max abs diff 0.0), which is what
  licenses the superset record below.
- The released checkpoint loads with **7,336,420** parameters — the paper's
  "7.3 million" — and `strict=True` state-dict load succeeds.

- Both variants train end to end, sharing backbone hash `e3309ae7c591`.
  `dopfn_1d` 7,336,420 params / 100 outputs; `joint_2d` 7,346,417 / 113.
- The paired stream is verified through the full batch pipeline, not just the
  prior: identical context tensors, identical split point, and the 1-D target
  equals whichever joint arm the coin flip selected.
- Initialisation is reproducible and complete — 24/24 attention tensors respond
  to the seed, against 0/24 under a naive `reset_parameters` sweep.

**Not built yet:** evaluation (the existing `benchmarks/eval_density_tauC.py`
pipeline is where these should plug in), and a mediator-restricted splice test.

## What J=10 costs

Measured by [`borders.py`](borders.py) (`python -m training_dopfn_repro.borders`).
The answer splits cleanly, and conflating the two halves is the easiest way to
mistake a resolution limit for a model failure:

| | J=10 | J=32 |
|---|---|---|
| tau knots representable | 19 | 63 |
| grid bin width | 0.73 (0.49 sd) | 0.25 (0.15 sd) |
| **CATE / mean floor** | **0.00%** | **0.00%** |
| **p(tau\|x) best-case rel. L2** | **72.5%** | **49.9%** |
| mass in tail regions | 3.8% | 3.0% |

**Means are free.** A bar or grid head represents `E[y]` as `Σ pᵢcᵢ`, which
interpolates continuously between bin centres — so the grid constrains the mean
only by the hull of its centres, and nothing here falls outside it. J=10 costs
*nothing* for CATE. (An earlier snap-to-bin-centre estimate of this was wrong:
neither head predicts a bin centre, they predict distributions.)

**Densities are not.** The tau density a uniform J×J grid can express is
piecewise linear with knots one bin width apart. At J=10 that is 19 knots, and
even a perfect model carries ~72% relative L2 error against the true density.
Since `eval_density_tauC.py` scores exactly `p(tau|x)`, this is the binding
constraint on that metric — and it is a floor, not a model failure.

Caveat: the 72.5% is computed on the *pooled* tau distribution. Conditional
densities are likely sharper still, since `noise_std` averages 0.05, so treat
it as a lower bound on the cost. `--j-2d` is a one-line change if it turns out
to matter.

## One prior stream, both models

The 1-D target is a strict subset of the joint target. In the prior, `t_int` is
a per-row coin flip and `y_int` is propagated under `do(t_int)` sharing the
observational exogenous noise — so `y_int[k]` *is* `y_do1[k]` or `y_do0[k]`,
selected by the flip. Propagation is row-wise, so selecting per row from two
pure arms is identical to propagating the mixed treatment vector in one pass
(verified exactly, not approximately).

`sample_batch` therefore emits one superset record and each trainer takes what
it needs. Both models see the same SCMs in the same order, which makes the
comparison paired rather than merely matched.

```
x_obs (S,B,F+1)   col 0 = observational treatment
y_obs (S,B)       observational outcome
t_int01 (S,B)     coin-flipped intervention label
y_int (S,B)       = where(t_int01, y_do1, y_do0)      <- 1-D target
y_do0, y_do1      both arms, shared noise             <- joint target
x_int (S,B,F+1)   post-intervention covariates (diagnostic)
```

## Parity results

64 paired batches at seq_len 1200, plus a 20-batch confirmation at 700. Deltas
are against the inferred contract, in raw-y nats; positive means worse.

| Variant | Δ NLL | Verdict |
|---|---|---|
| `y_space=raw` | **+2.60 ± 0.52** | **Settled: y must be context-standardised** |
| `splice=flipped` | +4.98 ± 1.24 | Query polarity must agree with context |
| `splice=obs_only` | +2.71 ± 0.66 | Model genuinely uses the intervention signal |
| `splice=flipped_both` | −0.01 ± 0.04 | Tie — consistent relabelling is free |
| `splice=x_int_full` | −0.05 ± 0.03 | Tie — see caveat below |
| `eval_pos=fixed_query` | −0.05 ± 0.01 | Uninformative — see caveat below |

**The headline catch.** Feeding raw y is 2.6 nats worse than feeding
context-standardised y. The released borders span `[-56.1, 256.1]`, which reads
like raw prior y — but that is just what standardising a heavy-tailed prior
leaves behind: fine buckets (0.049) near zero, very long tails. The config's
`transform_target=True` was the clue. Defaulting to raw would have silently
degraded the entire reproduction, which is exactly what this harness is for.

*Downstream consequence:* predictions come back in standardised units. Multiply
by the context sigma before reporting CATE.

**`binary_strategy`** — judged on excess over a context-fitted Gaussian, since
it changes the data rather than its presentation. `extreme` beat `mean` in both
runs (excess 0.182 vs 0.240, and −2.053 vs −1.750). Weak-to-moderate, consistent,
and it agrees with what `PairedDoPFNDataset` already chose. Adopted.

**Two things this harness cannot settle.** `eval_pos=fixed_query` uses a larger
context than the paper sampler's average, and more context trivially lowers NLL
— the comparison measures context size, not faithfulness. And `x_int_full`
ties because the two splices only differ on covariates that are *descendants of
t*; giving it real power needs a prior restricted to SCMs where such mediators
exist. So the splice rests on the evaluation contract and the paper, not on NLL
evidence — worth stating plainly in the write-up.

## The three inferred assumptions

1. **The splice.** Query rows keep *observational* covariates; only column 0 is
   overwritten with the intervened treatment. Taken from the evaluation contract
   at `Do-PFN/datasets/__init__.py:222-227`, which agrees with the paper's
   Algorithm 1 and with how `predict_cate` holds covariates fixed. The training
   loop is gone, so this is inferred — it is the assumption everything rests on.
2. **`binary_strategy`.** Read by `set_binarization_params`, never assigned
   anywhere, and absent from the checkpoint config. A genuine free choice.
3. ~~**The y space.**~~ **Settled** — context-standardised, by 2.6 nats. See
   above.

`parity.py` adjudicates these empirically. Assumption 2 is settled, assumption 3
is settled, and assumption 1 is *not* settleable this way — the harness ties.

## Reading the harness

Variants that change only the *presentation* of fixed data (the splice
corruptions) are compared on paired NLL deltas. Variants that change the
*data* (`binary_strategy`) would otherwise just measure which distribution is
intrinsically easier, so they are compared on **excess over a context-fitted
Gaussian**, which cancels the difficulty term. Different y spaces are put in
common units by the log-Jacobian, without which their NLLs are not comparable
at all.

## Why reproduce rather than use the library

The upstream repo ships a trained checkpoint and an inference wrapper, but no
training script. Without it, several details of the training recipe are
ambiguous — and every one of them is a detail a joint model would also have to
commit to. Building a joint head on top of the library would mean guessing those
answers silently and differently from whatever Do-PFN did, which makes any
1-D-vs-joint comparison uninterpretable.

So the reproduction exists to **pin the recipe down well enough that the joint
version is a controlled comparison.** That reframes what "faithful" has to mean
here. Where an ambiguity can be resolved against the released weights, resolve
it. Where it cannot, the requirement weakens from *historically correct* to
**identical across both runs and written down** — which is enough for the
comparison, and is all that the missing script denies us.

## Decisions on record

Choices the released artifact does not determine. Each is fixed, shared by every
variant, and justified here rather than buried in code.

| Choice | Value | Basis |
|---|---|---|
| `binary_strategy` | `extreme` | Unrecorded in the checkpoint. Parity favoured it in both runs (excess 0.182 vs 0.240; −2.053 vs −1.750) — weak-to-moderate, not decisive. Matches `PairedDoPFNDataset`. |
| `y_space` | `zscore_ctx` | Settled empirically: +2.60 ± 0.52 nats better than raw. |
| Query splice | observational x, column 0 intervened | The repo's own eval contract and the paper. Parity ties, so this is not independently confirmed. |
| `M_ob` sampler | paper (`U{10..2200}`) | Config contradicts itself; parity cannot adjudicate (confounded by context size). |
| Initialisation | from scratch, shared file | Cleanest claim: head is the only moving part. |

## Deviations from the shipped code

Each is forced by a gap in the released artifact.

- **Repairs A, B, C** in `prior.py` — the prior does not otherwise run.
- **Re-implementation, not a wrapper.** `get_batch` emits one arm and reads
  ambient RNG state; both are incompatible with the goals here.
- **Treatment binarization via the level midpoint** rather than
  `get_zero_one_treatment`'s column mean. Identical whenever both levels are
  present; the column-mean version degenerates to all-zeros on a pure arm,
  which the joint task needs.
- **Two arm propagations per batch** instead of one mixed propagation.
  Numerically identical, verified.
- **Exact weights are unreachable.** The checkpoint is a continuation of an
  unpublished chain (`continue_model_path` is set) with no recorded seed.
- **The two heads allocate resolution differently, and cannot be made to
  match.** Do-PFN's 1-D borders are quantile-allocated; `neg_log_prob_2d`
  carries a single scalar `bin_width` and so *requires* a uniform grid. This is
  the one asymmetry between the tasks that is forced rather than chosen.
- **The 2-D grid spans a robust quantile range, not min-to-max.**
  `fit_edges_2d` uses min/max, which on this prior means roughly `[-17, 13]` and
  a J=10 bin width near 3.0 — wide enough that nearly all mass lands in two
  bins. We span `[1%, 99%]` instead (bin width 0.73, 98% of mass inside) and let
  the 9-region tail structure absorb the rest.
- **Degenerate batches are skipped, not trained on.** `doscm` signals a bad SCM
  by filling the batch with `-100`; the loop steps over those and reports the
  count rather than letting them move the weights.
- **The 2-D head's tail scales start wide** (`TAIL_SCALE_INIT_LOGIT = 6.0`).
  See below — this is a robustness fix, not a preference.

## The heavy tail problem, and why it only bites the joint head

Worth knowing before the long run, because it is the one place where the two
tasks genuinely pull against each other.

`BarDistribution2D` sets `scale = bin_width * (softplus(raw) + 1e-3)`, so at
`raw = 0` the Gaussian tails have scale ≈ 0.45 in standardised units. The
prior's heaviest draws reach `|y| ≈ 50`, which then sits ~108 sigma outside the
grid and costs thousands of nats on a *single* query point. Measured on 40
batches with an untrained head: batch loss correlates with `max|y|` at
**+0.96**, median 2.95 nats/outcome but worst case **63**.

The 1-D head does not suffer this: `FullSupportBarDistribution` ties its
half-normal tail width to the outermost bucket, which quantile allocation makes
wide.

Biasing the four tail logits to 6.0 fixes initialisation — worst case drops
19.6 → 3.14 with the median unchanged at 2.95, and only tail *shape* is
affected since mass allocation is the separate 9-way region softmax.

**The underlying tension is not fully resolved, and it is worth a decision.**
Gaussian tails cannot cheaply accommodate a heavy-tailed target. Note that
`PairedDoPFNDataset` never hit this because it min-max scaled y to `[-1, 1]`,
which *bounds* the target. Switching to the standardisation that parity proved
the 1-D head wants reintroduces unbounded tails. The options:

1. **Keep standardisation for both** (current). Faithful to what parity
   measured; the joint run leans on the wide tail init and gradient clipping.
2. **Min-max both to `[-1, 1]`.** Well-behaved 2-D loss, but gives up ~2.6 nats
   of measured 1-D fidelity.
3. **Standardise, then winsorise** at e.g. ±6 sd. Keeps the validated transform
   for ~99.9% of the data and bounds the pathology — but clamping is a change
   to the *task*, not a reparameterisation, so it is not something the parity
   harness can score cleanly.

Using different target spaces per head is the one option that is not available:
it would break exactly the comparability the reproduction exists to buy.

## Quirks preserved deliberately

- **One graph per batch.** `batch_size` datasets share a graph *and* one set of
  structural-equation weights, differing only in noise. So a run of N steps sees
  N distinct SCMs, not N × batch_size.
- **Inverted treatment polarity.** `get_zero_one_treatment` maps *below-mean*
  raw values to 1, so the lower raw level is the "treated" arm.
- **Noise drawn once at SCM construction**, which is what makes the two arms
  genuine counterfactuals rather than independent draws.
- **The `-100` sentinel** on non-finite batches (`on_nonfinite="resample"`
  switches to redrawing; the change is then yours to document).
