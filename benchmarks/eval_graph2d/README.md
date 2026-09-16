# CATE density evaluation on RealCause IHDP and ACIC

`eval_density_tauC.py` scores the conditional distribution of
`tau = Y(1) - Y(0)` and reports point errors for its mean (CATE).
Choose the model with `--model uwyk`, `--model dopfn`, `--model causalpfn`,
or `--model all` (all three pairs).
`MODEL_FAMILY` provides the same selection for the Slurm launcher and remains
available for existing commands; the command-line flag takes precedence.

| Family | Density rows | Model inputs |
| --- | --- | --- |
| UWYK / g4cfm | `uwyk_native`, `uwyk_matched`, `joint` | Existing `UWYK_CKPT`, `UWYK_CFG`, `CKPT` |
| DoPFN | `dopfn_native`, `dopfn_joint` | `DOPFN_ROOT` and optional `DOPFN_JOINT_CKPT` |
| CausalPFN | `causalpfn_native`, `causalpfn_joint` | `CAUSALPFN` checkout; optional `CAUSALPFN_CKPT`, `CAUSALPFN_JOINT_CKPT` |

## CausalPFN

The default checkpoints are
`Required_checkpoints/cpfn1d_j1024_headrand_step_50000.pt` and
`Required_checkpoints/cpfn2d_j32_random_step_50000.pt`. Run one realization:

```bash
CAUSALPFN=/path/to/CausalPFN \
REAL_START=0 REAL_END=1 EVAL_MAX_CONTEXT=1000 \
OUT=./results_density_tauC/causalpfn/IHDP \
python -u benchmarks/eval_graph2d/eval_density_tauC.py \
  --model causalpfn --dataset IHDP
```

Use `--dataset ACIC`, the corresponding `OUT=.../ACIC`, and the existing
`ACIC_CACHE_DIR` for ACIC. CausalPFN-only runs need no UWYK or DoPFN
checkpoints. The shared harness still requires the `g4cfm` source tree.
`CAUSALPFN_QUERY_CHUNK=512` controls batching for both CausalPFN heads.

```bash
MODEL_FAMILY=causalpfn sbatch benchmarks/cluster/submit_density_tauC.sbatch
python benchmarks/eval_graph2d/summarize_density_tauC.py \
  results_density_tauC/causalpfn --model causalpfn
```

Both models use the shared context selection and standardized covariates,
padded to their checkpoint feature counts (99 for these checkpoints). They
receive no graph. Both use **pooled outcome standardization** computed from
that context (`torch.std`, correction=1), matching the joint checkpoint's
training. This differs from the per-arm default of the older point-evaluation
scripts. `STD_MODE` does not change this density adapter. The model outcome
axes are mapped to the shared `Y_SCALING` scoring axis after inference;
`Y_SCALING` and `STD_TARGET` do not change the CausalPFN input transform.

The 1D model uses all 1024 finite bins on its native `[-10, 10]` axis. Its
independence convolution is analytic and preserves exact zero density outside
its support. Such observations produce **NLL = +inf**, retained in summaries;
`frac_zero_density_causalpfn_native` reports their frequency. Grid KL follows
the existing scorer's density floor and is not full-support KL for this
finite-support model. The joint model uses `Joint2D` with all 32² bins,
nine region weights and four tail scales. Its tails are not discarded or
renormalized. `causalpfn_joint_inner` reports the additional interior-mean
point metrics. This is a comparison of the supplied models at their native
resolutions; no resolution-matched CausalPFN row is added.

Prediction dumps include both 1D logits, full joint logits, native and
transformed edges, context outcome transform, checkpoint paths and feature
counts. Reconstruct native arms with `CausalPFN1D.from_pred` using
`causalpfn_edges1d`. Reconstruct the joint with `Joint2D.from_pred` using
`causalpfn_edges2d_native`, then call `.affine(causalpfn_y_scale / y_scale,
(causalpfn_y_shift - y_shift) / y_scale)` to preserve the native correlation
and transform its tail scales. The transformed bin
knots generally do not align with the fixed tau grid: point NLL is evaluated
directly, while grid metrics retain integration error. Check mass and
finite-grid mean diagnostics as with DoPFN.

## DoPFN and shared scoring

`DOPFN_ROOT` is the upstream [Do-PFN checkout](https://github.com/jr2021/Do-PFN)
containing `scripts/`, `model/`, and the pretrained files
`artifacts/dopfn_config.pkl`, `artifacts/dopfn_model.pkl`, and
`artifacts/model_submitit_0ccc_id_171b69db_epoch_-1.cpkt`.
The regular model uses
`scripts.transformer_prediction_interface.base.DoPFNRegressor`, as in the
point benchmarks, but calls `predict_full()` for each treatment arm.
The shared shim in `benchmarks/methods/dopfn.py` accepts both scikit-learn
finite-validation keyword names (`force_all_finite` and `ensure_all_finite`)
and translates to the installed API. No scikit-learn downgrade is needed for
the removed-keyword error on newer installations.
The density adapter loads this shim by its file path and binds it to the
actual regressor immediately before `fit()`, including validation methods
whose modules were replaced during model loading. Before the first fit it
prints `[dopfn-compat] sklearn=... shim=... validated=...` after checking both
keyword spellings. If the error persists after updating, sync **both**
`benchmarks/methods/dopfn.py` and `benchmarks/eval_graph2d/density_dopfn.py`
to the cluster checkout used by the job and start a new job. A run reaching
`fit()` without that log line is not using the updated adapter.
The joint model uses `DoPFNBackboneWith2DHead` and defaults to
`Required_checkpoints/dopfn_bb_j10_step_150000.pt` (J=10, 150,000 steps).
The checkpoint's `num_features=6` records its training setting; it is not an
inference feature cap.

For a one-realization DoPFN smoke run, from the repository root:

```bash
DOPFN_ROOT=/path/to/Do-PFN \
CAUSALPFN=/path/to/CausalPFN \
REAL_START=0 REAL_END=1 EVAL_MAX_CONTEXT=1000 \
OUT=./results_density_tauC/dopfn/IHDP \
python -u benchmarks/eval_graph2d/eval_density_tauC.py \
  --model dopfn --dataset IHDP
```

Set `DATASET=ACIC` and change `OUT` to the corresponding `ACIC` directory
for ACIC; provide its cached data via `ACIC_CACHE_DIR` as in the existing
runner. `CAUSALPFN` supplies the RealCause loaders/data. The shared harness
still imports graph utilities from `UWYK` (defaults to this repo's `g4cfm`),
but DoPFN-only runs require no UWYK checkpoints or YAML config.

To include DoPFN in the existing Slurm array:

```bash
MODEL_FAMILY=all DOPFN_ROOT=/path/to/Do-PFN \
sbatch benchmarks/cluster/submit_density_tauC.sbatch
```

Use `MODEL_FAMILY=dopfn` to run only the new pair. `DOPFN_QUERY_CHUNK=20`
controls query batching for both DoPFN models. Combined runs take longer
than the existing three-row evaluation; adjust Slurm time if needed.

```bash
python benchmarks/eval_graph2d/summarize_density_tauC.py \
  results_density_tauC/dopfn --model dopfn
```

`--model uwyk` displays only UWYK/g4cfm, with its selected graph. `--model
dopfn` displays only native and joint DoPFN, with `graph=none`. The default
`--model auto` detects every available family but prints each one as a separate
section, never in a combined table. `--model all` is an explicit synonym for
that behavior. Existing mixed result shards remain usable, so changing the
summary selection does not require rerunning inference.

New adapters can be added as another entry in `MODEL_METHODS` and
`MODEL_LABEL` in `summarize_density_tauC.py`; the evaluator also needs the
corresponding inference adapter before accepting a new `--model` value. Use a
separate output directory for each evaluation configuration to avoid
overwriting shards or mixing configurations across realizations.

All models share the deterministically selected context rows and the outcome
axis defined by `Y_SCALING=minmax` (default) or `Y_SCALING=std` with
`STD_TARGET=0.3`. Native DoPFN receives raw features/outcomes and applies its
own preprocessing. Its returned outcome borders are mapped to the common
axis. Joint DoPFN receives the harness's standardized features and scaled
factual outcomes. Both DoPFN models receive all covariates; UWYK retains its
checkpoint feature cap. DoPFN consumes no adjacency matrix.

Native DoPFN assumes independence between the two predicted arms. Its first
and last logits represent half-normal tails anchored at the *inner* borders;
the other bins may have unequal widths. The density adapter preserves all
mass and evaluates the independence integral analytically, including tails.
It derives widths from returned outcome borders because upstream versions
can leave `criterion.bucket_widths` stale after rescaling borders. Thus the
reported mean is the mean of the reconstructed full density; it can differ
from upstream `predict_cate()` when that stale-width behavior is present.
All joint models use `Joint2D` and the existing diagonal integration with
tails. Native DoPFN is not rebinned to J=10: its comparison includes resolution
and preprocessing differences, as well as the learned joint dependence.

NLL, L2, KL, and grid mass use the shared scaled tau axis; PEHE, CATE L1, and
absolute ATE error use original outcome units. The 0.0005 tau grid contains
the J=10 joint's nominal knots; native DoPFN's adaptive knots generally do
not align, so its *grid metrics* retain numerical integration error. Check
the saved mass and finite-grid mean diagnostics. `SAVE_PREDICTIONS=1` saves
both native logits, raw borders/tail scales, joint logits/edges, outcome
transform, and truth for CPU-only rescoring. `dopfn_joint_inner` additionally
reports point errors for the joint's interior mean.

Validation:

```bash
python benchmarks/eval_graph2d/test_density_common.py
python -m unittest discover -s benchmarks/eval_graph2d -p 'test_density_dopfn.py'
python -m unittest discover -s benchmarks/eval_graph2d -p 'test_density_causalpfn.py'
python -m unittest discover -s benchmarks/methods -p 'test_dopfn_compat.py'
```
