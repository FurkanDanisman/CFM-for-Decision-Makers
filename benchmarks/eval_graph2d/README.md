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
| DoPFN | `dopfn_<name>`, one per `DOPFN_MODELS` entry | `DOPFN_ROOT` and `DOPFN_MODELS` |
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

`DOPFN_MODELS` lists the DoPFN models, comma-separated. `native` is the library
model above and takes no path; every other entry is `name=checkpoint` and is
scored as row `dopfn_<name>`. The Slurm launcher holds the list
(`DOPFN_MODEL_LIST` in `benchmarks/cluster/submit_density_tauC.sbatch`):

    native
    repro_1d_J10=Required_checkpoints/new/dopfn_repro_1d_J10_step150000.pt
    repro_1d_J100=Required_checkpoints/new/dopfn_repro_1d_J100_step150000.pt
    repro_joint2d=Required_checkpoints/new/dopfn_repro_joint2d_step150000.pt

Direct runs without `DOPFN_MODELS` fall back to the same four. A checkpoint's
kind comes from the file, never from its name:

| Checkpoint | Kind | Density |
| --- | --- | --- |
| `training_dopfn_repro`, `dopfn_1d` | 1D | FullSupportBarDistribution arms, convolved under independence |
| `training_dopfn_repro`, `joint_2d` | joint | `Joint2D`, diagonal-integrated with tails |
| `training_dopfn_base` (`dopfn_bb_*`) | joint | `Joint2D`, diagonal-integrated with tails |

Checkpoints' training-time feature counts (at most 6) are not inference
feature caps.

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

Use `MODEL_FAMILY=dopfn` to run only the DoPFN models. `DOPFN_QUERY_CHUNK=20`
controls query batching for all of them. Each entry adds a density row, so
check the first task's runtime against the Slurm time limit.
`DOPFN_JOINT_CKPT` is no longer read; add that checkpoint to `DOPFN_MODELS`.

```bash
python benchmarks/eval_graph2d/summarize_density_tauC.py \
  results_density_tauC/dopfn --model dopfn
```

`--model uwyk` displays only UWYK/g4cfm, with its selected graph. `--model
dopfn` displays the DoPFN rows the run produced, with `graph=none`, the
checkpoint behind each row, and a contrast from every 1D row to every joint
row. It flags a row name that points at different checkpoints across shards. The default
`--model auto` detects every available family but prints each one as a separate
section, never in a combined table. `--model all` is an explicit synonym for
that behavior. Existing mixed result shards remain usable, so changing the
summary selection does not require rerunning inference.

New model families can be added as another entry in `MODEL_METHODS` and
`MODEL_LABEL` in `summarize_density_tauC.py`; the evaluator also needs the
corresponding inference adapter before accepting a new `--model` value. Use a
separate output directory for each evaluation configuration to avoid
overwriting shards or mixing configurations across realizations.

All models share the deterministically selected context rows and the outcome
axis defined by `Y_SCALING=minmax` (default) or `Y_SCALING=std` with
`STD_TARGET=0.3`. Each DoPFN model gets the inputs it was trained on. Native
DoPFN receives raw features/outcomes and applies its own preprocessing; its
returned outcome borders are mapped to the common axis. `training_dopfn_repro`
models receive raw features with the treatment in column 0 (their transformer
normalizes features itself) and factual outcomes z-scored with context
statistics (`torch.std`, correction=1), as in `training_dopfn_repro/batch.py`.
The 1D query sets column 0 to the arm; the joint query sets it to NaN, the
training placeholder. Their outputs are mapped affinely from the z-scored axis to
the common axis. `training_dopfn_base` joints receive the harness's standardized
features and scaled factual outcomes. Every DoPFN model receives all covariates;
UWYK retains its checkpoint feature cap. DoPFN consumes no adjacency matrix.

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
the saved mass and finite-grid mean diagnostics. `SAVE_PREDICTIONS=1` saves,
for CPU-only rescoring, the truth and outcome transform plus each model's raw
outputs: `dopfn_pred0/1`, `dopfn_borders0/1_raw` and `dopfn_tail_scales0/1_raw` for
native DoPFN, and keys prefixed by the row name for checkpoints (`_pred0/1`,
`_borders_native` or `_logits`, `_edges2d_native`, `_y_shift`, `_y_scale`,
`_ckpt`). `dopfn_methods`, `dopfn_kinds` and `dopfn_sources` record the list.
Every joint row also gets `<row>_inner` point errors for its interior mean.

Validation:

```bash
python benchmarks/eval_graph2d/test_density_common.py
python -m unittest discover -s benchmarks/eval_graph2d -p 'test_density_dopfn.py'
python -m unittest discover -s benchmarks/eval_graph2d -p 'test_density_causalpfn.py'
python -m unittest discover -s benchmarks/methods -p 'test_dopfn_compat.py'
```

## MALC arm

`eval_density_tauC_malc.py` re-scores the raw run's prediction dumps with
region 0 — the inner × inner block — replaced by a K=1 log-concave MLE. The
8 tail regions go through the identical code path with identical quadrature,
so a raw-vs-MALC difference is attributable to region 0 and nothing else. It
is CPU-only: no checkpoint, no GPU, no dataset loader. Output schema matches
the raw arm, so `summarize_density_tauC.py` reads either directory and the two
tables compare row by row.

Both model families reach MALC the same way, which is the point of doing it
this way. A trained joint head hands over its own `p_mat`; a pair of 1D arms
hands over `f0 ⊗ f1` under independence. Same estimator, same input shape,
same output path, so 1D-vs-joint stays a comparison rather than two pipelines.

DoPFN's 1D arms need one extra step, and skipping it fails *silently*.
`MALC_2D` calibrates its Beta jitter from a single bin width read off
`grid_x[1] - grid_x[0]` and validates nothing about the rest, while DoPFN's
borders are quantile allocated — measured on IHDP r000 at 0.107 to 536, a
**5032× ratio**. So the arms are rebinned onto a uniform grid first
(`DOPFN_MALC_BINS`, default 1024). CDF interpolation is exact on a histogram's
piecewise-linear CDF, so this costs 0.004% relative L2 against the exact
unequal-bar tau density (0.35% at 100 bins, 0.013% at 512). UWYK's bars are
already uniform — measured ratio 1.0001 — so the same code path is a no-op
there and its numbers are unchanged. Fallback queries are scored on the
*native* unequal bars, exactly as the raw arm scored them.

`MODEL_FAMILY` selects `uwyk`, `dopfn`, `all`, or `auto` (default: whichever
families the dump carries). Both dump schemas are read — the current
`DoPFNModelSet` layout and the older single-model `DoPFNDensityModels` one.

**`MALC_B` matters more for DoPFN than for UWYK.** Measured on one IHDP
realization at `B=100`: `dopfn_native` 0/75 fallbacks, but the DoPFN joint
11/75 (14.7%), an order of magnitude above UWYK's joint. Its head is J=10, so
its `p_mat` has 100 bins against the UWYK joint's 1024, and 100 synthetic
points drawn from that coarse a grid hull a much smaller region. Use the
default `B=1000` for anything reportable.

```bash
DUMPS=./results_density_tauC/<JOBID>/IHDP/predictions \
OUT=./results_density_tauC_malc/IHDP \
MODEL_FAMILY=dopfn MALC_B=1000 N_WORKERS=32 \
python -u benchmarks/eval_graph2d/eval_density_tauC_malc.py

python benchmarks/eval_graph2d/test_density_malc.py   # 12 gates, no model
```
