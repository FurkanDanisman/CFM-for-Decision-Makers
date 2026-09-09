# CATE density evaluation on RealCause IHDP and ACIC

`eval_density_tauC.py` scores the conditional distribution of
`tau = Y(1) - Y(0)` and reports point errors for its mean (CATE).
Choose `MODEL_FAMILY=uwyk` (default), `dopfn`, or `all`.

| Family | Density rows | Model inputs |
| --- | --- | --- |
| UWYK / g4cfm | `uwyk_native`, `uwyk_matched`, `joint` | Existing `UWYK_CKPT`, `UWYK_CFG`, `CKPT` |
| DoPFN | `dopfn_native`, `dopfn_joint` | `DOPFN_ROOT` and optional `DOPFN_JOINT_CKPT` |

`DOPFN_ROOT` is the upstream [Do-PFN checkout](https://github.com/jr2021/Do-PFN)
containing `scripts/`, `model/`, and the pretrained files
`artifacts/dopfn_config.pkl`, `artifacts/dopfn_model.pkl`, and
`artifacts/model_submitit_0ccc_id_171b69db_epoch_-1.cpkt`.
The regular model uses
`scripts.transformer_prediction_interface.base.DoPFNRegressor`, as in the
point benchmarks, but calls `predict_full()` for each treatment arm.
The joint model uses `DoPFNBackboneWith2DHead` and defaults to
`Required_checkpoints/dopfn_bb_j10_step_150000.pt` (J=10, 150,000 steps).
The checkpoint's `num_features=6` records its training setting; it is not an
inference feature cap.

For a one-realization DoPFN smoke run, from the repository root:

```bash
MODEL_FAMILY=dopfn \
DOPFN_ROOT=/path/to/Do-PFN \
CAUSALPFN=/path/to/CausalPFN \
DATASET=IHDP REAL_START=0 REAL_END=1 EVAL_MAX_CONTEXT=1000 \
OUT=./results_density_tauC/dopfn/IHDP \
python -u benchmarks/eval_graph2d/eval_density_tauC.py
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
python benchmarks/eval_graph2d/summarize_density_tauC.py results_density_tauC/dopfn
```

The summary detects available rows and adds a paired DoPFN native-to-joint
contrast. Use a separate output directory for each configuration to avoid
overwriting shards or mixing model families across realizations.

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
Both joint models use `Joint2D` and the existing diagonal integration with
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
python -m unittest discover -s benchmarks/eval_graph2d -p 'test_density_*.py'
```
