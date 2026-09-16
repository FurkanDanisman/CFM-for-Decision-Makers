# DoPFN-native (not trained here)

DoPFN exactly as released. There is no checkpoint of ours — `DoPFNRegressor`
loads three artifact files from the DoPFN checkout:

```
$DOPFN_ROOT/artifacts/dopfn_model.pkl
$DOPFN_ROOT/artifacts/dopfn_config.pkl
$DOPFN_ROOT/artifacts/model_submitit_0ccc_id_171b69db_epoch_-1.cpkt
```

`benchmarks/eval_graph2d/density_dopfn.py` asserts all three exist before
loading. To make them fetchable alongside the other checkpoints:

```bash
mkdir -p Required_checkpoints/dopfn_native
cp $DOPFN_ROOT/artifacts/dopfn_model.pkl \
   $DOPFN_ROOT/artifacts/dopfn_config.pkl \
   $DOPFN_ROOT/artifacts/model_submitit_0ccc_id_171b69db_epoch_-1.cpkt \
   Required_checkpoints/dopfn_native/
git lfs track "Required_checkpoints/dopfn_native/*"
git add .gitattributes Required_checkpoints/dopfn_native
```

The `.pkl` files unpickle DoPFN class references, so `$DOPFN_ROOT` must still
be importable at evaluation time. Copying makes them *retrievable*, not
*independent*.

## A property of the released model we reproduce deliberately

`predict_full` rescales `criterion.borders` to raw outcome units but leaves
`criterion.bucket_widths` normalised, because the latter is a
`register_buffer` fixed at construction. `FullSupportBarDistribution.mean`
then forms `borders[:-1] + bucket_widths/2`, adding a normalised half-width
to a raw-unit border, and derives both half-normal tail scales from
normalised widths.

`predict_cate` is that mean, so the PEHE we report inherits it. We keep it:
the reported PEHE must be what the released model computes. The calibration
pipeline therefore scores this model's density at the same representative
points, so PEHE and the intervals never describe different estimators. The
closed form of the difference this makes — and why it vanishes on a uniform
grid but not on DoPFN's quantile-spaced one — is in `../../paper_appendix/`.
