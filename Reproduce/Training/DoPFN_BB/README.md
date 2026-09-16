# DoPFN-bb, and the DoPFN-native artifacts

## DoPFN-bb (ours)

Trained by `training_dopfn_base/train.py`: DoPFN's backbone with a 2D joint
head. Produces `dopfn_bb_j10_step_150000.pt`.

## DoPFN-native has no checkpoint of ours

`dopfn_native` is DoPFN exactly as released. It loads the authors' artifacts
through `DoPFNRegressor`, and those are three files in the DoPFN checkout,
not a single `.pt`:

```
$DOPFN_ROOT/artifacts/dopfn_model.pkl
$DOPFN_ROOT/artifacts/dopfn_config.pkl
$DOPFN_ROOT/artifacts/model_submitit_0ccc_id_171b69db_epoch_-1.cpkt
```

`benchmarks/eval_graph2d/density_dopfn.py` asserts all three exist before
loading. To make them fetchable like the rest:

```bash
mkdir -p Required_checkpoints/dopfn_native
cp $DOPFN_ROOT/artifacts/dopfn_model.pkl \
   $DOPFN_ROOT/artifacts/dopfn_config.pkl \
   $DOPFN_ROOT/artifacts/model_submitit_0ccc_id_171b69db_epoch_-1.cpkt \
   Required_checkpoints/dopfn_native/

git lfs track "Required_checkpoints/dopfn_native/*"
git add .gitattributes Required_checkpoints/dopfn_native
git commit -m "Required_checkpoints: DoPFN-native artifacts"
```

The `.pkl` files unpickle DoPFN class references, so `$DOPFN_ROOT` must still
be importable at eval time even once the files live here — copying them makes
them *retrievable*, not *independent*.

## A bug in the released model, which we reproduce deliberately

`predict_full` rescales `criterion.borders` to raw outcome units but leaves
`criterion.bucket_widths` in normalised units, because the latter is a
`register_buffer` set once in `__init__`. `FullSupportBarDistribution.mean`
then computes `borders[:-1] + bucket_widths/2` — a normalised half-width
added to a raw-unit border — and derives both half-normal tail scales from
normalised widths.

`predict_cate` is that mean, so the PEHE we report for DoPFN-native inherits
it. We keep it: the reported PEHE must be what the released model computes.
The calibration pipeline therefore scores DoPFN-native's density at those
same `bucket_means`, so PEHE and the intervals describe one estimator. See
`paper_appendix/` for the derivation.
