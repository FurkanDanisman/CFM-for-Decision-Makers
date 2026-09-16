# CausalPFN-1D

CausalPFN with a 1D bar-distribution head, `J=1024` bars, **random head
initialisation** (no warm start from a pretrained head).

```bash
export DEPLOY_ROOT=$SCRATCH/rpfn_bench_kit
export CAUSALPFN=$DEPLOY_ROOT/external/causalpfn
sbatch Reproduce/Training/CausalPFN1D/train.sbatch
```

Produces `step_checkpoints/run/step_0050000.pt`; copy to
`Required_checkpoints/cpfn1d_j1024_headrand_step_50000.pt`.

## Evaluation note that belongs with the model

At evaluation this model is run with `STD_MODE=pooled`, not the `per_arm`
default. The density dump asserts that both arms share `y_shift`/`y_scale`,
because a single affine map cannot un-scale a difference of two
differently-scaled arms — see `../../paper_appendix/`, the per-arm
standardisation section. Its PEHE in our tables is therefore pooled-scaled
and will differ slightly from a `per_arm` run of the same checkpoint.
