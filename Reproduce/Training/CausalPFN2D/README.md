# CausalPFN-2D

CausalPFN with a 2D joint head over $(Y^{(0)}, Y^{(1)})$, `J=32`, random init.
Same backbone and training data as CausalPFN-1D; only the head differs.

```bash
export DEPLOY_ROOT=$SCRATCH/rpfn_bench_kit
export CAUSALPFN=$DEPLOY_ROOT/external/causalpfn
sbatch Reproduce/Training/CausalPFN2D/train.sbatch
```

Produces `step_checkpoints/run/step_0050000.pt`; copy to
`Required_checkpoints/cpfn2d_j32_random_step_50000.pt`.

`J=32` gives a $32\times32$ joint. The head emits $J^2$ bin logits plus region
and tail parameters; the $\tau$ density is the anti-diagonal projection of the
resulting matrix, which is why no independence assumption is needed.

`COMPILE=0` is set deliberately — `torch.compile` miscompiles this head.
