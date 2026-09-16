# DoPFN-bb

DoPFN's backbone with a 2D joint head trained on top of it — the 2D
counterpart to DoPFN-native in the 1D/2D comparison.

```bash
export DEPLOY_ROOT=$SCRATCH/rpfn_bench_kit
export DOPFN_ROOT=$DEPLOY_ROOT/external/dopfn
sbatch Reproduce/Training/DoPFN_BB/train.sbatch
```

Produces `dopfn_bb_j10_step_150000.pt`. `J=10` — a coarser joint than
CausalPFN-2D's `J=32`, which shows up directly as a coarser $\tau$ support
($2J-1$ atoms).

The head is `training_dopfn_base/dopfn_backbone_head.py`
(`DoPFNBackboneWith2DHead`), which loads the frozen DoPFN backbone from
`$DOPFN_ROOT` and trains only the joint head. The checkpoint stores `config`
(carrying `J`) and `edges`, and `density_dopfn.py` asserts the edges are
uniform and increasing before use.
