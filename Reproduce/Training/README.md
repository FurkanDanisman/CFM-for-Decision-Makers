# Training

Four checkpoints. All land in `Required_checkpoints/` (git LFS).

| model | entry point | checkpoint |
|---|---|---|
| CausalPFN 2D | `CausalPFN2D/train.sbatch` | `cpfn2d_j32_random_step_50000.pt` |
| CausalPFN 1D | `CausalPFN2D/train_1d.sbatch` | `cpfn1d_j1024_headrand_step_50000.pt` |
| Graph2D | `Graph2D/train.sbatch` | `graph2d_step_50000.pt` |
| DoPFN-bb | `DoPFN_BB/train.sbatch` | `dopfn_bb_j10_step_150000.pt` |

UWYK is **not** trained here — `uwyk_reproduce_best_model.pt` comes from the
UWYK authors' own release.

## Not self-contained: CausalPFN training runs upstream's loop

`submit_train_cpfn2d_*` invokes `$CAUSALPFN/train.py`, i.e. CausalPFN's own
training loop, with our patches installed over it from
`rpfn_patches/causalpfn_step_ckpt/`:

- `checkpoint.py` — per-epoch saves, carries the true optimizer-step counter
- `trainer.py` — step checkpoints, step counter incremented only after a
  successful `optimizer.step()`

So reproducing these two checkpoints needs the CausalPFN checkout at
`$CAUSALPFN`. `training_causalpfn2d/train_causalpfn_2d.py` exists in the repo
but is NOT the path these checkpoints came from.

## Checkpoint retention

The training sbatch files set `callbacks.checkpoint.top_k=10000`, i.e. keep
every epoch. At j32 (216 MB/epoch) that is affordable; at j1024 (9.2 GB every
~23 min) it fills a 2 TB scratch in days. Set `top_k=3` unless you
specifically need the full history. Eviction is by train loss, not recency,
and `latest.pt` is always kept for resume.
