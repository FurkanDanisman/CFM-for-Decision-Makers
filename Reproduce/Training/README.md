# Training

Six models. Four we train; two are upstream releases we evaluate as published.

| folder | model | checkpoint | ours? |
|---|---|---|---|
| `CausalPFN1D/` | CausalPFN, 1D bar head, random init | `cpfn1d_j1024_headrand_step_50000.pt` | yes |
| `CausalPFN2D/` | CausalPFN, 2D joint head | `cpfn2d_j32_random_step_50000.pt` | yes |
| `Graph2D/` | graph-conditioned 2D joint | `graph2d_step_50000.pt` | yes |
| `DoPFN_BB/` | DoPFN backbone + 2D head | `dopfn_bb_j10_step_150000.pt` | yes |
| `UWYK/` | Use-What-You-Know 1D | `uwyk_reproduce_best_model.pt` | no — authors' release |
| `DoPFN_Native/` | DoPFN as published | three artifact files | no — authors' release |

Every checkpoint lives in `Required_checkpoints/` (git LFS), except
DoPFN-native's artifacts — see `DoPFN_Native/README.md`.

## The 1D/2D pairs

The comparison the benchmarks are built around is a 1D head against a 2D head
on the same backbone and the same data:

```
CausalPFN1D  <->  CausalPFN2D
UWYK         <->  Graph2D
DoPFN_Native <->  DoPFN_BB
```

Within each pair the backbone is shared and only the output head differs, so
a difference in interval width is attributable to the head rather than to
capacity or training data. The mathematical consequence — that a 1D head
cannot identify the law of $\tau$ from two marginals, and the gap is governed
by the inter-arm correlation — is derived in `../paper_appendix/`.

## CausalPFN training runs upstream's loop

`CausalPFN1D` and `CausalPFN2D` both invoke `$CAUSALPFN/train.py`, CausalPFN's
own training loop, with our patches installed over it from
`rpfn_patches/causalpfn_step_ckpt/`:

- `checkpoint.py` — per-epoch saves carrying the true optimizer-step counter
- `trainer.py` — step checkpoints; the counter advances only after a
  successful `optimizer.step()`

So those two need the CausalPFN checkout at `$CAUSALPFN`.
`training_causalpfn2d/train_causalpfn_2d.py` is a standalone implementation
kept for reference, but it is **not** the path the released checkpoints came
from.

## Checkpoint retention

The training scripts set `callbacks.checkpoint.top_k=10000`, i.e. keep every
epoch. At J=32 (~216 MB/epoch) that is affordable. At J=1024 it writes 9.2 GB
roughly every 23 minutes and will fill a 2 TB scratch in days. Set `top_k=3`
unless you need the full history; eviction is by training loss, not recency,
and `latest.pt` is always retained for resume.
