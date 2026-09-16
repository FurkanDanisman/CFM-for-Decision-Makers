# Graph2D

Graph-conditioned interventional PFN with a 2D joint head. Consumes a partial
ancestor matrix (PAM) alongside the context, so it can be conditioned on
known graph structure at evaluation time.

```bash
export DEPLOY_ROOT=$SCRATCH/rpfn_bench_kit
sbatch Reproduce/Training/Graph2D/train.sbatch
```

Produces `graph2d_step_50000.pt`.

## Adjacency modes at evaluation

Training is graph-agnostic; the adjacency is supplied per evaluation. The
modes used in the paper are built by `build_mode_list` in
`benchmarks/eval_graph2d/eval_graph2d_realcause.py`:

- `noanc` — all-zero PAM, no structural information
- `v3a` (`paper_anc`) — asserted $+1$ edges only
- `v3b` (`full`) — the same $+1$ edges with reverses set to $-1$

`v3ab_only` emits all three in one pass and is what the RealCause and
ComplexMech runs use. The **case studies use `case_family` instead**, which
builds the per-case DAG: `v3a`/`v3b` from `build_anc_v3*` hardcode a
confounder layout ($X\to T$, $X\to Y$) that is the wrong graph for the
mediator, spurious-covariate and front-door cases.

Note `v3_family` additionally contains `v3d`, whose $+1$ diagonal UWYK's PAM
validator rejects; graph2d does not validate and would run it, so the two
models would silently diverge. Use `v3ab_only`.
