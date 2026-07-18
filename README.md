# R-PFN

Paired-outcome interventional PFN with 2D BarDistribution head and MALC-based
CATE density estimation.

## Layout

```
R-PFN/
├── models/         # InterventionalPFN — transformer with 2D BarDist head
├── losses/         # BarDistribution2D loss + MALC bridge
├── MALC/           # 2D log-concave mixture density fitter
│   └── Optimal_Transport/
│                   # 1D W2 barycenter for population ATE aggregation
├── checkpoints/    # Trained model (step_50000_final.pt)
├── training/       # Training pipeline (see training/README.md)
│   ├── data/           #   PairedInterventionalDataset (SCM streaming)
│   ├── docs/           #   design notes (2D head, SCM prior)
│   └── cluster/        #   SLURM submit scripts
└── benchmarks/     # Table 3 reproduction pipeline (see benchmarks/README.md)
```

## Reproducing paper Table 3

See [`benchmarks/README.md`](benchmarks/README.md). One-liner overview:

```bash
bash benchmarks/cluster/deploy_local.sh /tmp/rpfn_bench_kit
# rsync payload to your SLURM cluster, then:
sbatch benchmarks/cluster/submit.sbatch
python benchmarks/aggregate.py --results ./results
```

## Method summary

Given `(X_train, t_train, y_train, X_test)`, the model outputs — per test
query — a joint distribution `p(Y_do0, Y_do1)` on a 100 × 100 grid. From that
we derive six CATE estimates:

| variant | description |
|---|---|
| `ours_mean`          | E[τ] from raw p_mat marginals — no MALC smoothing |
| `ours_malc_mean`     | E[τ] under MALC-smoothed p(τ) |
| `ours_malc_mean_msk` | same, after diagonal masking (τ=0 attractor removed) |
| `ours_malc_mode`     | argmax MALC-smoothed p(τ) |
| `ours_malc_mode_msk` | argmax MALC-smoothed masked p(τ) |
| `ours_ot_mode`       | argmax of the W2 barycenter of masked per-query densities (population ATE only) |

See `benchmarks/plots/plot_mask_example.py` for an illustration of the
diagonal masking mechanism.

## Reproducing training

The shipped checkpoint is reproducible end-to-end. See
[`training/README.md`](training/README.md) — it covers the architecture,
the SCM prior, the exact differences vs UWYK (we change only the output
head; backbone / optimizer / schedule / SCM sampler are UWYK Appendix G
verbatim), and the SLURM sequence used on Trillium.
