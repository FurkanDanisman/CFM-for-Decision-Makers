# graph_eval — anc-info vs no-anc-info evaluation for a graph-conditioned checkpoint

Runs the anc / noanc PEHE and err_ATE evaluation for a graph-conditioned 2D-head
checkpoint on the RealCause suite (IHDP, ACIC, Lalonde CPS/PSID/PSID_bal).
Uses **UWYK's `PartialGraphConditionedInterventionalPFN`** architecture as the
model backbone and **UWYK's `propagate_ancestor_knowledge`** for the adjacency
matrix.

## What this outputs

Per realization, two adjacency modes × two mean estimators = 4 CATE variants:

| mode  | adjacency                                             | estimator | derivation                       |
|-------|--------------------------------------------------------|-----------|----------------------------------|
| anc   | T→Y = +1, X→T = +1, X→Y = +1, padded −1               | raw       | `E[Y] = Σ centres · p_marginal`  |
| anc   | same                                                   | em        | fixed-point Gaussian correction  |
| noanc | zeros (padded still −1)                                | raw       | same                             |
| noanc | same                                                   | em        | same                             |

Per-realization outputs land in `--out-dir` as `<DATASET>_r<idx>.npz` with keys
`pehe_raw_anc`, `err_raw_anc`, `pehe_em_anc`, `err_em_anc`, and the noanc
counterparts. The script also prints a rolling per-realization summary and a
final mean ± SE table.

## Requirements

External dependencies (must exist on disk somewhere):

- **UWYK** — clone `Graphs4CausalFoundationModels` (any branch that has
  `src/models/PartialGraphConditionedInterventionalPFN.py`). Set `UWYK` env var
  to its root.
- **CausalPFN** — clone `vdblm/CausalPFN`. Set `CAUSALPFN` env var to its root.
  We use it for the dataset loaders (`IHDPDataset`, `ACIC2016Dataset`,
  `RealCauseLalondeCPSDataset`, `RealCauseLalondePSIDDataset`).

Python deps: PyTorch ≥ 2.1, numpy, scipy. GPU recommended but not required.

Checkpoint: a graph2d 2D-head `.pt` saved by
`training_graph2d/train_graph_2d.py`. Example checkpoints are in
`../Required_checkpoints/graph2d_step_50000.pt` and `graph2d_step_58000.pt`.

## Running

Single dataset, one realization for a smoke test:

```bash
cd R-PFN
export UWYK=/path/to/uwyk
export CAUSALPFN=/path/to/causalpfn

python graph_eval/run.py \
    --ckpt Required_checkpoints/graph2d_step_50000.pt \
    --dataset IHDP \
    --out results_graph_eval/step_50000/IHDP \
    --max-realizations 1
```

Full sweep on all 5 datasets (typical usage — copies the eval sbatch's default settings):

```bash
CKPT=Required_checkpoints/graph2d_step_50000.pt
OUT_BASE=results_graph_eval/step_50000

for DS in IHDP ACIC CPS PSID PSID_bal; do
    python graph_eval/run.py \
        --ckpt "$CKPT" \
        --dataset $DS \
        --out $OUT_BASE/$DS \
        --max-context 1000
done

python graph_eval/aggregate.py --root $OUT_BASE
```

## Options

- `--ckpt PATH` (required) — graph2d checkpoint
- `--dataset {IHDP,ACIC,CPS,PSID,PSID_bal}` (required)
- `--out DIR` (required) — output directory for per-realization npz files
- `--max-context N` (default 1000) — cap the context set per realization
  (matches ArikReuter reproduce-realcause-results branch). CPS/PSID have
  ~15k rows per context so subsampling is essential for wall-clock.
- `--propagate {0,1}` (default 1) — apply `propagate_ancestor_knowledge`
  to the anc matrix so its shape matches what training saw
- `--max-realizations N` (default all) — cap number of realizations; useful
  for smoke tests
- `--anc-mode {full,ty_only,ty_antisym,all_variants}` (default `full`)  —
  probes different anc-matrix contents (see the file for exact semantics)

## Expected numbers

| dataset  | anc PEHE (target) | noanc PEHE (target) |
|----------|-------------------|---------------------|
| IHDP     | ≈ 4.3             | ≈ 4.5               |
| ACIC     | ≈ 2.7             | ≈ 2.7               |
| CPS      | ≈ 12.3k           | ≈ 12.3k             |
| PSID     | ≈ 21.4k           | ≈ 21.1k             |
| PSID_bal | ≈ 19.6k           | ≈ 18.7k             |

Absolute values depend on the checkpoint (step 50000 vs 58000 will differ)
but anc-vs-noanc structural pattern should be consistent.

## Troubleshooting

- **`ModuleNotFoundError: models.PartialGraphConditioned…`** →
  `UWYK` env var is wrong. Its src/ subdir must contain
  `models/PartialGraphConditionedInterventionalPFN.py`.
- **`ModuleNotFoundError: benchmarks`** → `CAUSALPFN` env var is wrong.
  Its `src/` subdir needs to expose the dataset classes.
- **`KeyError: 'edges'` while loading ckpt** → older checkpoint format;
  fine, the loader falls back to `make_edges(J)`.
- **Hangs on first forward** → try `--max-context 500` if RAM is tight.
