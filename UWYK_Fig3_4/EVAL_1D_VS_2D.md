# 1D-vs-2D head comparison on the ComplexMech PEHE benchmark

Extends the `realcause_eval/Table1` comparison (dopfn_native vs dopfn_bb,
uwyk1d vs graph2d, cpfn1d vs cpfn2d) onto UWYK's Figure-4 ComplexMech prior,
where the ground-truth CATE is known exactly.

## Why ComplexMech only, and why the split

From `distributions/` and `learnable_heterogeneity.md`:

* **LinGaus (Fig 3) is excluded.** Its within-dataset CATE is constant
  (sd 4e-8), so no method can beat a constant predictor and PEHE there is an
  ATE error. Measured learnable heterogeneity: 0.000 at every node count.
* **n=2 is excluded.** A 2-node SCM is T and Y only — no covariates.
* **ComplexMech n=5..50 carries real signal**, learnable fraction 0.78 (n=5)
  down to 0.14 (n=50).

12–27% of ComplexMech queries have an **exactly zero** treatment effect: tree
and saturating mechanisms often do not respond to flipping T. Those ask "can the
model report no effect?", a different skill from grading a non-zero effect, so
they are scored separately:

    CMECH_n<N>_nonzero    true tau != 0
    CMECH_n<N>_zero       true tau == 0 exactly

The split is bimodal at the dataset level — at n=5, 72/100 realizations contain
no zero-effect query at all — so each subset exposes only the realizations where
it is non-empty. The subsets are disjoint, so pooled PEHE is exact:

    pehe_all^2 = (n0*pehe_zero^2 + n1*pehe_nonzero^2) / (n0 + n1)

## Node counts

`--nodes 5 20 30 40 50`. UWYK ships YAMLs for {2,5,10,20,35,50} only, so **30
and 40 are synthesized**: `resolve_config` clones the nearest shipped config and
overrides `num_nodes`. Diffing UWYK's per-n YAMLs shows exactly two numeric
differences — `scm_config.num_nodes` and `model_config.num_features`
(= num_nodes - 3, a transformer width we never read) — so the prior is otherwise
untouched. Runs log `[synth]` when this happens.

## Step 1 — generate the data (CPU, ~3 min, no GPU)

```bash
export UWYK_SRC=$DEPLOY_ROOT/external/uwyk/src
export UWYK_ROOT=$DEPLOY_ROOT/external/uwyk

python $REPO/UWYK_Fig3_4/generate_pehe_benchmark.py \
    --prior complexmech --nodes 5 20 30 40 50 --regimes path_TY \
    --hide-fractions 0.0 --n-realizations 100 \
    --out-dir $REPO/UWYK_Fig3_4/data
```

Sanity-check first with `--nodes 5 --self-test` (asserts bitwise regeneration).

## Step 2 — run the array

```bash
cd $DEPLOY_ROOT
OUT_ROOT=$SCRATCH/cmech_1d2d \
  sbatch $REPO/benchmarks/cluster/submit_cmech_pehe_1d_vs_2d.sbatch
```

60 tasks = 6 models x 5 node counts x 2 subsets. Smoke-test one task first:

```bash
OUT_ROOT=$SCRATCH/cmech_smoke MAX_REAL=1 \
  sbatch --array=30 $REPO/benchmarks/cluster/submit_cmech_pehe_1d_vs_2d.sbatch
```

First task of each model: **0** `dopfn_native`, **10** `dopfn_bb`,
**20** `uwyk1d`, **30** `graph2d`, **40** `cpfn1d`, **50** `cpfn2d`.
Within a model, `(id%10)/2` indexes nodes (5,20,30,40,50) and `id%2` picks the
subset (0 = nonzero, 1 = zero). Smoke-test one task per model — the six
harnesses differ enough that one passing does not imply the rest do:

```bash
OUT_ROOT=$SCRATCH/cmech_smoke MAX_REAL=1 \
  sbatch --array=0,10,20,30,40,50 \
  $REPO/benchmarks/cluster/submit_cmech_pehe_1d_vs_2d.sbatch
```

Checkpoints default to the Table 1 paths and are overridable via
`CKPT_CPFN1D`, `CKPT_CPFN2D`, `CKPT_GRAPH2D`, `CKPT_DOPFN_BB`, `UWYK_CKPT_DIR`.

## Step 3 — aggregate

```bash
python $REPO/benchmarks/aggregate_cmech_1d_vs_2d.py --root $SCRATCH/cmech_1d2d
```

`delta = 2D - 1D`, so **negative means the 2D head wins**. `n_2d_better` counts
datasets where 2D beat 1D, which is more robust than the mean given the heavy
tail.

For `uwyk1d` vs `graph2d` the aggregator reads the **`noanc`** tag by default:
this comparison is about the head, so both sides should get the same (absent)
graph information. `--anc-tag v3b` compares the ancestor-conditioned variants.

## How it plugs in

`benchmarks/uwyk_fig34_dataset.py` exposes each cell with the IHDPDataset
interface the eval harnesses already expect (`ds.n_tables`, `ds[r] ->
(cate_slice, None)` with `X_train/t_train/y_train/X_test/true_cate`). Zero-padding
columns are trimmed, since the harnesses standardize X and would divide by ~0.

Dispatch was added to six entry points, each guarded so RealCause names are
untouched:

| file | serves |
|---|---|
| `benchmarks/eval_graph2d/eval_graph2d_realcause.py` | graph2d **and** uwyk1d (which execs it as module `H`) |
| `benchmarks/eval_causalpfn2d/eval_causalpfn_v0_realcause.py` | cpfn1d |
| `benchmarks/eval_causalpfn2d/eval_cpfn2d_realcause.py` | cpfn2d |
| `benchmarks/l2_ihdp/eval_dopfn_bb_raw.py` | dopfn_bb |
| `benchmarks/eval_scm_case_studies/eval_native_dopfn.py` | dopfn_native (cluster path) |
| `realcause_eval/Table1/do_pfn.py` | dopfn_native (Table1 wrapper) |

## Units caveat

Y and tau are on the generator's [-1, 1] target scale (UWYK's own
`target_negative_one_one_scaling`). PEHE is comparable across models and node
counts within this benchmark, but **not** to RealCause PEHE, which is in each
dataset's own outcome units.
