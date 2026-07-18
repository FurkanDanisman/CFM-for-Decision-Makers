# Training

How the shipped `checkpoints/step_50000_final.pt` was produced, and how to
reproduce it end-to-end.

## Layout

```
training/
├── README.md              (this file)
├── train_cfm.py           production trainer — reproduces the shipped ckpt
├── train_cfm_small.py     debug / laptop trainer (smaller model, shorter run)
├── data/
│   └── PairedInterventionalDataset.py   streaming dataset (SCM → paired outcomes)
├── docs/
│   ├── CFM_2D_BarDistribution_Design.md
│   └── prior_summary.md
└── cluster/
    ├── submit_train.sh    SLURM script for a single H100, full run
    └── submit_train_debug.sh   short SLURM job with the production config
```

## What is being trained

An `InterventionalPFN` (`models/InterventionalPFN.py`) — a transformer that
consumes an observational context `(X_obs, T_obs, Y_obs)` plus a set of test
covariates `X_intv`, and predicts a **joint** distribution over the pair of
potential outcomes `(Y_do(0), Y_do(1))` for each test query.

The model architecture is `PriorFittedNetwork` (UWYK, 2024) with a **2D
BarDistribution head** we designed (`losses/BarDistribution2D.py`). The head
outputs the joint `p(Y_do0, Y_do1)` on a `J × J` grid (default `J = 100`)
plus 9 outer region weights and 4 outer-region tail scales, for a total
output dimension of `J² + 13 = 10,013`.

Training minimizes the negative log-likelihood of the ground-truth
`(Y_do0, Y_do1)` under the predicted joint density.

## What differs from UWYK

UWYK's original model predicts a **marginal** `p(Y | X, do(T))` per treatment
arm. Ours predicts the **joint** `p(Y_do0, Y_do1 | X)` — the paired-outcome
distribution — which is what individual-level CATE questions actually need
(and what enables our OT-based ATE aggregation, per-query MALC smoothing,
and diagonal-masking analysis).

Concrete differences vs UWYK's `PriorFittedNetwork`:

| aspect | UWYK | Ours |
|---|---|---|
| output head | 1D BarDistribution (`J = 100` bins per arm) | **2D BarDistribution** (`J × J` grid + 9 regions + 4 tails) |
| head output dim | 100 | **10,013** |
| training loss | 1D `neg_log_prob` on `Y \| do(T)` marginals | **2D `neg_log_prob_2d`** on `(Y_do0, Y_do1)` paired samples |
| data yield per SCM | `(X, T, Y)` triples | **paired `(X, T_obs, Y_obs, Y_do0, Y_do1)`** via `_propagate_paired` |
| backbone | UWYK's `PriorFittedNetwork` | **verbatim UWYK backbone** — we changed only the head |
| optimizer / schedule / LR / batch / steps | UWYK Appendix G defaults | **identical to UWYK Appendix G** |
| SCM prior | UWYK's `SCMSampler` + `BinarizingMechanism` | **identical — we import their sampler unchanged** |

That last row is important: we do **not** modify UWYK's SCM prior, mechanism
sampling, or transformer backbone. Only the output head and the loss change.
Everything else — including the exact `Adam(lr=1e-4, wd=1e-5)`, cosine
schedule with 10% linear warmup, effective batch size 32, N_context=1000,
50,000 training steps, bf16 mixed precision, gradient clipping at 1.0 — is
UWYK Appendix G verbatim.

The point of that alignment: any difference in downstream benchmark
performance between OURS and UWYK is attributable to the head, not to
optimizer or data-scale confounders.

## Training data

Generated on-the-fly by `data/PairedInterventionalDataset.py` (a streaming
`torch.utils.data.Dataset`). Each __getitem__ call runs one iteration of:

1. `SCMSampler.sample(seed=…)` — draw a fresh SCM from UWYK's prior
2. Pick treatment node + outcome node such that a directed path exists
3. Binarize the treatment node via UWYK's `BinarizingMechanism`
4. Sample `N=1000` observational rows through the SCM → `(X_obs, T_obs, Y_obs)`
5. `_propagate_paired` — for a fresh set of `M ~ U[1, 500]` test covariates,
   propagate BOTH `do(T=0)` and `do(T=1)` interventions through the SCM to
   obtain the ground-truth pair `(Y_do0, Y_do1)`
6. Standardize features, scale Y to `[-1, 1]` jointly across
   `(Y_obs, Y_do0, Y_do1)`, pad features to `NUM_FEATURES=50`

Rejection loop rejects samples with target variance below threshold or
whose targets contain fewer than 5 unique values (mirrors UWYK's own
`_sample_passes_thresholds` guard).

**No pre-generated dataset on disk.** The prior draws are cheap enough
(~0.2 s per SCM at N=1000) that streaming is faster than reading a corpus.

## The shipped checkpoint

`checkpoints/step_50000_final.pt` was trained by running
`training/cluster/submit_train.sh` three times in sequence on Trillium
(H100 × 1). Wallclock ~53 hours total. The 20-hour SLURM chunks resume
automatically via `RESUME=1` — each invocation picks up at the latest
saved checkpoint.

Configuration frozen into the checkpoint (in `ckpt['config']`):

```
J             = 100
d_model       = 256
depth         = 8
heads         = 8
hidden_mult   = 4
num_features  = 50
edges         = 101-bin edges over [-1, 1]
```

Training loss curve is available in the SLURM logs.

## Reproducing the training run

### 1. Environment

```bash
# UWYK source (needed by SCMSampler + BinarizingMechanism at data time)
git clone --depth 1 https://github.com/ArikReuter/Graphs4CausalFoundationModels.git \
    /path/to/g4cfm
export UWYK_SRC=/path/to/g4cfm/src

# R-PFN venv
cd /path/to/R-PFN
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch~=2.11.0 numpy~=2.4.0 scipy~=1.15.0 scikit-learn~=1.8.0 \
            pandas~=3.0.0 matplotlib~=3.10.0 PyYAML~=6.0.0 einops~=0.8.0 \
            tqdm~=4.68.0
```

### 2. Local smoke test (CPU or MPS, ~2 min)

Runs 100 steps of the small variant to verify imports + streaming pipeline.

```bash
cd /path/to/R-PFN
N_STEPS=100 CHECKPOINT_EVERY=100 CHECKPOINT_DIR=./checkpoints_smoke \
    UWYK_SRC=/path/to/g4cfm/src \
    python training/train_cfm_small.py
```

### 3. Full run on cluster (H100 × 1, ~53 hours in three 20h chunks)

```bash
cd /path/to/R-PFN
export UWYK_SRC=/path/to/g4cfm/src

FIRST=$(sbatch --parsable training/cluster/submit_train.sh)
for i in 2 3; do
    FIRST=$(sbatch --parsable --dependency=afterany:$FIRST training/cluster/submit_train.sh)
done
```

Each invocation:
- Reads the latest checkpoint from `checkpoints/`
- Trains until the 20-hour SLURM limit
- Saves checkpoint (both step-numbered and `latest.pt`)
- Exits cleanly

After all three chunks the final checkpoint is
`checkpoints/step_50000_final.pt` — identical to the one shipped.

## Environment knobs

Every field of the config is overridable from the shell. Common ones:

```
J                  head grid resolution        default 100
D_MODEL, DEPTH, HEADS, HIDDEN_MULT, DROPOUT     model arch  (UWYK App G)
LR, WEIGHT_DECAY, WARMUP_FRAC, MIN_LR_RATIO     optimizer
N_STEPS, MICROBATCH, GRAD_ACCUM                 iteration budget
N_CONTEXT_TRAIN, N_QUERY_TRAIN                  per-task shape
STREAM_WORKERS, STREAM_SEED, STREAM_WARMUP      data pipeline
USE_BF16, USE_CHECKPOINT                        precision + memory
CHECKPOINT_DIR, CHECKPOINT_EVERY, RESUME        persistence
```

See `train_cfm.py` header for the exhaustive list.

## Debugging notes preserved for posterity

`train_cfm.py` contains a `LOSS_WARN_THRESH` mechanism that dumps the
offending batch's target and logit ranges whenever the training loss
crosses a threshold (default 1e3) or goes non-finite. It's a cheap
early-warning that ties data-side outliers to loss spikes.

`training/data/PairedInterventionalDataset.py` has a per-worker
`_NoneRefcountTracker` and breadcrumb file — kept from a debugging session
where a UWYK-side C extension leaked `None` refcounts across the SCM draw
loop and eventually crashed the worker (`none_dealloc` abort). Both stay
in place because the underlying leak is deep in UWYK's SCM machinery and
the tracker is cheap enough to leave enabled by default (`DATA_VERBOSE=0`).
Set `DATA_VERBOSE=1` if you want per-sample logging.
