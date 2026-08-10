# Training an R-PFN model on Do-PFN's TabPFN backbone

## Why this exists

The existing `training/train_cfm_dopfn.py` trains an R-PFN checkpoint on
Do-PFN's SCM prior, but keeps UWYK's `InterventionalPFN` architecture.
That architecture has a hard-coded `num_features` cap in its row-MLP
(`nn.Linear(num_features + 1, d_model)` in
`models/InterventionalPFN.py:188`). Consequences on RealCause:

- On IHDP (d=25) the fn=10 checkpoint truncates to first 10 features
  (loses 60% of covariates).
- On ACIC (d=58) fn=10 truncates to first 10 (loses 83%).
- Do-PFN (built on TabPFN with `use_per_feature_transformer=True`,
  `max_num_features_ = -1`) has no cap — it processes each feature as a
  separate attention token and generalises to any number of features at
  inference.

For a genuinely "same-base as Do-PFN, only head changed" story, R-PFN
needs to use Do-PFN's TabPFN backbone (per-feature attention, no cap)
with our 2D joint-density head. This folder scaffolds that retraining.

## Plan

```
training_dopfn_base/
├── README.md                    # this file
├── __init__.py
├── dopfn_backbone_head.py       # wraps Do-PFN's TransformerModel + swaps decoder
├── train.py                     # training loop (mirrors train_cfm_dopfn.py)
└── cluster/
    └── submit_train.sbatch      # SLURM script
```

## Architecture — what needs to change

Do-PFN's underlying transformer is defined in
`external/dopfn/model/transformer.py::TransformerModel`. Its high-level
structure:

```
encoder (per-feature)  →  transformer_encoder (attention layers)  →  decoder_dict['standard'] (BarDistribution 1D logits)
```

To make it a joint-head model:

1. **Load** Do-PFN's checkpoint (`artifacts/dopfn_model.pkl` plus
   `model_submitit_0ccc_id_171b69db_epoch_-1.cpkt`) with its usual loader.
2. **Replace** `decoder_dict['standard']` with a new 2D decoder that
   outputs `K² + 9 + 4` values per query (matches our existing
   `losses/BarDistribution2D.total_params(K)` count).
3. **Two-input encoding for Y**: Do-PFN's encoder currently expects one
   scalar Y per context example. Paired training feeds `(Y_do0, Y_do1)` per
   context example. The cleanest fix is to concatenate `(T, Y_do0, Y_do1)`
   into the Y-input dim (or add a separate embedding for the second
   outcome). See ARCHITECTURE NOTES below.
4. **Training**: two options for parameter-update scope:
   - **Head-only**: freeze the pre-trained transformer, train only the new
     2D decoder + new Y-embedding for the second outcome. Cheap
     (~few hours). Preserves Do-PFN's learned attention. Feasible if
     Do-PFN's transformer embeddings already carry enough information for
     the joint task.
   - **Full fine-tune**: unfreeze everything, retrain end-to-end. More
     expensive (~50-100 GPU-hours) but recovers the full capacity of the
     model for the joint prediction task.

## ARCHITECTURE NOTES — the (T, Y_do0, Y_do1) input problem

Do-PFN was trained on `(X, Y)` context: one outcome per training row. R-PFN
needs two outcomes per row `(Y_do0, Y_do1)` (paired-outcome training).

Options:

**A. Concatenate along Y-embed dim.** Encode `Y_do0` and `Y_do1` with the
same y_encoder into two embeddings, concatenate them, then a linear
projection back to `d_model`. Simple; keeps the transformer input
unchanged in sequence length.

**B. Two separate y_encoders + sum.** One encoder for Y_do0, another for
Y_do1, sum their embeddings. Symmetric under arm swap.

**C. Treatment-conditioned single Y.** Encode T as an embedding, feed
`Y_factual` (the observed one) with T-conditioning through the existing
y_encoder, and treat the missing arm as a masked token that the
transformer fills in. Closest to Do-PFN's original semantics.

**Recommendation**: Option A is the smallest change with maximal
information preservation. Fits into a wrapper class in one afternoon.

## Data loader

Reuse `training/data/PairedDoPFNDataset.make_dopfn_streaming_loader` —
already emits paired outcomes from Do-PFN's SCM prior. Only change is that
our new model consumes them via the adapted encoder (Option A above)
instead of dropping one arm.

## Loss

Reuse `losses/BarDistribution2D.neg_log_prob_2d` unchanged. This is
head-agnostic — it takes the raw decoder output and paired labels.

## Compute estimate

- **Head-only fine-tune** at Do-PFN's compute regime (num_features up to
  a few hundred, d_model=~512, batch size ~64 queries):
  ~6-12 GPU-hours on a single H100.
- **Full fine-tune**: ~50-100 GPU-hours, matches Do-PFN's original
  training budget.

Start with head-only for a quick check that the architecture wiring is
correct; then decide whether full fine-tune is needed based on the
head-only checkpoint's density-L2 numbers.

## What's implemented in this scaffold

- `dopfn_backbone_head.py` — **fully implemented** model wrapper:
  - Loads Do-PFN's TransformerModel via its own `load_model` helper
    (mirrors `scripts/transformer_prediction_interface/model_builder.py`).
  - Swaps `decoder_dict['standard']` with a 2D decoder head that emits
    `K² + 9 + 4` values (matches `losses/BarDistribution2D.total_params(K)`).
  - Swaps `y_encoder` with a `PairedYEncoder` that consumes `(T, Y_do0,
    Y_do1)` per context row and warm-starts the Y_factual slot from
    DoPFN's original pretrained encoder.
  - Adapter forward that shapes our batch-first
    `(X_context, T_context, Y_context_pair, X_query)` into DoPFN's
    sequence-first `(train_x, y_src, test_x)` and returns
    `{'predictions': (B, M, K² + 9 + 4)}`.
  - Head-only training freezes everything under `backbone.*` except the
    new head (`decoder_dict.standard.*`) and the paired-Y encoder
    (`y_encoder.*`).
- `train.py` — training loop mirrors `training/train_cfm_dopfn.py`; only
  the model construction changes (uses `DoPFNBackboneWith2DHead` and
  packs `Y_context_pair = stack([Y_do0, Y_do1])` before the forward call).
- `cluster/submit_train.sh` — SLURM script mirroring the existing DoPFN
  submitter with a new `HEAD_ONLY` toggle (default 1).

## What still needs cluster-side verification

The scaffold *should* run end-to-end but has NOT been executed yet. A
short smoke test is essential before launching a long run:

```bash
cd /scratch/furkanbd/rpfn_bench_kit
DOPFN_ROOT=/scratch/furkanbd/rpfn_bench_kit/external/dopfn \
DOPFN_SRC=/scratch/furkanbd/rpfn_bench_kit/external/dopfn \
N_STEPS=20 MICROBATCH=2 GRAD_ACCUM=1 \
NUM_FEATURES=5 N_TRAIN=200 N_TEST=100 STREAM_WORKERS=2 \
CHECKPOINT_DIR=/scratch/furkanbd/rpfn_bench_kit/checkpoints_dopfn_backbone_smoke \
/scratch/furkanbd/rpfn_bench_kit/venv/bin/python \
  /scratch/furkanbd/rpfn_bench_kit/R-PFN/training_dopfn_base/train.py
```

Expected: 20 training steps with a decreasing NLL, ending in a saved
checkpoint. If it errors, the most likely culprits are:

- **Encoder input shape**: DoPFN's `encoder` may not accept `test_x`
  passed as a separate argument in newer versions. If so, the fix is a
  one-liner: concat `train_x` and `test_x` before the call, following the
  pattern in `TransformerModel.forward()` (lines 253-259 of DoPFN's
  `model/transformer.py`).
- **`d_model` mismatch**: the pretrained Y-encoder's output dim might
  differ from `backbone.ninp`. If so, wrap it in an extra projection.
- **`y_encoder` being called with `single_eval_pos` kwarg**: `PairedYEncoder`
  ignores it (query rows are zero-padded anyway). If DoPFN's `_forward`
  calls it in a way that broadcasts oddly, replace the `y_src` construction
  in `DoPFNBackboneWith2DHead.forward` to only emit context rows.

After the smoke test passes, launch full training (~6-12h for head-only,
~50-100h for full fine-tune).

## Execution plan

1. Copy Do-PFN's `model/` folder locally as a reference (or install their
   package in dev mode).
2. Implement Option A (Y-input adaptation) in `dopfn_backbone_head.py`.
3. Smoke-test with a 5-step training run on a tiny prior.
4. If output shapes and gradients look right, launch head-only training
   for a few hours; evaluate the checkpoint on IHDP density-L2.
5. Depending on head-only results, either ship or unfreeze for full
   fine-tune.
