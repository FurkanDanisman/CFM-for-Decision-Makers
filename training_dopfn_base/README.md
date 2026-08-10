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

- `dopfn_backbone_head.py` — model wrapper skeleton with TODOs pointing
  to exactly which internal Do-PFN methods need to be intercepted.
- `train.py` — a copy of `training/train_cfm_dopfn.py` with the model
  construction swapped to use the new wrapper; the training loop, edge
  fitting, checkpointing, and cosine schedule are unchanged.
- `cluster/submit_train.sbatch` — SLURM script mirroring existing training
  submissions.

## What's NOT implemented (deliberately marked TODO)

- The Y-input adaptation (Option A). Requires looking at Do-PFN's
  `y_encoder` to know its input shape and reproducing its interface with a
  paired-outcome wrapper.
- The forward-pass integration for query rows. Do-PFN's TransformerModel
  distinguishes context vs. query rows internally; we need to make sure
  our decoder is applied to the query positions only.
- End-to-end smoke test on a tiny dataset. Should confirm the model
  produces a well-shaped `K² + 9 + 4` output per query before running any
  full training.

## Execution plan

1. Copy Do-PFN's `model/` folder locally as a reference (or install their
   package in dev mode).
2. Implement Option A (Y-input adaptation) in `dopfn_backbone_head.py`.
3. Smoke-test with a 5-step training run on a tiny prior.
4. If output shapes and gradients look right, launch head-only training
   for a few hours; evaluate the checkpoint on IHDP density-L2.
5. Depending on head-only results, either ship or unfreeze for full
   fine-tune.
