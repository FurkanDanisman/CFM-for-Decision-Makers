# Training Pipeline — Active Problem Log

**Status as of 2026-06-10:** Training crashes on Trillium with
`Fatal Python error: none_dealloc` inside the data pipeline after
~3,000 – 16,000 SCM generations (depending on configuration).
Local laptop runs the same code for 20,000+ samples with zero errors.
Cause not yet localized to a specific library or function.

---

## What we are trying to do

Train a Causal Foundation Model with a **2D paired-outcome head**. The
architecture is identical to UWYK's `InterventionalPFN` (verbatim
`TwoWayBlock` / SwiGLU / role embeddings / sinusoidal feature PE) except:

- T is binary {0, 1}
- The query has no T (learned `null_t_intv` token fills the slot)
- The output is a **2D BarDistribution** over `(Y_do(0), Y_do(1))`
  instead of a 1D head over `Y_intv`
  - Output dim = J² + 9 + 4 = 10,013 at J = 100
  - 9 region mixture weights (inner / 4 mixed / 4 corner)
  - 4 learned tail scales (`σ_L0, σ_R0, σ_L1, σ_R1`)
  - Correlation ρ derived from `p_mat`, not learned

Training data comes from a **streaming `PairedInterventionalDataset`** that
samples fresh SCMs and produces `(Y_do(0), Y_do(1))` from the **same
exogenous noise draw** (paired potential outcomes).

Target wall-clock: ~45 hours on a single H100, modeled after UWYK's
Appendix G (50 K steps × effective batch 32, Adam @ 1e-4, cosine + 10 %
warmup, bf16 + activation checkpointing).

---

## The bug

```
Fatal Python error: none_dealloc: deallocating None: bug likely caused
by a refcount error in a C extension
```

The Python `None` singleton's refcount is being decremented incorrectly
by some C-extension code. Corruption accumulates over many SCM
generations and eventually the interpreter aborts.

### Where the crash is observed

| Environment | Configuration | Samples before crash |
|---|---|---|
| macOS laptop, mainline PyTorch, 1 process, sequential | exo layout fix in place | **20,000+ no crash** |
| Trillium login → compute, CC `torch-2.12.0+computecanada`, 1 process | `STREAM_WORKERS=0` | ~3,000 |
| Trillium login → compute, CC `torch-2.12.0+computecanada`, 8 forked workers | `STREAM_WORKERS=8` | ~16,000 |

### What this evidence proves

1. **Our pipeline code is correct in pure Python terms.** Same code, same
   inputs, ran 20K iterations on macOS with no error.
2. **The bug is environment-specific to Trillium**, NOT a code logic bug.
3. **It is not primarily a multiprocessing bug.** Single-process on the
   cluster also crashes.

### What this evidence does NOT prove

- We do not know whether the offending library is CC's custom
  `torch-2.12.0+computecanada` build, numpy, libc, the Linux malloc, or
  something else.
- Speculation about which library is to blame, without diagnostic data
  from the crash site, is not allowed (see "Forbidden patterns").

### Diagnostic in place

- `faulthandler.enable()` is registered at the top of `train_cfm.py`
  and inside `_worker_init_fn` so each forked worker also has it.
- On the next `none_dealloc` (or any fatal signal), the `.err` file
  will contain the full **C-level** stack with library and function
  names. This is what we need to localize the bug.

---

## Summary of the path here (chronological)

1. Built the 2D BarDistribution and `InterventionalPFN` (UWYK body
   verbatim, T_intv slot filled with learned null token).
2. Local smoke + medium runs: all clean.
3. Scaled-up cluster run with full UWYK Appendix G config: hit OOM at
   `MICROBATCH=8`. Resolved by halving `MICROBATCH` to 4 and adding
   activation checkpointing — fits comfortably in 80 GB.
4. Memory-fixed scaled-up run: clean 500 steps in 27 min, projected
   45 h for 50 K steps. ✅
5. Production chained run launched: crashed at ~32 min in with
   `none_dealloc` in `_propagate_paired`.
6. Diagnosed dict-vs-vec memory layout mismatch in `_propagate_paired`.
   Fixed to match UWYK's `sample_exogenous` layout (dict entries are
   views into the buffer). Local soak: 500 samples no error.
7. Resubmitted production: crashed at step 500 (16K samples) with same
   `none_dealloc`. The fix helped (5× more samples before crash) but
   did not fully resolve.
8. Local 20K-sample soak with the fix: **0 errors**. This proved the
   bug is environment-specific, not code-specific.
9. Added `faulthandler` for actual C-level diagnostics on the next
   crash. (Current state.)

---

## What is and is not allowed

### ✅ Allowed work

- Reading UWYK / PyTorch source to understand library invariants.
- Pure-diagnostic changes (faulthandler, logging, gc.collect) that do
  not alter pipeline semantics.
- Targeted fixes that flow from concrete diagnostic data (e.g., the
  C-level stack from faulthandler).
- Library version swaps if and only if there is direct evidence the
  current library is the culprit.

### ❌ Forbidden patterns (do not propose, do not implement)

These were tried, considered, or hypothesized earlier and explicitly
rejected. Do not raise them again:

1. **Pre-generating a corpus of SCMs to disk** ("disk corpus" /
   `generate_corpus.py` / `outputs_corpus/`). Even as a "workaround
   that gets us moving." The model must train on streaming SCMs,
   period.
2. **Switching to UWYK's 1D BarDistribution head** or otherwise
   abandoning the 2D paired-outcome design.
3. **`STREAM_WORKERS=0` as a normal training setting.** Acceptable as a
   one-shot diagnostic, never as production config (31 s/step is
   unworkable).
4. **Any change framed as "let's just try X and see if it works."**
   Every code or environment change must be justified by a clear
   hypothesis grounded in evidence, and must be small enough to roll
   back cleanly.
5. **Declaring a fix verified based on a small or non-representative
   local test.** The minimum acceptable verification standard for a
   data-pipeline fix is now **≥ 20 K samples in the same configuration
   that exhibited the bug**, plus the local test of equal duration on
   laptop.
6. **Suggesting library swaps (torch, numpy, etc.) without prior
   diagnostic data** showing the swap addresses an identified cause.
7. **Making any code change without explicit user approval.** Show the
   diff, explain the reasoning, get sign-off, then push.

---

## Current commit state

- Last useful production commit: `9568bbe` — adds `faulthandler` (no
  behavior change).
- Production training config (in `submit_train.sh`):
  - 3 × 20 h SLURM chunks, auto-resume via `RESUME=1`
  - `MICROBATCH=4`, `GRAD_ACCUM=8` (effective batch 32)
  - bf16 autocast, activation checkpointing on
  - 8 streaming DataLoader workers
  - 50,000 steps total

---

## Immediate next step

Restore the original Trillium environment (CC `torch-2.12.0+computecanada`,
since the recent install churn replaced it), then submit
`submit_scaledup.sh` (500 steps, ~30 min). With faulthandler in place,
the next `none_dealloc` will yield a real C-level stack trace. That
trace dictates what we do next. No speculation before then.
