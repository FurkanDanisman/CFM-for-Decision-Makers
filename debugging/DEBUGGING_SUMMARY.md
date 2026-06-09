# Debugging Summary: Paired Potential Outcome Generation

## Problem
Running `generate_paired_samples.py` produced `CATE = Y_do1 - Y_do0 = 0` for samples 0, 2, and 4
after preprocessing, despite the underlying SCM having a real causal path T → Y.

---

## Red Herrings Eliminated

### 1. Preprocessing (not the cause)
`scale_to_neg1_pos1` + `clamp(-2, 2)` was suspected. Debug confirmed Y_do0_raw and Y_do1_raw
were already identical BEFORE preprocessing. Preprocessing was innocent.

### 2. Tensor aliasing (not the cause)
Suspected that `res0[target_node]` and `res1[target_node]` shared memory after the second
`intv_scm.propagate()` call. Checked `data_ptr()` — they were different tensors with different
addresses. Aliasing was not the issue.

### 3. Missing T→Y path (not the cause)
Suspected `exists_treatment_outcome_path()` returned a false positive. Verified with NetworkX:
the directed path T → 2 → Y existed in the DAG. Path logic was correct.

---

## Root Cause: Batch Normalization

**The bug**: When propagating ALL N_TEST=500 samples with the same treatment value (all T=0
OR all T=1 in separate calls), the MLP mechanisms in downstream nodes have BatchNorm layers.
BatchNorm computes `(x − mean(x)) / std(x)`. When T is constant across the entire batch:

```
z_i = f(other_parents_i) + W_T · T    (T same for all i)
BN([z_i]) = (z_i − mean(z_i)) / std(z_i)
           = (f(other_parents_i) − mean(f(other_parents_i))) / std(f(other_parents_i))
```

The constant T term **shifts the mean** but leaves the normalized output unchanged. Treatment
effect is completely cancelled. This held even for T=100 — exact zero CATE regardless of
treatment magnitude.

**How it was found**: `debug_cate3.py` added prints inside the propagation loop. Inspecting
`parents_feat_0 == parents_feat_1` (the inputs to the target node's mechanism) showed they
were identical despite different treatment values. Tracing one node up (intermediate node 2,
parent of target) revealed its output was constant across T=0/1/100.

---

## Fix: Doubled Mixed Batch

Instead of two separate propagations (all T=0, then all T=1), propagate a **single doubled
batch** of size 2×N_TEST:
- Rows 0..N_TEST-1: T=0
- Rows N_TEST..2×N_TEST-1: T=1
- All other noise (exo + endo) tiled: same ε_k for row k and row k+N_TEST

BatchNorm now sees a mixed distribution (half 0s, half 1s) and computes real statistics.
The coupling is preserved: row k and row k+N_TEST share the same ε_k, so
`Y_do0[k] = output[k]` and `Y_do1[k] = output[k+N_TEST]` are proper coupled potential outcomes.

**Implemented in**: `propagate_paired()` in `generate_paired_samples.py`

---

## Debug Scripts (in this folder)

| File | Purpose |
|------|---------|
| `debug_cate.py` | First trace: showed Y_do0_raw == Y_do1_raw before preprocessing |
| `debug_cate2.py` | Confirmed with different SCM config that CATE IS non-zero (batch norm was less active there) |
| `debug_cate3.py` | Definitive diagnosis: showed `parents_feat_0 == parents_feat_1` and T=100 gives same output |

---

## Final Verification

All 5 samples after fix:

| Sample | CATE mean | CATE std |
|--------|-----------|----------|
| 0 | 1.2564 | 0.3702 |
| 1 | 0.2993 | 0.4950 |
| 2 | 0.0041 | 0.1628 |
| 3 | −0.5820 | 0.7227 |
| 4 | −0.0004 | 0.5240 |

Samples 2 and 4 have near-zero mean CATE but non-zero std — valid SCMs where treatment has
weak average effect but heterogeneous individual effects. Not a bug.
