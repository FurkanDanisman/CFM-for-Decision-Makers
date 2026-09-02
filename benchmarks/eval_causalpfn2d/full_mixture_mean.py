"""E[Y0] / E[Y1] under the full 9-region mixture density.

The 2D BarDistribution head emits, per query, J² inner bin logits +
9 region weights + 4 tail scales (softplus-transformed, then scaled by
δ = (hi - lo) / J). See losses/BarDistribution2D.py for the loss-side
formulation; here we compute the marginal MEAN of Y0 (and Y1) under
that same 9-region mixture, in the standardised Y space.

Callers still need to un-standardise back to raw Y (multiply by
per-task y_scale, then add y_shift).

Nine regions (indices match losses/BarDistribution2D.R_*):
    0 inner     y0 ∈ [lo,hi], y1 ∈ [lo,hi]        f = p_mat / δ²
    1 L0        y0 <  lo,     y1 ∈ [lo,hi]        f = p_L(y0;lo,σL0) · f_bdry(y1 | j0=0)
    2 R0        y0 >  hi,     y1 ∈ [lo,hi]        f = p_R(y0;hi,σR0) · f_bdry(y1 | j0=J-1)
    3 L1        y0 ∈ [lo,hi], y1 <  lo            f = f_bdry(y0 | j1=0)   · p_L(y1;lo,σL1)
    4 R1        y0 ∈ [lo,hi], y1 >  hi            f = f_bdry(y0 | j1=J-1) · p_R(y1;hi,σR1)
    5 L0-L1     both < lo                          f = truncated bivariate normal, corner (-,-)
    6 L0-R1     y0 < lo, y1 > hi                  f = truncated bivariate normal, corner (-,+)
    7 R0-L1     y0 > hi, y1 < lo                  f = truncated bivariate normal, corner (+,-)
    8 R0-R1     both > hi                          f = truncated bivariate normal, corner (+,+)

Notation matches the appendix in Draft/draft.tex.

For each region we compute the marginal mean contribution to E[Y0] and E[Y1]:

Inner:  Σ_j0 marg0[j0] · center[j0]      (marg0[j0] = Σ_j1 p_mat[j0,j1])
Edge L0:  y0 mean = lo − σL0 · √(2/π)    (half-Gaussian on left of lo)
Edge R0:  y0 mean = hi + σR0 · √(2/π)
Edge L1/R1: y0 from boundary conditional  (Σ_j0 col[j0]·center[j0] / col.sum())
Corner:  APPROXIMATION — treat corner y0 mean as the tail mean along y0
         axis, ignoring ρ. E[y0 | corner side s0] = c0 ± σ_{s0} · √(2/π).
         Exact under ρ=0 (independent axes); introduces small O(ρ²) bias
         otherwise. Since corner regions typically carry <1% total weight,
         the induced bias in E[Y0] is negligible in practice.

Return: (E_y0_std, E_y1_std) both shape (N_q,)
"""
from __future__ import annotations
import math
import numpy as np


SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
SOFTPLUS_FLOOR = 1e-3   # must match losses/BarDistribution2D.SOFTPLUS_FLOOR


def _softplus(x):
    # numerically stable softplus, works on numpy arrays
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


def unpack_2d_head(logits_np, J):
    """Given raw head output (shape [..., J² + 9 + 4]), return the components.

    Returns dict:
      p_mat   (..., J, J)  — inner softmax over J² bins
      w_reg   (..., 9)     — region-weight softmax
      sL0, sR0, sL1, sR1  (...)  — tail scales in the SAME units as the edges,
                                    already δ-scaled and floored
    """
    JJ = J * J
    inner_logits = logits_np[..., :JJ]
    reg_logits   = logits_np[..., JJ:JJ + 9]
    tail_raw     = logits_np[..., JJ + 9: JJ + 9 + 4]

    # softmax over inner
    m = inner_logits.max(axis=-1, keepdims=True)
    p_mat = np.exp(inner_logits - m)
    p_mat = p_mat / p_mat.sum(axis=-1, keepdims=True)
    p_mat = p_mat.reshape(*logits_np.shape[:-1], J, J)

    # softmax over regions
    m = reg_logits.max(axis=-1, keepdims=True)
    w = np.exp(reg_logits - m)
    w = w / w.sum(axis=-1, keepdims=True)

    return {'p_mat': p_mat, 'w_reg': w, 'tail_raw': tail_raw}


def full_mixture_mean(logits_np, J, edges_np):
    """Return E[Y0_std], E[Y1_std] per query under the 9-region mixture.

    logits_np: (N_q, J² + 9 + 4)  — model head output for each query
    J: inner-grid side length
    edges_np: (J+1,) inner bin edges in the standardised Y space

    Both outputs shape (N_q,).
    """
    if logits_np.shape[-1] != J * J + 9 + 4:
        raise ValueError(
            f'expected last dim = {J*J + 9 + 4}, got {logits_np.shape[-1]}'
        )

    unpacked = unpack_2d_head(logits_np, J)
    p_mat = unpacked['p_mat']          # (N_q, J, J)
    w     = unpacked['w_reg']          # (N_q, 9)
    tr    = unpacked['tail_raw']       # (N_q, 4)

    lo, hi = float(edges_np[0]), float(edges_np[-1])
    delta = (hi - lo) / J

    # Tail scales in std-Y units: δ · (softplus(raw) + floor)
    sL0 = delta * (_softplus(tr[..., 0]) + SOFTPLUS_FLOOR)  # (N_q,)
    sR0 = delta * (_softplus(tr[..., 1]) + SOFTPLUS_FLOOR)
    sL1 = delta * (_softplus(tr[..., 2]) + SOFTPLUS_FLOOR)
    sR1 = delta * (_softplus(tr[..., 3]) + SOFTPLUS_FLOOR)

    centers = 0.5 * (edges_np[:-1] + edges_np[1:])  # (J,)

    # ── Inner (region 0) ──────────────────────────────────────────────
    marg0 = p_mat.sum(axis=-1)   # (N_q, J)  Y0 marginal within inner
    marg1 = p_mat.sum(axis=-2)   # (N_q, J)  Y1 marginal within inner
    E_inner_y0 = (marg0 * centers).sum(axis=-1)  # (N_q,)
    E_inner_y1 = (marg1 * centers).sum(axis=-1)

    # ── Edge L0 / R0 (y0 outside, y1 inner) ────────────────────────────
    E_L0_y0 = lo - sL0 * SQRT_2_OVER_PI
    E_R0_y0 = hi + sR0 * SQRT_2_OVER_PI
    # y1 mean for L0/R0: conditional on j0 boundary row
    row_L0 = p_mat[..., 0,    :]  # (N_q, J) p(y0=lo boundary, y1 bins)
    row_R0 = p_mat[..., -1,   :]  # (N_q, J)
    sum_L0 = np.clip(row_L0.sum(axis=-1), 1e-45, None)
    sum_R0 = np.clip(row_R0.sum(axis=-1), 1e-45, None)
    E_L0_y1 = (row_L0 * centers).sum(axis=-1) / sum_L0
    E_R0_y1 = (row_R0 * centers).sum(axis=-1) / sum_R0

    # ── Edge L1 / R1 (y1 outside, y0 inner) ────────────────────────────
    E_L1_y1 = lo - sL1 * SQRT_2_OVER_PI
    E_R1_y1 = hi + sR1 * SQRT_2_OVER_PI
    col_L1 = p_mat[..., :, 0 ]    # (N_q, J) p(y0 bins, y1=lo boundary)
    col_R1 = p_mat[..., :, -1]
    sum_L1 = np.clip(col_L1.sum(axis=-1), 1e-45, None)
    sum_R1 = np.clip(col_R1.sum(axis=-1), 1e-45, None)
    E_L1_y0 = (col_L1 * centers).sum(axis=-1) / sum_L1
    E_R1_y0 = (col_R1 * centers).sum(axis=-1) / sum_R1

    # ── Corners (both outside) ─────────────────────────────────────────
    # APPROXIMATION: treat y0 and y1 tails as independent for the mean
    # (exact when ρ = 0). Justified because corner weights are typically
    # <1% each — the ρ-correction on the mean is O(ρ²) · σ_tail, well
    # below the resolution of pehe.
    E_L0L1_y0 = lo - sL0 * SQRT_2_OVER_PI   # corner (-,-)
    E_L0L1_y1 = lo - sL1 * SQRT_2_OVER_PI
    E_L0R1_y0 = lo - sL0 * SQRT_2_OVER_PI   # corner (-,+)
    E_L0R1_y1 = hi + sR1 * SQRT_2_OVER_PI
    E_R0L1_y0 = hi + sR0 * SQRT_2_OVER_PI   # corner (+,-)
    E_R0L1_y1 = lo - sL1 * SQRT_2_OVER_PI
    E_R0R1_y0 = hi + sR0 * SQRT_2_OVER_PI   # corner (+,+)
    E_R0R1_y1 = hi + sR1 * SQRT_2_OVER_PI

    # ── Combine over regions ───────────────────────────────────────────
    # Region order matches losses/BarDistribution2D.R_*: 0..8.
    E_y0 = (
          w[..., 0] * E_inner_y0
        + w[..., 1] * E_L0_y0
        + w[..., 2] * E_R0_y0
        + w[..., 3] * E_L1_y0
        + w[..., 4] * E_R1_y0
        + w[..., 5] * E_L0L1_y0
        + w[..., 6] * E_L0R1_y0
        + w[..., 7] * E_R0L1_y0
        + w[..., 8] * E_R0R1_y0
    )
    E_y1 = (
          w[..., 0] * E_inner_y1
        + w[..., 1] * E_L0_y1
        + w[..., 2] * E_R0_y1
        + w[..., 3] * E_L1_y1
        + w[..., 4] * E_R1_y1
        + w[..., 5] * E_L0L1_y1
        + w[..., 6] * E_L0R1_y1
        + w[..., 7] * E_R0L1_y1
        + w[..., 8] * E_R0R1_y1
    )
    return E_y0, E_y1


def region_weight_summary(logits_np, J):
    """Per-query mean region-weight distribution — for diagnostic printing.

    Returns (9,) mean weight over queries. Handy for verifying whether the
    tail regions actually carry mass under a given model + input.
    """
    unpacked = unpack_2d_head(logits_np, J)
    return unpacked['w_reg'].mean(axis=tuple(range(unpacked['w_reg'].ndim - 1)))
