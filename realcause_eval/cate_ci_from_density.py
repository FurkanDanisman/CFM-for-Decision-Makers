"""CATE 95% CI / coverage / length from 1D-marginal OR 2D-joint density dumps.

Reads per-realization NPZs written by realcause_eval/{do_pfn,cpfn1d,uwyk1d,
cpfn2d,graph2d,do_pfn_bb} runs with DENSITY_DUMP=1. Two schemas supported:

  1D marginals (cpfn1d, dopfn, uwyk1d):
    edges          shape (nbins+1,)
    p_y0_scaled    shape (N_q, nbins) softmax marginal p(Y|do(0)) per query
    p_y1_scaled    shape (N_q, nbins) softmax marginal p(Y|do(1)) per query
    y_shift, y_scale (scalars)  — raw = scaled*y_scale + y_shift
    true_cate_per_query shape (N_q,)  — ground-truth CATE per query
    (UWYK also carries base_s_left/right + per-arm sL_raw_0/1 / sR_raw_0/1
     for the half-Gaussian tail atoms.)

  2D joint (cpfn2d, graph2d, dopfn_bb — for 2D-head models):
    edges          shape (J+1,)
    p_joint_scaled shape (N_q, J, J)  — axis 1 = Y0 bin, axis 2 = Y1 bin
    y_shift, y_scale, true_cate_per_query as above

  For dopfn_bb the per-realization density lives at
  <method_dir>/<DATASET>/density_r<###>.npz (NOT a summary NPZ), and PEHE/
  ε_ATE come from an aggregate <method_dir>/<DATASET>/summary.npz that
  stores pehe[] / eps_ate[] arrays indexed by realization.

Method:
  1D marginals path — assume Y|do(0) ⊥ Y|do(1) → p(τ) = p_y1 * flip(p_y0)
    (1D discrete convolution / FFT for uniform bins; N²-atom enumeration
    for non-uniform bins; MC sampling for UWYK per-arm tail atoms).
  2D joint path — no independence assumption; p(τ = c[j] − c[i]) is the
    anti-diagonal sum of p_joint[:, i, j] (uniform bins → 2J−1 discrete τ
    values evenly spaced by bin width).
  Point CATE for the sanity check is E[Y1] − E[Y0] under the density's
  marginals; by construction this equals the mean of p(τ).

  Per-query CI: linear-interp inverse-CDF at 0.025 / 0.975, un-scale by
  y_scale (y_shift cancels in τ = Y1 − Y0). Coverage = mean 1[τ_true ∈ CI],
  Length = mean (τ_hi − τ_lo). Aggregate per realization.

Usage:
    python realcause_eval/cate_ci_from_density.py \\
        --out-root /scratch/.../results_realcause_all \\
        --methods cpfn1d dopfn uwyk1d cpfn2d graph2d dopfnbb
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np


DATASETS = ('IHDP', 'ACIC', 'CPS', 'PSID', 'PSID_bal')

# Per-method point-estimate keys (from the same NPZs).
# 1D methods:
#   cpfn1d writes pehe_raw / err_raw (unsuffixed).
#   dopfn  writes pehe_dopfn / err_dopfn.
#   uwyk1d writes suffixed keys — use noanc for the point row (matches paper
#     Table 3 UWYK No-Anc). Change to _v3b via --uwyk-tag if you want that row.
# 2D methods:
#   cpfn2d  writes pehe_raw / err_raw (unsuffixed) alongside p_joint_scaled.
#   graph2d writes pehe_raw_<mode> / err_raw_<mode>; noanc matches paper.
#     Change via --graph2d-tag if you want v3b/v6a/etc.
#   dopfnbb stores per-realization pehe / eps_ate in an aggregate summary.npz
#     — resolved separately via _load_dopfnbb_realization (not this table).
_POINT_KEYS = {
    'cpfn1d':  ('pehe_raw',       'err_raw'),
    'dopfn':   ('pehe_dopfn',     'err_dopfn'),
    'uwyk1d':  ('pehe_raw_noanc', 'err_raw_noanc'),
    'cpfn2d':  ('pehe_raw',       'err_raw'),
    'graph2d': ('pehe_raw_noanc', 'err_raw_noanc'),
    # dopfnbb resolved via _load_dopfnbb_summary; entry unused.
    'dopfnbb': ('pehe',           'eps_ate'),
}

# Methods whose density lives in per-realization NPZs whose filename is
# <DATASET>_r<###>.npz alongside the point-CATE keys.
_INLINE_DENSITY_METHODS = {
    'cpfn1d', 'dopfn', 'uwyk1d', 'cpfn2d', 'graph2d',
}
# Methods whose density is dumped separately (density_r<###>.npz) with
# PEHE / ε_ATE arrays living in an adjacent summary.npz.
_SPLIT_DENSITY_METHODS = {
    'dopfnbb',
}

# Per-method dataset-directory name overrides. dopfn_bb legacy uses
# 'PSIDbal' (one word) even though the mega-sbatch table's canonical
# label is 'PSID_bal'. Only add entries that differ from the canonical
# name (used in the table header).
_DATASET_DIR_ALIASES = {
    'dopfnbb': {'PSID_bal': 'PSIDbal'},
}


def _load_density(npz_path: str):
    """Return a tuple describing the density in an NPZ, or None.

    Return shape depends on schema (first element is the schema tag):
      ('bar',    centers, centers, p_y0, p_y1, y_shift, y_scale, true_cate)
      ('uwyk',   centers0_pq, centers1_pq, p_y0, p_y1, y_shift, y_scale, true_cate)
      ('joint2d',centers, p_joint,          y_shift, y_scale, true_cate)

    Schemas:
    - 2D joint (cpfn2d, graph2d, dopfn_bb): 'p_joint_scaled' key present with
      shape (N_q, J, J). Axis 1 = Y0 bin, axis 2 = Y1 bin (matches every 2D
      wrapper's convention — p_y0 = joint.sum(axis=2), p_y1 = joint.sum(axis=1)).
      Centers are edge midpoints (uniform J bins).
    - uniform-bar (cpfn1d, dopfn): p_y{0,1} shape (N_q, K), edges (K+1).
      Reconstruct K bar-center atoms from edges midpoints (or use
      'effective_centers' for DoPFN's tail-adjusted first/last-bar means).
    - UWYK BarDistribution: p_y{0,1} shape (N_q, K+2) = [pL, pBars, pR],
      edges (K+1), plus sL_raw/sR_raw and base_s_left/right/scale_floor.
      Reconstruct K+2 atoms: [E_left, K bar mids, E_right] where
      E_left/right are the half-Gaussian tail expected values —
      matches bar_dist.mean(pred) exactly.
    """
    try:
        with np.load(npz_path, allow_pickle=True) as z:
            base_required = {'edges', 'y_shift', 'y_scale', 'true_cate_per_query'}
            if not base_required.issubset(set(z.files)):
                return None
            edges = np.asarray(z['edges'], dtype=np.float64)
            y_shift = float(z['y_shift'])
            y_scale = float(z['y_scale'])
            true_cate_pq = np.asarray(z['true_cate_per_query'], dtype=np.float64)
            keys = set(z.files)

            # ── 2D joint schema first — takes precedence when present, since
            #    p_joint is the strictly-more-informative density (no
            #    independence assumption).
            if 'p_joint_scaled' in keys:
                p_joint = np.asarray(z['p_joint_scaled'], dtype=np.float64)
                assert p_joint.ndim == 3 and p_joint.shape[1] == p_joint.shape[2], (
                    f'p_joint_scaled expected (N_q, J, J); got {p_joint.shape} in {npz_path}')
                J = p_joint.shape[1]
                assert edges.size == J + 1, (
                    f'edges size {edges.size} inconsistent with joint J={J} in {npz_path}')
                # Per-query normalization; keeps arithmetic tight even if
                # the writer's softmax leaked ε mass.
                s = p_joint.sum(axis=(1, 2), keepdims=True)
                p_joint = p_joint / np.where(s > 0, s, 1.0)
                centers = 0.5 * (edges[:-1] + edges[1:])
                return ('joint2d', centers, p_joint, y_shift, y_scale, true_cate_pq)

            if 'p_y0_scaled' not in keys or 'p_y1_scaled' not in keys:
                return None
            p_y0  = np.asarray(z['p_y0_scaled'], dtype=np.float64)
            p_y1  = np.asarray(z['p_y1_scaled'], dtype=np.float64)

            K = edges.size - 1                     # bar count
            nbins = p_y0.shape[-1]

            if nbins == K + 2 and 'base_s_left' in keys:
                # UWYK BarDistribution: [pL, pBars, pR] over K+2 atoms.
                base_sL  = float(z['base_s_left']);  base_sR = float(z['base_s_right'])
                scale_floor = float(z['scale_floor'])
                # Per-query tail scales (arm 0 for the reconstructed atoms;
                # arm 1 tail scales differ but we only need centers, and the
                # tail center positions depend on sL/sR which are per-arm).
                # For a KISS reconstruction: use the arm's own sL/sR for each
                # arm's atom positions — return TWO center arrays.
                sL0 = np.asarray(z['sL_raw_0'], dtype=np.float64)
                sR0 = np.asarray(z['sR_raw_0'], dtype=np.float64)
                sL1 = np.asarray(z['sL_raw_1'], dtype=np.float64)
                sR1 = np.asarray(z['sR_raw_1'], dtype=np.float64)
                def _softplus(x): return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)
                sL0 = base_sL * (_softplus(sL0) + scale_floor)
                sR0 = base_sR * (_softplus(sR0) + scale_floor)
                sL1 = base_sL * (_softplus(sL1) + scale_floor)
                sR1 = base_sR * (_softplus(sR1) + scale_floor)
                sqrt2pi = np.sqrt(2.0 / np.pi)
                bar_mids = 0.5 * (edges[:-1] + edges[1:])           # (K,)
                # Per-query atom centers: (N_q, K+2) — tail atoms per query.
                centers0 = np.empty((p_y0.shape[0], nbins), dtype=np.float64)
                centers0[:, 0]    = edges[0]  - sqrt2pi * sL0
                centers0[:, 1:-1] = bar_mids[None, :]
                centers0[:, -1]   = edges[-1] + sqrt2pi * sR0
                centers1 = np.empty_like(centers0)
                centers1[:, 0]    = edges[0]  - sqrt2pi * sL1
                centers1[:, 1:-1] = bar_mids[None, :]
                centers1[:, -1]   = edges[-1] + sqrt2pi * sR1
                p_y0 /= p_y0.sum(axis=-1, keepdims=True).clip(min=1e-12)
                p_y1 /= p_y1.sum(axis=-1, keepdims=True).clip(min=1e-12)
                # Return per-arm centers signalling UWYK schema.
                return ('uwyk', centers0, centers1, p_y0, p_y1,
                        y_shift, y_scale, true_cate_pq)

            # Prefer effective_centers when the eval wrote them (dopfn's
            # FullSupportBarDistribution: first/last bars have tail-adjusted
            # means, not simple midpoints). Falls back to midpoints if not
            # present.
            if 'effective_centers' in keys:
                centers = np.asarray(z['effective_centers'], dtype=np.float64)
                assert centers.size == nbins, (
                    f'effective_centers has {centers.size} entries but density has '
                    f'{nbins} bins — schema mismatch in {npz_path}')
            elif edges.size >= nbins + 1:
                centers = 0.5 * (edges[:nbins] + edges[1:nbins + 1])
            else:
                width = float(edges[1] - edges[0])
                centers = np.arange(nbins) * width + float(edges[0])
            p_y0 /= p_y0.sum(axis=-1, keepdims=True).clip(min=1e-12)
            p_y1 /= p_y1.sum(axis=-1, keepdims=True).clip(min=1e-12)
            return ('bar', centers, centers, p_y0, p_y1,
                    y_shift, y_scale, true_cate_pq)
    except Exception as e:
        print(f'  [warn] {npz_path}: {e}', file=sys.stderr)
        return None


def _is_uniform(centers: np.ndarray, rtol: float = 1e-4) -> bool:
    """True iff consecutive gaps are equal within rtol × mean gap."""
    if centers.size < 3:
        return True
    d = np.diff(centers)
    return bool(np.max(np.abs(d - d.mean())) < rtol * abs(d.mean()) + 1e-12)


def _ci_from_atoms_uniform(centers: np.ndarray, p_y0: np.ndarray, p_y1: np.ndarray,
                            lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    """FAST path: bins are uniform, so N² pairs bucket onto 2N-1 τ values.
    p_τ[q, k] = Σ_{i-j=k+(N-1)} p_y1[q, i] · p_y0[q, j]  via FFT conv.
    Then CDF → interp at lo/hi. Returns (τ_lo, τ_hi) shape (N_q,)."""
    from numpy.fft import rfft, irfft
    N = centers.size
    n_out = 2 * N - 1
    n_fft = 1 << (n_out - 1).bit_length()
    F1 = rfft(p_y1, n=n_fft, axis=-1)
    F0 = rfft(p_y0[:, ::-1], n=n_fft, axis=-1)
    p_tau = irfft(F1 * F0, n=n_fft, axis=-1)[:, :n_out]
    p_tau = np.clip(p_tau, 0.0, None)
    p_tau /= p_tau.sum(axis=-1, keepdims=True).clip(min=1e-12)
    width = float(centers[1] - centers[0])
    tau = (np.arange(n_out) - (N - 1)).astype(np.float64) * width
    # tau is already sorted → cumulative interpolate directly.
    cdf = np.cumsum(p_tau, axis=-1)
    return _quantile_from_sorted(tau[None, :], cdf, lo), _quantile_from_sorted(tau[None, :], cdf, hi)


def _ci_from_atoms_general(centers: np.ndarray, p_y0: np.ndarray, p_y1: np.ndarray,
                            lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    """GENERAL path: bins may be non-uniform. Enumerate all N² atoms per query,
    sort by τ, cumulate probabilities, interpolate CDF quantiles.
    Cost: O(N_q · N² · log N²)  — fine for N ≤ ~200. For large N use the
    uniform path via _is_uniform dispatch."""
    N_q = p_y0.shape[0]
    N = centers.size
    tau_mat = centers[:, None] - centers[None, :]   # (N, N) shared across queries
    tau_flat = tau_mat.ravel()                       # (N²,)
    order = np.argsort(tau_flat, kind='stable')
    tau_sorted = tau_flat[order]                     # (N²,) — same for every query

    tau_lo_arr = np.empty(N_q, dtype=np.float64)
    tau_hi_arr = np.empty(N_q, dtype=np.float64)
    for q in range(N_q):
        p_mat = p_y1[q, :, None] * p_y0[q, None, :]  # (N, N)
        p_sorted = p_mat.ravel()[order]
        cdf = np.cumsum(p_sorted)
        cdf /= max(cdf[-1], 1e-12)
        tau_lo_arr[q] = _quantile_from_sorted(tau_sorted[None, :], cdf[None, :], lo)[0]
        tau_hi_arr[q] = _quantile_from_sorted(tau_sorted[None, :], cdf[None, :], hi)[0]
    return tau_lo_arr, tau_hi_arr


def _ci_uwyk_per_query(centers0_pq: np.ndarray, centers1_pq: np.ndarray,
                        p_y0: np.ndarray, p_y1: np.ndarray,
                        lo: float, hi: float,
                        n_samples: int = 8000, seed: int = 0
                        ) -> tuple[np.ndarray, np.ndarray]:
    """UWYK: per-query centers (tail atoms depend on per-arm sL/sR).

    Exact enumeration is O(N_q · N²) — for K+2 ≈ 1000 bins × 1618 CPS queries
    that's 1.6B atoms per realization, hours per dataset. Use MC instead:
    inverse-CDF sample n_samples atoms per arm per query, take τ = y1 - y0,
    empirical quantiles. Under independence this converges to the exact CI
    at O(1/√n_samples); at 8000 samples the CI edge error is ~1% of density
    std — well below the model's PEHE and comparable to per-realization
    Monte Carlo variance we already carry.

    Cost drops from N² = 1M per query to n_samples = 8k per query — 125×
    faster. For CPS: seconds instead of hours.
    """
    N_q, N = p_y0.shape
    rng = np.random.default_rng(seed)

    # Per-arm per-query CDFs → invert via searchsorted.
    cdf0 = np.cumsum(p_y0, axis=-1); cdf0 /= cdf0[:, -1:].clip(min=1e-12)
    cdf1 = np.cumsum(p_y1, axis=-1); cdf1 /= cdf1[:, -1:].clip(min=1e-12)
    u0 = rng.random((n_samples, N_q))
    u1 = rng.random((n_samples, N_q))

    tau_lo = np.empty(N_q, dtype=np.float64)
    tau_hi = np.empty(N_q, dtype=np.float64)
    for q in range(N_q):
        idx0 = np.searchsorted(cdf0[q], u0[:, q], side='right').clip(0, N - 1)
        idx1 = np.searchsorted(cdf1[q], u1[:, q], side='right').clip(0, N - 1)
        y0s = centers0_pq[q][idx0]
        y1s = centers1_pq[q][idx1]
        tau_s = y1s - y0s
        tau_lo[q] = float(np.quantile(tau_s, lo))
        tau_hi[q] = float(np.quantile(tau_s, hi))
    return tau_lo, tau_hi


_ALPHA_CI = 0.05    # 95% CI


def _winkler_is_vec(lo, hi, y, alpha=_ALPHA_CI):
    """Vectorized Winkler Interval Score IS_α per query. Lower = better.
    IS = (hi-lo) + (2/α)·max(lo-y, 0) + (2/α)·max(y-hi, 0)."""
    length = hi - lo
    return length + (2.0 / alpha) * np.maximum(lo - y, 0.0) \
                  + (2.0 / alpha) * np.maximum(y - hi, 0.0)


def _crps_from_uniform_atoms(mass, atoms, y):
    """Empirical CRPS from atomic distributions with atoms sorted ascending.

    Uses closed-form for step-function CDF:
        CRPS(F, y) = Σ_k (a_{k+1} - a_k) · (F_k - 1[a_k ≥ y])²
    where F_k = cumulative mass up to and including a_k.

    Vectorized over queries: mass (N_q, K), atoms (K,) uniform, y (N_q,).
    All in same units; caller applies y_scale if needed.
    """
    da = float(atoms[1] - atoms[0])
    F  = np.cumsum(mass, axis=-1) / mass.sum(axis=-1, keepdims=True).clip(min=1e-12)
    step = (atoms[None, :] >= y[:, None]).astype(np.float64)   # (N_q, K)
    return np.sum((F - step) ** 2, axis=-1) * da


def _crps_from_sorted_atoms_pq(atoms_sorted, mass_sorted, y):
    """Per-query CRPS when each query has its own sorted (atoms, masses).

    atoms_sorted, mass_sorted: (N_q, N_atoms), sorted ascending per row.
    y: (N_q,). All in same units. O(N_q · N_atoms) — vectorized.

    Discrete CRPS with step CDF: CRPS = Σ_k Δa_k · (F_k − 1[a_k ≥ y])²  where
    Δa_k = a_{k+1} − a_k (or 0 for last atom)."""
    N_q, N = atoms_sorted.shape
    F = np.cumsum(mass_sorted, axis=-1)
    F = F / F[:, -1:].clip(min=1e-12)
    step = (atoms_sorted >= y[:, None]).astype(np.float64)
    diffs = np.diff(atoms_sorted, axis=-1)                     # (N_q, N-1)
    diffs = np.concatenate([diffs, np.zeros((N_q, 1))], axis=-1)  # last atom width = 0
    return np.sum(diffs * (F - step) ** 2, axis=-1)


def _reconstruct_p_tau_uniform_raw(loaded):
    """Reconstruct p(τ|x_q) in RAW units on a uniform tau grid.

    Returns (p_tau_raw, tau_raw) OR (None, None) if the schema has
    per-query non-uniform atoms (uwyk, dopfn's non-uniform effective_centers)
    → caller falls back to _crps_from_sorted_atoms_pq.

    - joint2d: antidiagonal of p_joint → 2J-1 uniform atoms (scaled), scale to raw.
    - bar uniform: FFT convolution → 2K-1 uniform atoms, scale to raw.
    - bar non-uniform (dopfn) / uwyk: returns (None, None).
    """
    if loaded[0] == 'joint2d':
        _, centers, p_joint, _, y_scale, _ = loaded
        J = p_joint.shape[1]
        w_scaled = float(centers[1] - centers[0])
        mass_tau = np.zeros((p_joint.shape[0], 2 * J - 1), dtype=np.float64)
        for k in range(2 * J - 1):
            mass_tau[:, k] = np.trace(p_joint, axis1=-2, axis2=-1, offset=k - (J - 1))
        tau_raw = (np.arange(2 * J - 1) - (J - 1)).astype(np.float64) * w_scaled * y_scale
        p_tau_raw = mass_tau / (w_scaled * y_scale).clip(min=1e-12) if hasattr(np, 'clip') else mass_tau
        # Renormalize as densities.
        dtau = float(tau_raw[1] - tau_raw[0])
        s = mass_tau.sum(axis=-1, keepdims=True).clip(min=1e-12)
        p_tau_raw = (mass_tau / s) / dtau
        return p_tau_raw, tau_raw
    schema, centers0, centers1, p_y0, p_y1, _, y_scale, _ = loaded
    if schema == 'uwyk':
        return None, None
    centers = centers0
    if not _is_uniform(centers):
        return None, None
    # bar uniform: FFT convolution → 2K-1 uniform atoms.
    from numpy.fft import rfft, irfft
    K = p_y0.shape[1]
    n_out = 2 * K - 1
    n_fft = 1 << (n_out - 1).bit_length()
    F1 = rfft(p_y1, n=n_fft, axis=-1)
    F0 = rfft(p_y0[:, ::-1], n=n_fft, axis=-1)
    mass = np.clip(irfft(F1 * F0, n=n_fft, axis=-1)[:, :n_out], 0.0, None)
    w_scaled = float(centers[1] - centers[0])
    tau_raw = (np.arange(n_out) - (K - 1)).astype(np.float64) * w_scaled * y_scale
    dtau = float(tau_raw[1] - tau_raw[0])
    s = mass.sum(axis=-1, keepdims=True).clip(min=1e-12)
    p_tau_raw = (mass / s) / dtau
    return p_tau_raw, tau_raw


def _crps_per_query_any_schema(loaded, true_cate_pq_raw):
    """Compute CRPS per query in RAW units for any schema.

    Uses uniform-atom fast path when possible; otherwise falls back to
    per-query sorted-atom enumeration (dopfn non-uniform, uwyk per-arm).
    """
    p_tau_raw, tau_raw = _reconstruct_p_tau_uniform_raw(loaded)
    if p_tau_raw is not None:
        dtau = float(tau_raw[1] - tau_raw[0])
        F = np.cumsum(p_tau_raw, axis=-1) * dtau
        F = F / F[:, -1:].clip(min=1e-12)
        step = (tau_raw[None, :] >= true_cate_pq_raw[:, None]).astype(np.float64)
        return np.sum((F - step) ** 2, axis=-1) * dtau
    # Fallback: per-query sorted atoms. Iterates queries with a TRANSIENT
    # (K²,) allocation each — avoids preallocating (N_q, K²) which is a
    # multi-GB memory bomb (dopfn K=1000, CPS N_q=1618 → 13 GB otherwise).
    schema, centers0, centers1, p_y0, p_y1, _, y_scale, _ = loaded
    N_q, K = p_y0.shape
    crps = np.empty(N_q, dtype=np.float64)
    if schema == 'uwyk':
        # UWYK K is often ~1000 → K² = 1M atoms per query, argsort ~50ms
        # (CPS N_q=1618 → 2h). Use empirical CRPS from MC samples instead:
        # CRPS_emp = (1/n)·Σ|x_i - y| − (1/n²)·Σ(2i-n-1)·x_(i)  for sorted x_(i).
        # O(n log n) per query at n=8000 → ~1ms; matches the same MC
        # scheme used by _ci_uwyk_per_query.
        n_samples = 8000
        from numpy.random import default_rng
        rng = default_rng(0)
        cdf0 = np.cumsum(p_y0, axis=-1); cdf0 /= cdf0[:, -1:].clip(min=1e-12)
        cdf1 = np.cumsum(p_y1, axis=-1); cdf1 /= cdf1[:, -1:].clip(min=1e-12)
        u0 = rng.random((n_samples, N_q))
        u1 = rng.random((n_samples, N_q))
        i_arr = np.arange(1, n_samples + 1)
        weights = (2 * i_arr - n_samples - 1) / (n_samples * n_samples)
        for q in range(N_q):
            idx0 = np.searchsorted(cdf0[q], u0[:, q], side='right').clip(0, K - 1)
            idx1 = np.searchsorted(cdf1[q], u1[:, q], side='right').clip(0, K - 1)
            y0s = centers0[q][idx0]
            y1s = centers1[q][idx1]
            tau_s = (y1s - y0s) * y_scale
            tau_sorted = np.sort(tau_s)
            y = float(true_cate_pq_raw[q])
            term1 = float(np.mean(np.abs(tau_sorted - y)))
            term2 = float(np.sum(weights * tau_sorted))
            crps[q] = term1 - term2
        return crps
    # bar non-uniform (dopfn's effective_centers): shared centers across
    # queries, K² atoms. Sort atoms + widths ONCE outside the loop; only
    # the mass reshuffle happens per query.
    centers = centers0 * y_scale
    a_flat = (centers[None, :] - centers[:, None]).ravel()           # (K²,)
    order  = np.argsort(a_flat, kind='stable')
    a_sorted = a_flat[order]                                          # (K²,)
    diffs = np.diff(a_sorted)                                         # (K²-1,)
    for q in range(N_q):
        m_sorted = np.outer(p_y0[q], p_y1[q]).ravel()[order]          # transient (K²,)
        F = np.cumsum(m_sorted); F = F / max(float(F[-1]), 1e-12)
        step = (a_sorted >= true_cate_pq_raw[q]).astype(np.float64)
        crps[q] = float(np.sum(diffs * (F[:-1] - step[:-1]) ** 2))
    return crps


def _ci_from_joint_2d(centers: np.ndarray, p_joint: np.ndarray,
                       lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    """2D-joint path: NO independence assumption.

    p_joint has shape (N_q, J, J) with axis 1 = Y0 bin, axis 2 = Y1 bin.
    τ = Y1 − Y0 = c[j] − c[i]; for uniform bins this is (j − i) · w.

    p_τ[q, k] = Σ_{i, j : j − i = k − (J − 1)} p_joint[q, i, j]
              = anti-diagonal sum of the (J × J) joint at offset (k − (J − 1))

    Uses np.trace on the last two axes (supports leading batch dim) to sum
    each diagonal in one vectorized call. Cost: O(N_q · J) per k × (2J − 1)
    diagonals = O(N_q · J²) total — same as forming the marginals once.
    """
    N_q, J, _ = p_joint.shape
    assert _is_uniform(centers), (
        f'joint 2D CI expects uniform centers; got spacings '
        f'{np.diff(centers)[:3]}… (min={np.diff(centers).min():.3g}, '
        f'max={np.diff(centers).max():.3g})')
    w = float(centers[1] - centers[0])
    n_tau = 2 * J - 1
    p_tau = np.empty((N_q, n_tau), dtype=np.float64)
    for k in range(n_tau):
        offset = k - (J - 1)                       # j − i
        p_tau[:, k] = np.trace(p_joint, axis1=-2, axis2=-1, offset=offset)
    p_tau = np.clip(p_tau, 0.0, None)
    p_tau /= p_tau.sum(axis=-1, keepdims=True).clip(min=1e-12)
    tau = (np.arange(n_tau) - (J - 1)).astype(np.float64) * w
    cdf = np.cumsum(p_tau, axis=-1)
    return (_quantile_from_sorted(tau[None, :], cdf, lo),
            _quantile_from_sorted(tau[None, :], cdf, hi))


def _quantile_from_sorted(tau: np.ndarray, cdf: np.ndarray, level: float) -> np.ndarray:
    """Linear-interp inverse-CDF at `level`. tau and cdf are sorted along axis=-1.
    Broadcasts if tau shape (1, K) and cdf shape (N_q, K), returns (N_q,)."""
    # For each row of cdf, find first index where cdf >= level.
    below = cdf < level
    idx = np.argmax(~below, axis=-1)
    idx = np.where(cdf[..., -1] < level, cdf.shape[-1] - 1, idx)
    row = np.arange(cdf.shape[0])
    c_hi = cdf[row, idx]
    c_lo = np.where(idx > 0, cdf[row, np.maximum(idx - 1, 0)], 0.0)
    t_row = np.broadcast_to(tau, cdf.shape)
    t_hi = t_row[row, idx]
    t_lo = np.where(idx > 0, t_row[row, np.maximum(idx - 1, 0)], t_row[row, 0])
    w = np.where(c_hi > c_lo, (level - c_lo) / (c_hi - c_lo), 0.0)
    return t_lo + w * (t_hi - t_lo)




def process_npz(npz_path: str, pehe_key: str, err_key: str):
    """→ dict with STORED PEHE / ε_ATE (identical to the mega-sbatch numbers)
    plus DENSITY-DERIVED coverage / length, plus a consistency diagnostic.

    Point CATE (PEHE, ε_ATE) comes from the stored NPZ keys — same values as
    realcause_eval's original run. NOT re-derived from density. This makes
    the table's point row match the paper reproduction exactly.

    CI (Coverage, Length) comes from the density:
      - 2D-joint schema (cpfn2d, graph2d, dopfn_bb): anti-diagonal projection
        of the joint gives the exact p(τ) — NO independence assumption.
      - 1D-marginal schema (cpfn1d, dopfn, uwyk1d): assume Y|do(0) ⊥ Y|do(1),
        enumerate the N²-atom PMF of p(τ). Uniform bins get a fast FFT
        convolution; UWYK per-arm tail atoms use MC sampling.
    In every case: CDF → linear interp at 0.025 / 0.975, un-scale by y_scale.

    Consistency diagnostic: |mean(density-derived cate) − stored ate|. If
    near-zero, density and point agree (CI is on the same distribution as
    the point). If not, the density in the NPZ is not the one that produced
    the stored point CATE — CI is still valid for the SAVED density but
    doesn't calibrate the reported PEHE row.
    """
    loaded = _load_density(npz_path)
    if loaded is None:
        return None

    if loaded[0] == 'joint2d':
        # 2D-joint schema (cpfn2d, graph2d, dopfn_bb): no independence
        # assumption. Point CATE from marginals; CI from anti-diagonal p(τ).
        _, centers, p_joint, y_shift, y_scale, true_cate_pq = loaded
        p_y0_marg = p_joint.sum(axis=2)                 # (N_q, J)  — Y0 marginal
        p_y1_marg = p_joint.sum(axis=1)                 # (N_q, J)  — Y1 marginal
        e_y0 = (p_y0_marg * centers).sum(axis=-1)
        e_y1 = (p_y1_marg * centers).sum(axis=-1)
        cate_hat = (e_y1 - e_y0) * y_scale
        ate_hat_density = float(cate_hat.mean())
        tau_lo_axis, tau_hi_axis = _ci_from_joint_2d(centers, p_joint, 0.025, 0.975)
    else:
        schema, centers0, centers1, p_y0, p_y1, y_shift, y_scale, true_cate_pq = loaded
        # ── point CATE from density: sum(centers * p) per query, per arm.
        #    For UWYK per-arm centers differ (tail-atom positions depend on arm-
        #    specific sL/sR). For the bar schema centers0 == centers1.
        e_y0 = (p_y0 * centers0).sum(axis=-1)
        e_y1 = (p_y1 * centers1).sum(axis=-1)
        cate_hat = (e_y1 - e_y0) * y_scale            # y_shift cancels
        ate_hat_density = float(cate_hat.mean())

        # ── EXACT CI: enumerate atoms of p(τ) under independence, sort, quantile.
        # For the bar schema (cpfn1d, dopfn), centers0 == centers1 = 1D array;
        # can use the uniform-convolution or non-uniform N²-atom path.
        # For UWYK, centers differ per arm per query — must enumerate all N²
        # atoms per query with per-query per-arm centers. Slower but exact.
        if schema == 'uwyk':
            tau_lo_axis, tau_hi_axis = _ci_uwyk_per_query(centers0, centers1, p_y0, p_y1,
                                                         0.025, 0.975)
        else:
            centers = centers0        # bar schema: shared 1D
            if _is_uniform(centers):
                tau_lo_axis, tau_hi_axis = _ci_from_atoms_uniform(centers, p_y0, p_y1, 0.025, 0.975)
            else:
                tau_lo_axis, tau_hi_axis = _ci_from_atoms_general(centers, p_y0, p_y1, 0.025, 0.975)
    tau_lo = tau_lo_axis * y_scale
    tau_hi = tau_hi_axis * y_scale
    coverage = float(np.mean((true_cate_pq >= tau_lo) & (true_cate_pq <= tau_hi)))
    length   = float(np.mean(tau_hi - tau_lo))

    # ── Combined coverage+length metrics (both single-number, lower = better).
    # Winkler IS_0.05 is trivial from stored CI + true; CRPS requires
    # reconstructing per-query density on a shared grid (uniform-atom fast path
    # + fallback for uwyk / non-uniform dopfn).
    is_per_q = _winkler_is_vec(tau_lo, tau_hi, true_cate_pq)
    winkler = float(is_per_q.mean())
    if os.environ.get('SKIP_CRPS', '0') == '1':
        crps = float('nan')
    else:
        try:
            crps_per_q = _crps_per_query_any_schema(loaded, true_cate_pq)
            crps = float(crps_per_q.mean())
        except Exception as e:
            print(f'  [warn] CRPS compute failed for {npz_path}: {e}', file=sys.stderr)
            crps = float('nan')

    # ── STORED point estimates — these are the PEHE / ε_ATE we report.
    #    Come from the same NPZ that realcause_eval's mega-sbatch writes, so
    #    the point row here matches the paper reproduction row exactly.
    try:
        with np.load(npz_path, allow_pickle=True) as z:
            pehe_stored = float(z[pehe_key]) if pehe_key in z.files else float('nan')
            err_stored  = float(z[err_key])  if err_key  in z.files else float('nan')
            # Grab whichever stored ate scalar is available for the consistency
            # diagnostic. cpfn1d saves ate_raw; uwyk1d saves ate_raw_<tag>; dopfn
            # saves per-query cate_pred (mean-it here).
            ate_stored = float('nan')
            ate_key = None
            # Prefer the ate scalar matching the pehe_key's suffix (e.g.
            # pehe_raw_v3b → ate_raw_v3b), else any ate_* key.
            _prefer = pehe_key.replace('pehe_', 'ate_', 1)
            if _prefer in z.files:
                ate_stored = float(z[_prefer]); ate_key = _prefer
            else:
                # ate_pred is what dopfn_bb writes into density_r<###>.npz
                # (mean of per-query cate). Other 1D emitters use ate_raw /
                # ate_em / ate_dopfn / ate_full — check in that order.
                for k in ('ate_raw', 'ate_em', 'ate_dopfn', 'ate_full', 'ate_pred'):
                    if k in z.files:
                        ate_stored = float(z[k]); ate_key = k; break
                if ate_key is None and 'cate_pred' in z.files:
                    ate_stored = float(np.asarray(z['cate_pred']).mean())
                    ate_key = 'mean(cate_pred)'
    except Exception:
        pehe_stored, err_stored, ate_stored, ate_key = (
            float('nan'), float('nan'), float('nan'), None)

    density_vs_stored_ate = float('nan')
    if np.isfinite(ate_stored):
        density_vs_stored_ate = ate_hat_density - ate_stored

    return {
        # REPORTED (match the mega-sbatch table exactly).
        'pehe':          pehe_stored,
        'err':           err_stored,
        # CI derived from density.
        'coverage':      coverage,
        'length':        length,
        # Combined coverage+length metrics (lower = better).
        'winkler':       winkler,
        'crps':          crps,
        'n_queries':     int(true_cate_pq.size),
        # Diagnostics — surface how far the SAVED density's mean is from the
        # STORED point CATE. Small = density trustworthy for CI. Large = the
        # density in the NPZ is not the density that produced the point PEHE.
        'ate_density':   ate_hat_density,
        'ate_stored':    ate_stored,
        'ate_key':       ate_key,
        'density_vs_stored_ate': density_vs_stored_ate,
    }


def _mean_se(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float('nan'), float('nan')
    m = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else float('nan')
    return m, se


def _load_dopfnbb_summary(dataset_dir: str, pehe_key: str, err_key: str):
    """Load per-realization pehe / err arrays from dopfnbb's aggregate
    summary.npz. Returns (pehe_arr, err_arr) or (None, None) if missing.

    The dopfnbb eval writes an aggregate summary.npz alongside the
    per-realization density_r<###>.npz files. Point rows are indexed by
    realization order — same order density_r000, density_r001, ... are
    written."""
    path = os.path.join(dataset_dir, 'summary.npz')
    if not os.path.isfile(path):
        return None, None
    try:
        with np.load(path, allow_pickle=True) as z:
            pehe_arr = np.asarray(z[pehe_key], dtype=np.float64) if pehe_key in z.files else None
            err_arr  = np.asarray(z[err_key],  dtype=np.float64) if err_key  in z.files else None
        return pehe_arr, err_arr
    except Exception as e:
        print(f'  [warn] {path}: {e}', file=sys.stderr)
        return None, None


def summarize_method_dataset(method_dir: str, dataset: str,
                              pehe_key: str, err_key: str,
                              method: str = ''):
    """Walk NPZs; return per-cell aggregates + a max |derived - stored| diff
    diagnostic so we can eyeball whether density-derived point estimates
    reproduce the stored ones (should be ~0 under independence + normalization).

    Two layouts:
      - inline (cpfn1d, dopfn, uwyk1d, cpfn2d, graph2d):
          {method_dir}/{dataset}/{dataset}_r<###>.npz  contains density AND
          the pehe/err scalars.
      - split (dopfnbb): {method_dir}/{dataset}/density_r<###>.npz contains
          only the density; pehe[] / eps_ate[] arrays live in
          {method_dir}/{dataset}/summary.npz indexed by realization order.
    """
    # Per-method dir aliasing (e.g. dopfn_bb legacy stores PSID_bal under PSIDbal/).
    dataset_dir_name = _DATASET_DIR_ALIASES.get(method, {}).get(dataset, dataset)
    dataset_dir = os.path.join(method_dir, dataset_dir_name)
    is_split = (method in _SPLIT_DENSITY_METHODS)
    if is_split:
        paths = sorted(glob.glob(os.path.join(dataset_dir, 'density_r*.npz')))
        pehe_arr, err_arr = _load_dopfnbb_summary(dataset_dir, pehe_key, err_key)
    else:
        paths = sorted(glob.glob(os.path.join(dataset_dir, f'{dataset_dir_name}_r*.npz')))
        pehe_arr = err_arr = None
    if not paths:
        return None
    pehes, errs, covs, lens = [], [], [], []
    winklers, crpses = [], []
    ate_diffs = []          # density-mean vs stored ate (per realization)
    for r_idx, p in enumerate(paths):
        got = process_npz(p, pehe_key, err_key)
        if got is None:
            continue
        # For split layout, override stored pehe/err from the aggregate summary.
        if is_split:
            if pehe_arr is not None and r_idx < pehe_arr.size:
                got['pehe'] = float(pehe_arr[r_idx])
            if err_arr is not None and r_idx < err_arr.size:
                got['err'] = float(err_arr[r_idx])
        # REPORTED columns: stored PEHE / ε_ATE (identical to mega-sbatch).
        if np.isfinite(got['pehe']): pehes.append(got['pehe'])
        if np.isfinite(got['err']):  errs.append(got['err'])
        covs.append(got['coverage']); lens.append(got['length'])
        winklers.append(got['winkler']); crpses.append(got['crps'])
        if np.isfinite(got['density_vs_stored_ate']):
            ate_diffs.append(got['density_vs_stored_ate'])
    if not covs:
        return None
    return {
        'pehe':         _mean_se(pehes),
        'err':          _mean_se(errs),
        'cov':          _mean_se(covs),
        'len':          _mean_se(lens),
        'winkler':      _mean_se(winklers),
        'crps':         _mean_se(crpses),
        'n':            len(covs),
        # Consistency diagnostic: how far is the density mean from the stored ate?
        # If ≈ 0 the density is trustworthy; if not, CI here is on a different
        # distribution than the point CATE the mega-sbatch reports.
        'ate_max_diff': float(np.max(np.abs(ate_diffs))) if ate_diffs else float('nan'),
    }


def _fmt(m, se, big=False):
    if not np.isfinite(m):
        return '—'
    if big:
        m_s = f'{m:,.2f}'
        se_s = f'{se:,.2f}' if np.isfinite(se) else '—'
    else:
        m_s = f'{m:.3f}'
        se_s = f'{se:.3f}' if np.isfinite(se) else '—'
    return f'{m_s} ± {se_s}'


def _debug_single(npz_path: str, pehe_key: str, err_key: str) -> None:
    """Print a fingerprint of one realization: edges range, density means,
    density-derived cate summary, and every stored scalar we can grab for
    cross-check (pehe_raw, ate_raw, true_ate, cate_pred per-query if present).
    If mean(density-derived cate) differs from stored ate_raw, the density
    saved in the NPZ is NOT the density used to compute stored PEHE.

    Handles all three schemas ('bar', 'uwyk', 'joint2d')."""
    with np.load(npz_path, allow_pickle=True) as z:
        edges = np.asarray(z['edges'], dtype=np.float64)
        y_shift = float(z['y_shift']); y_scale = float(z['y_scale'])
        true_cate_pq = np.asarray(z['true_cate_per_query'], dtype=np.float64)
        cate_pred = np.asarray(z['cate_pred']).astype(np.float64) if 'cate_pred' in z.files else None
        pehe_stored = float(z[pehe_key]) if pehe_key in z.files else float('nan')
        # Pull any stored ate scalar for cross-check.
        ate_stored = float('nan')
        ate_stored_key = None
        for k in ('ate_raw', 'ate_em', 'ate_full', 'ate_pred'):
            if k in z.files:
                ate_stored = float(z[k]); ate_stored_key = k; break
        true_ate_stored = float(z['true_ate']) if 'true_ate' in z.files else float('nan')
        stored_keys = list(z.files)

    # Route through the same loader used by process_npz so debug matches
    # what the aggregator actually computes (handles UWYK per-arm atoms and
    # the 2D-joint schema).
    loaded = _load_density(npz_path)
    if loaded is None:
        print(f'  [debug] _load_density returned None for {npz_path}')
        return

    if loaded[0] == 'joint2d':
        schema = 'joint2d'
        _, centers, p_joint, _, _, _ = loaded
        p_y0_marg = p_joint.sum(axis=2)
        p_y1_marg = p_joint.sum(axis=1)
        e_y0 = (p_y0_marg * centers).sum(axis=-1)
        e_y1 = (p_y1_marg * centers).sum(axis=-1)
        p_shape = p_joint.shape
    else:
        schema, centers0, centers1, p_y0, p_y1, _, _, _ = loaded
        e_y0 = (p_y0 * centers0).sum(axis=-1) if centers0.ndim == 2 else (p_y0 * centers0[None, :]).sum(axis=-1)
        e_y1 = (p_y1 * centers1).sum(axis=-1) if centers1.ndim == 2 else (p_y1 * centers1[None, :]).sum(axis=-1)
        p_shape = p_y0.shape
        centers = centers0 if centers0.ndim == 1 else centers0[0]
    cate_from_diff = (e_y1 - e_y0) * y_scale

    print(f'\n[{npz_path}]')
    print(f'  density schema: {schema}')
    print(f'  NPZ keys: {sorted(stored_keys)}')
    print(f'  edges: shape={edges.shape}  range=[{edges.min():.4g}, {edges.max():.4g}]')
    print(f'  density shape: {p_shape}')
    if schema != 'joint2d':
        print(f'  centers[0:3]={centers[:3]}   centers[-3:]={centers[-3:]}')
    print(f'  y_shift={y_shift:.4g}  y_scale={y_scale:.4g}')
    print(f'  E[Y_0] (density-axis) mean q: {float(e_y0.mean()):.4g}')
    print(f'  E[Y_1] (density-axis) mean q: {float(e_y1.mean()):.4g}')
    print(f'  cate_from_diff (density → raw) mean q: {float(cate_from_diff.mean()):.4g}')
    if cate_pred is not None:
        print(f'  cate_pred (stored per-query)  mean q: {float(cate_pred.mean()):.4g}')
        print(f'  max |cate_from_diff - cate_pred|: {float(np.max(np.abs(cate_from_diff - cate_pred))):.4g}')
    if ate_stored_key is not None:
        gap = float(cate_from_diff.mean()) - ate_stored
        print(f'  stored {ate_stored_key} (scalar ate_hat) = {ate_stored:.4g}    '
              f'Δ vs density-mean = {gap:+.4g}')
    print(f'  true_cate_per_query mean q = {float(true_cate_pq.mean()):.4g}   '
          f'stored true_ate = {true_ate_stored:.4g}')
    print(f'  stored PEHE ({pehe_key}) = {pehe_stored:.4g}')
    if ate_stored_key is not None and np.isfinite(pehe_stored) and np.isfinite(true_ate_stored):
        bias = abs(ate_stored - true_ate_stored)
        print(f'  sanity: |ate_stored - true_ate| = {bias:.4g}  must be ≤ PEHE = {pehe_stored:.4g}  '
              f'→ {"OK" if bias <= pehe_stored + 1e-6 else "VIOLATED"}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-root', required=True,
                    help='Parent dir; expects <out-root>/<method>/<DATASET>/<D>_r<###>.npz.')
    ap.add_argument('--methods', nargs='+', default=['cpfn1d', 'dopfn', 'uwyk1d'],
                    help='Methods to include (one row each).')
    ap.add_argument('--datasets', nargs='+', default=None,
                    choices=list(DATASETS),
                    help='Restrict to a subset of datasets (default: all 5). '
                         'Use --datasets IHDP for a fast validation pass.')
    ap.add_argument('--uwyk-tag', default='noanc', choices=['noanc', 'v3b'],
                    help='Which anc-tag row of uwyk1d to use for PEHE/ε_ATE. '
                         'Default: noanc (matches paper Table 3 UWYK No-Anc).')
    ap.add_argument('--graph2d-tag', default='noanc',
                    help='Which anc-tag row of graph2d to use for PEHE/ε_ATE. '
                         'Default: noanc (matches paper Table 3). Also '
                         'accepts v3b, v6a, v5a, etc. — must match a key '
                         "written by eval_graph2d_realcause.py.")
    ap.add_argument('--dopfnbb-pehe-key', default='pehe',
                    choices=['pehe', 'pehe_em'],
                    help='Which pehe array in dopfnbb summary.npz to use. '
                         'Default: pehe (inner-9-region). pehe_em is the '
                         'EM-smoothed variant.')
    ap.add_argument('--out-md', default=None,
                    help='Also write the markdown table to this path.')
    ap.add_argument('--debug-one', nargs=2, metavar=('METHOD', 'DATASET'),
                    help='Print one-realization fingerprint (edges range, density '
                         'means, cate_from_diff vs cate_pred) for the first NPZ of '
                         'the given (method, dataset). Skips the aggregate table.')
    args = ap.parse_args()

    # Resolve point-key overrides once — same table shared by debug + aggregate.
    point_keys = dict(_POINT_KEYS)
    point_keys['uwyk1d']  = (f'pehe_raw_{args.uwyk_tag}',  f'err_raw_{args.uwyk_tag}')
    point_keys['graph2d'] = (f'pehe_raw_{args.graph2d_tag}',
                              f'err_raw_{args.graph2d_tag}')
    point_keys['dopfnbb'] = (args.dopfnbb_pehe_key,
                              'eps_ate' if args.dopfnbb_pehe_key == 'pehe' else 'eps_ate_em')

    if args.debug_one is not None:
        method, dataset = args.debug_one
        pehe_key, err_key = point_keys.get(method, ('pehe_raw', 'err_raw'))
        dataset_dir_name = _DATASET_DIR_ALIASES.get(method, {}).get(dataset, dataset)
        pattern = 'density_r*.npz' if method in _SPLIT_DENSITY_METHODS \
                                    else f'{dataset_dir_name}_r*.npz'
        paths = sorted(glob.glob(os.path.join(args.out_root, method, dataset_dir_name, pattern)))
        if not paths:
            sys.exit(f'FATAL: no NPZs found for {method}/{dataset}  '
                     f'(looked in {os.path.join(args.out_root, method, dataset_dir_name)})')
        _debug_single(paths[0], pehe_key, err_key)
        return

    if not os.path.isdir(args.out_root):
        sys.exit(f'FATAL: --out-root not found: {args.out_root}')

    big_pehe = {'CPS', 'PSID', 'PSID_bal'}
    big_len  = big_pehe

    # Restrict columns to --datasets if given (default: all 5).
    datasets_run = tuple(args.datasets) if args.datasets else DATASETS

    header = '| Method | ' + ' | '.join(datasets_run) + ' |'
    sep    = '|' + '|'.join(['---'] * (1 + len(datasets_run))) + '|'
    lines = [
        f'\nRealCause density-CI — {args.out_root}',
        '',
        '(each cell, top → bottom: √PEHE / ε_ATE (from stored point CATE — '
        'matches realcause_eval mega-sbatch), Coverage / Length (95% CI from '
        'the density; 1D marginals use Y|do(0) ⊥ Y|do(1) convolution, 2D '
        'joint uses anti-diagonal projection with no independence '
        'assumption), Winkler IS_0.05 (length + 40·miss) and CRPS — both '
        'single-number combined coverage+length metrics, lower = better; '
        'n = realizations)',
        '',
        header, sep,
    ]

    verify_lines = ['', '## Sanity: max |mean(density) − stored ate| across realizations',
                    '(should be ≈ 0 — the density used for CI IS the density that '
                    'produced the reported point CATE; if not, CI is on a different '
                    'distribution than the stored PEHE row)',
                    '', header, sep]

    import time
    _t_all = time.time()
    for method in args.methods:
        method_dir = os.path.join(args.out_root, method)
        pehe_key, err_key = point_keys.get(method, ('pehe_raw', 'err_raw'))
        cells = [method]
        verify_cells = [method]
        for d in datasets_run:
            _t0 = time.time()
            print(f'[{time.strftime("%H:%M:%S")}] processing {method:8s} / {d:9s} ...',
                  end='', flush=True)
            got = summarize_method_dataset(method_dir, d, pehe_key, err_key, method=method)
            print(f' done in {time.time() - _t0:5.1f}s'
                  + (f' (n={got["n"]})' if got is not None else ' (no data)'),
                  flush=True)
            if got is None:
                cells.append('—')
                verify_cells.append('—')
                continue
            pehe_str = _fmt(*got['pehe'], big=d in big_pehe)
            err_str  = _fmt(*got['err'],  big=False)
            cov_str  = _fmt(*got['cov'],  big=False)
            len_str  = _fmt(*got['len'],  big=d in big_len)
            is_str   = _fmt(*got['winkler'], big=d in big_len)
            crps_str = _fmt(*got['crps'],    big=d in big_len)
            n = got['n']
            cells.append(
                f'PEHE {pehe_str}<br>'
                f'ε_ATE {err_str}<br>'
                f'Cov {cov_str}<br>'
                f'Len {len_str}<br>'
                f'IS {is_str}<br>'
                f'CRPS {crps_str} (n={n})'
            )
            ad_max = got.get('ate_max_diff', float('nan'))
            if np.isfinite(ad_max):
                _fmt_ad = f'{ad_max:.2e}' if abs(ad_max) < 1 else f'{ad_max:.4f}'
                verify_cells.append(f'|Δate density-vs-stored| max = {_fmt_ad}')
            else:
                verify_cells.append('(no stored ate)')
        lines.append('| ' + ' | '.join(cells) + ' |')
        verify_lines.append('| ' + ' | '.join(verify_cells) + ' |')

    lines.extend(verify_lines)

    md = '\n'.join(lines) + '\n'
    print(md)
    if args.out_md:
        with open(args.out_md, 'w') as f:
            f.write(md)
        print(f'wrote {args.out_md}')


if __name__ == '__main__':
    main()
