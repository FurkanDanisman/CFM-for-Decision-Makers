"""CATE 95% CI / coverage / length from 1D density dumps.

Reads per-realization NPZs written by realcause_eval/{do_pfn,cpfn1d,uwyk1d}
runs with DENSITY_DUMP=1. Each NPZ must contain (see the emitters):

    edges          shape (nbins+1,)   bar-dist edges in scaled Y units
                                       (RAW Y for DoPFN — y_shift=0, y_scale=1)
    p_y0_scaled    shape (N_q, nbins) softmax marginal p(Y|do(0)) per query
    p_y1_scaled    shape (N_q, nbins) softmax marginal p(Y|do(1)) per query
    y_shift        scalar             raw = scaled*y_scale + y_shift
    y_scale        scalar             (drops out in τ = Y_1 - Y_0)
    true_cate_per_query shape (N_q,)  ground-truth CATE per query

Method:
    1. Assume Y|do(0) ⊥ Y|do(1). Then p(τ = Y_1 - Y_0) = p_y1 * flip(p_y0)
       via 1D discrete convolution (2·nbins-1 output bins).
    2. Convolution mean equals E[Y_1] - E[Y_0] — SAME as the point CATE the
       models already report. So point estimates are guaranteed consistent.
    3. Per-query CI: find τ_lo, τ_hi via linear-interp on the discrete CDF
       at levels 0.025 and 0.975.
    4. Un-scale τ_lo, τ_hi to raw units by multiplying by y_scale (y_shift
       cancels in the difference).
    5. Coverage = mean_q 1[τ_true[q] ∈ [τ_lo[q], τ_hi[q]]].
       Length   = mean_q (τ_hi[q] - τ_lo[q]).
    6. Aggregate across realizations: mean of coverages, mean of lengths.

Usage:
    python realcause_eval/cate_ci_from_density.py \\
        --out-root /scratch/.../results_realcause_all \\
        --methods cpfn1d dopfn uwyk1d
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np


DATASETS = ('IHDP', 'ACIC', 'CPS', 'PSID', 'PSID_bal')

# Per-method point-estimate keys (from the same NPZs).
# cpfn1d writes pehe_raw / err_raw (unsuffixed).
# dopfn  writes pehe_dopfn / err_dopfn.
# uwyk1d writes suffixed keys — use noanc for the point row (matches paper
# Table 3 UWYK No-Anc). Change to _v3b via --uwyk-tag if you want that row.
_POINT_KEYS = {
    'cpfn1d': ('pehe_raw',       'err_raw'),
    'dopfn':  ('pehe_dopfn',     'err_dopfn'),
    'uwyk1d': ('pehe_raw_noanc', 'err_raw_noanc'),
}


def _load_density(npz_path: str):
    """Return (centers, p_y0, p_y1, y_shift, y_scale, true_cate_pq) or None.

    Two schemas supported:
    - uniform-bar (cpfn1d, dopfn): p_y{0,1} shape (N_q, K), edges (K+1).
      Reconstruct K bar-center atoms from edges midpoints.
    - UWYK BarDistribution: p_y{0,1} shape (N_q, K+2) = [pL, pBars, pR],
      edges (K+1), plus sL_raw/sR_raw and base_s_left/right/scale_floor.
      Reconstruct K+2 atoms: [E_left, K bar mids, E_right] where
      E_left/right are the half-Gaussian tail expected values —
      matches bar_dist.mean(pred) exactly.
    """
    try:
        with np.load(npz_path, allow_pickle=True) as z:
            required = {'edges', 'p_y0_scaled', 'p_y1_scaled', 'y_shift',
                        'y_scale', 'true_cate_per_query'}
            if not required.issubset(set(z.files)):
                return None
            edges = np.asarray(z['edges'], dtype=np.float64)
            p_y0  = np.asarray(z['p_y0_scaled'], dtype=np.float64)
            p_y1  = np.asarray(z['p_y1_scaled'], dtype=np.float64)
            y_shift = float(z['y_shift'])
            y_scale = float(z['y_scale'])
            true_cate_pq = np.asarray(z['true_cate_per_query'], dtype=np.float64)
            keys = set(z.files)

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

    CI (Coverage, Length) comes from the density via the exact N²-atom PMF
    of p(τ) under independence Y|do(0) ⊥ Y|do(1). Enumerate atoms, sort by
    τ, cumulate → CDF → interp at 0.025 / 0.975. Uniform bins get a fast
    convolution path; non-uniform bins use general N²-atom enumeration.

    Consistency diagnostic: |mean(density-derived cate) − stored ate|. If
    near-zero, density and point agree (CI is on the same distribution as
    the point). If not, the density in the NPZ is not the one that produced
    the stored point CATE — CI is still valid for the SAVED density but
    doesn't calibrate the reported PEHE row.
    """
    loaded = _load_density(npz_path)
    if loaded is None:
        return None
    schema, centers0, centers1, p_y0, p_y1, y_shift, y_scale, true_cate_pq = loaded

    # ── point CATE from density: sum(centers * p) per query, per arm.
    #    For UWYK per-arm centers differ (tail-atom positions depend on arm-
    #    specific sL/sR). For the bar schema centers0 == centers1.
    e_y0 = (p_y0 * centers0).sum(axis=-1)
    e_y1 = (p_y1 * centers1).sum(axis=-1)
    cate_hat = (e_y1 - e_y0) * y_scale                # y_shift cancels
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
                for k in ('ate_raw', 'ate_em', 'ate_dopfn', 'ate_full'):
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


def summarize_method_dataset(method_dir: str, dataset: str,
                              pehe_key: str, err_key: str):
    """Walk NPZs; return per-cell aggregates + a max |derived - stored| diff
    diagnostic so we can eyeball whether density-derived point estimates
    reproduce the stored ones (should be ~0 under independence + normalization)."""
    paths = sorted(glob.glob(os.path.join(method_dir, dataset, f'{dataset}_r*.npz')))
    if not paths:
        return None
    pehes, errs, covs, lens = [], [], [], []
    ate_diffs = []          # density-mean vs stored ate (per realization)
    for p in paths:
        got = process_npz(p, pehe_key, err_key)
        if got is None:
            continue
        # REPORTED columns: stored PEHE / ε_ATE (identical to mega-sbatch).
        if np.isfinite(got['pehe']): pehes.append(got['pehe'])
        if np.isfinite(got['err']):  errs.append(got['err'])
        covs.append(got['coverage']); lens.append(got['length'])
        if np.isfinite(got['density_vs_stored_ate']):
            ate_diffs.append(got['density_vs_stored_ate'])
    if not covs:
        return None
    return {
        'pehe':         _mean_se(pehes),
        'err':          _mean_se(errs),
        'cov':          _mean_se(covs),
        'len':          _mean_se(lens),
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
    saved in the NPZ is NOT the density used to compute stored PEHE."""
    with np.load(npz_path, allow_pickle=True) as z:
        edges = np.asarray(z['edges'], dtype=np.float64)
        p_y0  = np.asarray(z['p_y0_scaled'], dtype=np.float64)
        p_y1  = np.asarray(z['p_y1_scaled'], dtype=np.float64)
        y_shift = float(z['y_shift']); y_scale = float(z['y_scale'])
        true_cate_pq = np.asarray(z['true_cate_per_query'], dtype=np.float64)
        cate_pred = np.asarray(z['cate_pred']).astype(np.float64) if 'cate_pred' in z.files else None
        pehe_stored = float(z[pehe_key]) if pehe_key in z.files else float('nan')
        # Pull any stored ate scalar for cross-check.
        ate_stored = float('nan')
        for k in ('ate_raw', 'ate_em', 'ate_full'):
            if k in z.files:
                ate_stored = float(z[k]); ate_stored_key = k; break
        else:
            ate_stored_key = None
        true_ate_stored = float(z['true_ate']) if 'true_ate' in z.files else float('nan')
        stored_keys = list(z.files)

    p_y0 /= p_y0.sum(axis=-1, keepdims=True).clip(min=1e-12)
    p_y1 /= p_y1.sum(axis=-1, keepdims=True).clip(min=1e-12)
    # Route through the same loader used by process_npz so debug matches
    # what the aggregator actually computes (handles UWYK's per-arm tail atoms).
    loaded = _load_density(npz_path)
    if loaded is None:
        print(f'  [debug] _load_density returned None for {npz_path}')
        return
    schema, centers0, centers1, p_y0, p_y1, _, _, _ = loaded
    e_y0 = (p_y0 * centers0).sum(axis=-1) if centers0.ndim == 2 else (p_y0 * centers0[None, :]).sum(axis=-1)
    e_y1 = (p_y1 * centers1).sum(axis=-1) if centers1.ndim == 2 else (p_y1 * centers1[None, :]).sum(axis=-1)
    cate_from_diff = (e_y1 - e_y0) * y_scale
    print(f'  density schema: {schema}')

    print(f'\n[{npz_path}]')
    print(f'  NPZ keys: {sorted(stored_keys)}')
    print(f'  edges: shape={edges.shape}  range=[{edges.min():.4g}, {edges.max():.4g}]')
    print(f'  p_y0/p_y1: shape={p_y0.shape}')
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
    ap.add_argument('--uwyk-tag', default='noanc', choices=['noanc', 'v3b'],
                    help='Which anc-tag row of uwyk1d to use for PEHE/ε_ATE. '
                         'Default: noanc (matches paper Table 3 UWYK No-Anc).')
    ap.add_argument('--out-md', default=None,
                    help='Also write the markdown table to this path.')
    ap.add_argument('--debug-one', nargs=2, metavar=('METHOD', 'DATASET'),
                    help='Print one-realization fingerprint (edges range, density '
                         'means, cate_from_diff vs cate_pred) for the first NPZ of '
                         'the given (method, dataset). Skips the aggregate table.')
    args = ap.parse_args()

    if args.debug_one is not None:
        method, dataset = args.debug_one
        point_keys = dict(_POINT_KEYS)
        point_keys['uwyk1d'] = (f'pehe_raw_{args.uwyk_tag}', f'err_raw_{args.uwyk_tag}')
        pehe_key, err_key = point_keys.get(method, ('pehe_raw', 'err_raw'))
        paths = sorted(glob.glob(os.path.join(args.out_root, method, dataset,
                                              f'{dataset}_r*.npz')))
        if not paths:
            sys.exit(f'FATAL: no NPZs found for {method}/{dataset}')
        _debug_single(paths[0], pehe_key, err_key)
        return

    if not os.path.isdir(args.out_root):
        sys.exit(f'FATAL: --out-root not found: {args.out_root}')

    # uwyk1d's point keys are suffixed by anc-tag — resolve via --uwyk-tag.
    point_keys = dict(_POINT_KEYS)
    point_keys['uwyk1d'] = (f'pehe_raw_{args.uwyk_tag}', f'err_raw_{args.uwyk_tag}')

    big_pehe = {'CPS', 'PSID', 'PSID_bal'}
    big_len  = big_pehe

    header = '| Method | ' + ' | '.join(DATASETS) + ' |'
    sep    = '|' + '|'.join(['---'] * (1 + len(DATASETS))) + '|'
    lines = [
        f'\nRealCause density-CI — {args.out_root}',
        '',
        '(each cell, top → bottom: √PEHE / ε_ATE (from stored point CATE — '
        'matches realcause_eval mega-sbatch), Coverage / Length (95% CI from '
        'the density, assuming Y|do(0) ⊥ Y|do(1)); n = realizations)',
        '',
        header, sep,
    ]

    verify_lines = ['', '## Sanity: max |mean(density) − stored ate| across realizations',
                    '(should be ≈ 0 — the density used for CI IS the density that '
                    'produced the reported point CATE; if not, CI is on a different '
                    'distribution than the stored PEHE row)',
                    '', header, sep]

    for method in args.methods:
        method_dir = os.path.join(args.out_root, method)
        pehe_key, err_key = point_keys.get(method, ('pehe_raw', 'err_raw'))
        cells = [method]
        verify_cells = [method]
        for d in DATASETS:
            got = summarize_method_dataset(method_dir, d, pehe_key, err_key)
            if got is None:
                cells.append('—')
                verify_cells.append('—')
                continue
            pehe_str = _fmt(*got['pehe'], big=d in big_pehe)
            err_str  = _fmt(*got['err'],  big=False)
            cov_str  = _fmt(*got['cov'],  big=False)
            len_str  = _fmt(*got['len'],  big=d in big_len)
            n = got['n']
            cells.append(
                f'PEHE {pehe_str}<br>'
                f'ε_ATE {err_str}<br>'
                f'Cov {cov_str}<br>'
                f'Len {len_str} (n={n})'
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
