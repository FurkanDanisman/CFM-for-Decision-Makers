"""Dataset-generic density metrics: NLL / L2 / KL_fwd / KL_rev of
method p(τ|x_q) vs truth p_true(τ|x_q) on a tight raw τ grid.

Truth per dataset:
  IHDP  : analytic Gaussian via benchmarks/l2_ihdp/true_ihdp.py
          (μ_0(x), μ_1(x), σ shipped in ihdp_npci_1-100 NPZs; σ estimated
          per realization from training factual residuals). Under IHDP DGP
          Y_do_t | x ~ N(μ_t(x), σ²) with arms independent ⇒
          τ | x ~ N(μ_1 − μ_0, 2σ²) — closed form.
  ACIC  : analytic Gaussian via benchmarks/l2_acic/true_acic.py (same shape).
  CPS / PSID / PSID_bal:
          empirical MC truth from stacking 100 RealCause CSVs (shared X;
          each CSV is one noise draw of (Y_0, Y_1)). NOT IMPLEMENTED YET —
          this script prints a stub for those and exits gracefully.

Method densities:
  1D methods (cpfn1d, dopfn, uwyk1d) — reconstruct via p_y0 * flip(p_y1)
    marginals convolution (honors dopfn's `effective_centers` for tail atoms).
  2D methods MALC-B500 (cpfn2d, graph2d, dopfnbb) — read `p_taus_scaled`
    from malc_ci_B500_r<###>.npz, convert to raw via y_scale, interpolate.

Metrics per query (all in raw Y units):
  NLL     = -log p_est(τ_true(q))         (interp p_est at the point true CATE)
  L2      = sqrt(∫ (p_true − p_est)² dτ)
  KL_fwd  = ∫ p_true · log(p_true / p_est) dτ  (truth ‖ est)
  KL_rev  = ∫ p_est  · log(p_est  / p_true) dτ  (est ‖ truth)

Grid: shared per realization, tight (T=4001 by default) — covers every
query's ±6σ_τ tail.

Aggregation: per-realization means → per-cell mean ± SE. Output: one
markdown table (methods × datasets).

Usage:
    python realcause_eval/eval_density_metrics.py \\
        --dataset IHDP --root-1d $OUT_1D --root-2d $OUT_2D \\
        --causalpfn $CAUSALPFN --repo $REPO --out-md metrics_IHDP.md
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


DEFAULT_T = 4001            # very tight τ grid
_EPS = 1e-12
_ANALYTIC_GAUSSIAN = {'IHDP', 'ACIC'}
_REALCAUSE_CSV     = {'CPS', 'PSID', 'PSID_bal'}


# ── Basic math helpers ────────────────────────────────────────────────────
def _gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def _renorm(p, dx):
    p = np.clip(p, 0.0, None)
    s = p.sum(axis=-1, keepdims=True) * dx
    return p / np.where(s > 0, s, 1.0)


def _l2(f, g, dx):
    return np.sqrt(np.sum((f - g) ** 2, axis=-1) * dx)


def _kl(f, g, dx):
    f_ = np.clip(f, _EPS, None); g_ = np.clip(g, _EPS, None)
    return np.sum(f_ * np.log(f_ / g_), axis=-1) * dx


def _nll_pointwise(p_est_pq, tau_grid, y_true_pq):
    """NLL = −log p_est(τ_true) per query via linear interp of p_est at τ_true.
    Shapes: p_est_pq (N_q, T), tau_grid (T,), y_true_pq (N_q,). Returns (N_q,).
    """
    T = tau_grid.size
    dtau = float(tau_grid[1] - tau_grid[0])
    idx_f = (y_true_pq - tau_grid[0]) / dtau
    idx_lo = np.clip(np.floor(idx_f).astype(int), 0, T - 2)
    frac   = np.clip(idx_f - idx_lo, 0.0, 1.0)
    row    = np.arange(p_est_pq.shape[0])
    p_at_y = (1.0 - frac) * p_est_pq[row, idx_lo] + frac * p_est_pq[row, idx_lo + 1]
    return -np.log(np.clip(p_at_y, _EPS, None))


def _mean_se(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float('nan'), float('nan')
    m  = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else float('nan')
    return m, se


def _fmt(m, se, big=False):
    if not np.isfinite(m): return '—'
    if big:
        m_s  = f'{m:,.2f}'
        se_s = f'{se:,.2f}' if np.isfinite(se) else '—'
    else:
        m_s  = f'{m:.4f}'
        se_s = f'{se:.4f}' if np.isfinite(se) else '—'
    return f'{m_s} ± {se_s}'


# ── Truth loaders (per dataset) ───────────────────────────────────────────

def _truth_ihdp_gaussian(r, causalpfn_dir, ihdp_ds, tau_grid_raw):
    """Return (p_true_pq on tau_grid_raw, true_cate_pq_raw)."""
    from true_ihdp import load_ihdp_truth
    ds = ihdp_ds[r][0]
    y_train_raw = np.asarray(ds.y_train, dtype=np.float64).reshape(-1)
    truth = load_ihdp_truth(r, causalpfn_dir, y_train_raw)
    scale = truth.y_rng / 2.0
    mu_diff_raw = (truth.mu1_test_scaled - truth.mu0_test_scaled) * scale
    sigma_tau   = np.sqrt(2.0) * float(truth.sigma_scaled) * scale
    p_true = np.stack([_gaussian(tau_grid_raw, mu, sigma_tau) for mu in mu_diff_raw])
    dtau = float(tau_grid_raw[1] - tau_grid_raw[0])
    return _renorm(p_true, dtau), mu_diff_raw, sigma_tau


def _truth_acic_gaussian(r, causalpfn_dir, acic_ds, tau_grid_raw, cache_dir=None):
    """Analytic Gaussian truth for ACIC."""
    from true_acic import load_acic_truth
    ds = acic_ds[r][0]
    y_train_raw = np.asarray(ds.y_train, dtype=np.float64).reshape(-1)
    truth = load_acic_truth(r, cache_dir, y_train_raw) if cache_dir else \
            load_acic_truth(r, causalpfn_dir, y_train_raw)
    scale = truth.y_rng / 2.0
    mu_diff_raw = (truth.mu1_test_scaled - truth.mu0_test_scaled) * scale
    sigma_tau   = np.sqrt(2.0) * float(truth.sigma_scaled) * scale
    p_true = np.stack([_gaussian(tau_grid_raw, mu, sigma_tau) for mu in mu_diff_raw])
    dtau = float(tau_grid_raw[1] - tau_grid_raw[0])
    return _renorm(p_true, dtau), mu_diff_raw, sigma_tau


# Cache of stacked (y0, y1, X_pool) across the 100 RealCause CSVs — one entry
# per dataset. Loaded once, reused across realizations in the same run.
_RC_STACK_CACHE = {}


def _load_realcause_stack(dataset, causalpfn_dir):
    """Return (X_pool, y0_all, y1_all) stacked across the 100 sample CSVs.

    X_pool shape (N_pool, F)   — shared X across all 100 CSVs (fitted-SCM
                                  fixes X; only Y varies).
    y0_all shape (N_pool, 100) — K=100 potential-outcome draws for do(0).
    y1_all shape (N_pool, 100) — same for do(1).
    """
    if dataset in _RC_STACK_CACHE:
        return _RC_STACK_CACHE[dataset]
    import pandas as pd
    prefix_map = {'CPS': 'lalonde_cps_sample',
                  'PSID': 'lalonde_psid_sample',
                  'PSID_bal': 'lalonde_psid_sample'}   # PSID_bal shares PSID CSVs
    prefix = prefix_map[dataset]
    # Candidate directories: try common layouts.
    cand_dirs = [
        os.path.join(causalpfn_dir, 'benchmarks', 'realcause_datasets'),
        os.path.join(causalpfn_dir, 'src', 'benchmarks', 'realcause_datasets'),
        os.path.join(causalpfn_dir, 'realcause_datasets'),
    ]
    csv_dir = next((d for d in cand_dirs if os.path.isdir(d)), None)
    if csv_dir is None:
        raise FileNotFoundError(
            f'RealCause CSVs not found under any of: {cand_dirs}')
    # Feature columns are all columns EXCEPT the known outcome/treatment ones.
    # RealCause CSVs use plain feature names (age, education, black, ...) not
    # x1..x8 — the agent's initial report used generic 'x*' shorthand.
    _NON_X = {'t', 'y', 'y0', 'y1', 'ite', 'ycf', 'mu0', 'mu1'}
    y0_all = []; y1_all = []; X_pool = None
    for k in range(100):
        path = os.path.join(csv_dir, f'{prefix}{k}.csv')
        df = pd.read_csv(path)
        x_cols = [c for c in df.columns if c not in _NON_X]
        if not x_cols:
            raise RuntimeError(f'no feature columns in {path}; got {list(df.columns)}')
        if X_pool is None:
            X_pool = df[x_cols].values.astype(np.float64)
            _saved_x_cols = x_cols
        else:
            assert X_pool.shape == df[x_cols].shape, (
                f'X shape drift in {path}: {df[x_cols].shape} vs {X_pool.shape}')
        y0_all.append(df['y0'].values.astype(np.float64))
        y1_all.append(df['y1'].values.astype(np.float64))
    y0_all = np.stack(y0_all, axis=1)   # (N_pool, K=100)
    y1_all = np.stack(y1_all, axis=1)
    _RC_STACK_CACHE[dataset] = (X_pool, y0_all, y1_all)
    return X_pool, y0_all, y1_all


def _truth_realcause_kde(dataset, r, causalpfn_dir, ds_obj, tau_grid_raw,
                          kde_h_scale=1.0):
    """Empirical truth density for CPS / PSID / PSID_bal.

    For each query x_q in realization r's X_test, look up its row in the
    shared X_pool (via tuple-hash), extract the K=100 (y0, y1) draws at
    that row, form τ samples, and KDE onto tau_grid_raw with per-query
    Silverman bandwidth (h = 1.06·σ·K^(-1/5)).

    Returns (p_true_pq, true_cate_pq_raw, mean_sigma_tau).
    """
    _, y0_all, y1_all = _load_realcause_stack(dataset, causalpfn_dir)
    tau_all = y1_all - y0_all                   # (N_pool, K=100)

    ds = ds_obj[r][0]
    true_cate = np.asarray(ds.true_cate, dtype=np.float64).reshape(-1)

    # Match test queries to pool rows via ITE identity (robust to X standardization).
    test_idx = _match_by_ite(dataset, r, causalpfn_dir, ds)
    keep = test_idx >= 0
    if not keep.all():
        n_miss = int((~keep).sum())
        print(f'  [warn] {dataset} r={r}: {n_miss}/{len(test_idx)} '
              f'test rows unmatched via ITE; dropping', file=sys.stderr)
    test_idx = test_idx[keep]
    true_cate = true_cate[keep]
    tau_samples = tau_all[test_idx]              # (N_q, K)
    N_q, K = tau_samples.shape

    # Per-query KDE with Silverman-rule bandwidth on shared τ grid.
    sigmas = tau_samples.std(axis=1, ddof=1)     # (N_q,)
    # Guard tiny σ (degenerate) with a fraction of dtau; use dtau=1e-3 · tau range.
    tau_range = float(tau_grid_raw[-1] - tau_grid_raw[0])
    sigma_floor = 1e-3 * tau_range
    h = np.maximum(1.06 * sigmas * (K ** (-1.0 / 5.0)) * kde_h_scale, sigma_floor)  # (N_q,)

    dtau = float(tau_grid_raw[1] - tau_grid_raw[0])
    p_true = np.empty((N_q, tau_grid_raw.size), dtype=np.float64)
    for q in range(N_q):
        z = (tau_grid_raw[:, None] - tau_samples[q, None, :]) / h[q]   # (T, K)
        p_true[q] = np.exp(-0.5 * z ** 2).sum(axis=1) / (K * h[q] * np.sqrt(2.0 * np.pi))
    p_true = _renorm(p_true, dtau)
    return p_true, true_cate, float(np.mean(sigmas)) * np.sqrt(2.0)


# ── Method density loaders (per realization → per-query p_tau on shared grid).

def _match_by_ite(dataset, r, causalpfn_dir, ds):
    """Match test queries to shared-pool rows via ITE (true_cate) identity.

    Robust to any X standardization the RealCause loader may do — because
    the CSV `ite` column (real-valued τ per row) is raw and matches
    ds.true_cate byte-for-byte modulo float precision.

    Returns (test_idx, ite_pool_len). test_idx: (N_test,) int64 array;
    entries with no match are -1 (caller drops).
    """
    import pandas as pd
    prefix_map = {'CPS': 'lalonde_cps_sample',
                  'PSID': 'lalonde_psid_sample',
                  'PSID_bal': 'lalonde_psid_sample'}
    cand_dirs = [
        os.path.join(causalpfn_dir, 'benchmarks', 'realcause_datasets'),
        os.path.join(causalpfn_dir, 'src', 'benchmarks', 'realcause_datasets'),
        os.path.join(causalpfn_dir, 'realcause_datasets'),
    ]
    csv_dir = next((d for d in cand_dirs if os.path.isdir(d)), None)
    if csv_dir is None:
        raise FileNotFoundError(f'RealCause CSVs not under any of: {cand_dirs}')
    df_r = pd.read_csv(os.path.join(csv_dir, f'{prefix_map[dataset]}{r}.csv'))
    ite_pool = df_r['ite'].values.astype(np.float64)
    ite_test = np.asarray(ds.true_cate, dtype=np.float64).reshape(-1)
    # Round to 8 decimals to handle any tiny float precision drift.
    _key = lambda v: round(float(v), 8)
    ite_hash = {}
    for i, v in enumerate(ite_pool):
        ite_hash.setdefault(_key(v), []).append(i)
    idx = np.array([ite_hash.get(_key(t), [-1])[0] for t in ite_test], dtype=np.int64)
    return idx


def _resolve_realization_npz(root, method, dataset_dir_name, r, prefix='',
                                suffix_dataset_name=None):
    """Try both 3-digit and 2-digit realization padding.

    cpfn1d/cpfn2d use r<r:02d> filenames for ACIC while all other emitters
    use r<r:03d> uniformly. Search both patterns.

    prefix='' → filenames like <DATASET>_r<###>.npz (inline layout).
    prefix='malc_ci_{tag}_' → tagged MALC files (2D MALC path).
    """
    ds_in_name = suffix_dataset_name or dataset_dir_name
    for pad in (3, 2):
        rstr = f'{r:0{pad}d}'
        if prefix:
            fname = f'{prefix}r{rstr}.npz'
        else:
            fname = f'{ds_in_name}_r{rstr}.npz'
        path = os.path.join(root, method, dataset_dir_name, fname)
        if os.path.isfile(path):
            return path
    return None


def _load_1d_ptau_raw_on_grid(root_1d, method, dataset, r, tau_grid_raw):
    """1D method: reconstruct p(τ|x_q) from p_y0 / p_y1 marginals via
    outer-product enumeration + rasterization onto the shared tau_grid_raw.
    Uses effective_centers if present (dopfn's tail-adjusted centers)."""
    path = _resolve_realization_npz(root_1d, method, dataset, r)
    if not path: return None
    with np.load(path, allow_pickle=True) as z:
        p_y0 = np.asarray(z['p_y0_scaled'], dtype=np.float64)
        p_y1 = np.asarray(z['p_y1_scaled'], dtype=np.float64)
        edges = np.asarray(z['edges'], dtype=np.float64)
        y_scale = float(z['y_scale'])
        K = p_y0.shape[1]
        # UWYK schema: (K_bars + 2 tail atoms) columns but only K_bars+1 edges.
        # Tail atoms live at per-query offsets outside the bar grid; without
        # sL/sR the tail centers can't be reconstructed. Trim tails and drop
        # their (small) mass — mild bias, avoids shape mismatch downstream.
        if K == edges.size + 1:
            p_y0 = p_y0[:, 1:-1]
            p_y1 = p_y1[:, 1:-1]
            K = p_y0.shape[1]
        centers_scaled = (np.asarray(z['effective_centers'], dtype=np.float64)
                          if 'effective_centers' in z.files
                          else 0.5 * (edges[:-1] + edges[1:]))
    p_y0 = p_y0 / p_y0.sum(axis=-1, keepdims=True).clip(min=_EPS)
    p_y1 = p_y1 / p_y1.sum(axis=-1, keepdims=True).clip(min=_EPS)
    centers_raw = centers_scaled * y_scale
    T = tau_grid_raw.size
    dtau = float(tau_grid_raw[1] - tau_grid_raw[0])
    tau_min = float(tau_grid_raw[0])
    tau_pairs = (centers_raw[None, :] - centers_raw[:, None]).ravel()      # (K²,)
    idx = np.clip(np.round((tau_pairs - tau_min) / dtau).astype(int), 0, T - 1)
    N_q = p_y0.shape[0]
    p_tau_mass = np.zeros((N_q, T), dtype=np.float64)
    for q in range(N_q):
        mass = np.outer(p_y0[q], p_y1[q]).ravel()
        np.add.at(p_tau_mass[q], idx, mass)
    return _renorm(p_tau_mass / dtau, dtau)


def _load_2d_malc_ptau_raw_on_grid(root_2d, method, dataset, r, tau_grid_raw,
                                     malc_tag='B500'):
    """2D method (MALC-smoothed): load p_taus_scaled, convert to raw,
    interpolate onto shared grid."""
    ds_on_disk = dataset
    if method == 'dopfnbb' and dataset == 'PSID_bal':
        ds_on_disk = 'PSIDbal'
    path = _resolve_realization_npz(root_2d, method, ds_on_disk, r,
                                     prefix=f'malc_ci_{malc_tag}_')
    if not path: return None
    with np.load(path, allow_pickle=True) as z:
        p_tau_scaled = np.asarray(z['p_taus_scaled'], dtype=np.float64)
        tau_scaled   = np.asarray(z['tau_scaled'],    dtype=np.float64)
        y_scale      = float(z['y_scale'])
    tau_native_raw = tau_scaled * y_scale
    p_tau_native_raw = p_tau_scaled / max(y_scale, _EPS)
    N_q = p_tau_scaled.shape[0]
    out = np.empty((N_q, tau_grid_raw.size), dtype=np.float64)
    for q in range(N_q):
        out[q] = np.interp(tau_grid_raw, tau_native_raw, p_tau_native_raw[q],
                            left=0.0, right=0.0)
    dtau = float(tau_grid_raw[1] - tau_grid_raw[0])
    return _renorm(out, dtau)


# ── Per-realization pipeline ─────────────────────────────────────────────

def evaluate_realization(r, dataset, causalpfn_dir, ds_obj,
                          root_1d, root_2d, methods_1d, methods_2d,
                          T=DEFAULT_T, tau_pad_sigmas=6.0):
    if dataset == 'IHDP':
        from true_ihdp import load_ihdp_truth
        ds = ds_obj[r][0]
        y_train_raw = np.asarray(ds.y_train, dtype=np.float64).reshape(-1)
        truth = load_ihdp_truth(r, causalpfn_dir, y_train_raw)
        scale = truth.y_rng / 2.0
        mu_diff = (truth.mu1_test_scaled - truth.mu0_test_scaled) * scale
        sigma_tau = np.sqrt(2.0) * float(truth.sigma_scaled) * scale
        _mode = 'gaussian'
    elif dataset == 'ACIC':
        from true_acic import load_acic_truth
        ds = ds_obj[r][0]
        y_train_raw = np.asarray(ds.y_train, dtype=np.float64).reshape(-1)
        # Signature: load_acic_truth(r, y_train_full, seed=42, test_ratio=0.1, cache_dir=None)
        truth = load_acic_truth(r, y_train_raw)
        scale = truth.y_rng / 2.0
        mu_diff = (truth.mu1_test_scaled - truth.mu0_test_scaled) * scale
        sigma_tau = np.sqrt(2.0) * float(truth.sigma_scaled) * scale
        _mode = 'gaussian'
    elif dataset in _REALCAUSE_CSV:
        _mode = 'kde'      # tau grid + p_true built inside _truth_realcause_kde
    else:
        raise NotImplementedError(f'Unknown dataset {dataset}')

    if _mode == 'gaussian':
        lo = float(mu_diff.min() - tau_pad_sigmas * sigma_tau)
        hi = float(mu_diff.max() + tau_pad_sigmas * sigma_tau)
        tau_grid = np.linspace(lo, hi, T)
        dtau     = tau_grid[1] - tau_grid[0]
        p_true_pq = np.stack([_gaussian(tau_grid, mu, sigma_tau) for mu in mu_diff])
        p_true_pq = _renorm(p_true_pq, dtau)
        true_cate_pq_raw = mu_diff.copy()          # point true CATE per query
    else:
        # KDE truth for RealCause datasets (CPS/PSID/PSID_bal).
        _, y0_all, y1_all = _load_realcause_stack(dataset, causalpfn_dir)
        tau_all = y1_all - y0_all                # (N_pool, K=100)
        ds = ds_obj[r][0]
        # Match via ITE (raw τ). Robust to X standardization by the loader.
        test_idx_full = _match_by_ite(dataset, r, causalpfn_dir, ds)
        keep_mask = test_idx_full >= 0
        if not keep_mask.all():
            print(f'  [warn] {dataset} r={r}: {int((~keep_mask).sum())}/'
                  f'{len(test_idx_full)} test rows unmatched via ITE',
                  file=sys.stderr)
        test_idx = test_idx_full[keep_mask]
        tau_samples = tau_all[test_idx]           # (N_q_matched, K)
        true_cate_pq_raw = np.asarray(ds.true_cate, dtype=np.float64).reshape(-1)[keep_mask]
        # Grid: pad by ±3σ of stacked samples.
        s_all = tau_samples.std(ddof=1)
        lo = float(tau_samples.min() - 3.0 * s_all)
        hi = float(tau_samples.max() + 3.0 * s_all)
        tau_grid = np.linspace(lo, hi, T)
        dtau     = tau_grid[1] - tau_grid[0]
        # KDE per matched query.
        N_q_m, K = tau_samples.shape
        sigmas = tau_samples.std(axis=1, ddof=1)
        sigma_floor = 1e-3 * (tau_grid[-1] - tau_grid[0])
        h = np.maximum(1.06 * sigmas * (K ** (-1.0 / 5.0)), sigma_floor)
        p_true_pq = np.empty((N_q_m, T), dtype=np.float64)
        for q in range(N_q_m):
            z = (tau_grid[:, None] - tau_samples[q, None, :]) / h[q]
            p_true_pq[q] = np.exp(-0.5 * z ** 2).sum(axis=1) / (K * h[q] * np.sqrt(2.0 * np.pi))
        p_true_pq = _renorm(p_true_pq, dtau)
    # keep_mask is None for IHDP/ACIC (all queries used).
    if _mode == 'gaussian':
        keep_mask = None

    results = {}
    for method, kind in [(m, '1d') for m in methods_1d] + [(m, '2d') for m in methods_2d]:
        try:
            if kind == '1d':
                p_est_pq = _load_1d_ptau_raw_on_grid(root_1d, method, dataset, r, tau_grid)
            else:
                p_est_pq = _load_2d_malc_ptau_raw_on_grid(root_2d, method, dataset, r,
                                                            tau_grid, malc_tag='B500')
        except Exception as e:
            print(f'  [warn] r={r:03d} {method}: {e}', file=sys.stderr)
            p_est_pq = None
        if p_est_pq is None:
            results[method] = None; continue
        # Filter method density to same subset as truth (RealCause KDE path drops
        # unmatched queries; IHDP/ACIC uses all queries → keep_mask is None).
        if keep_mask is not None:
            p_est_pq = p_est_pq[keep_mask]
        if p_est_pq.shape[0] != p_true_pq.shape[0]:
            print(f'  [warn] r={r:03d} {method}: shape mismatch after keep_mask '
                  f'(est {p_est_pq.shape[0]} vs truth {p_true_pq.shape[0]})',
                  file=sys.stderr)
            continue

        nll_pq    = _nll_pointwise(p_est_pq, tau_grid, true_cate_pq_raw)
        l2_pq     = _l2(p_true_pq, p_est_pq, dtau)
        kl_fwd_pq = _kl(p_true_pq, p_est_pq, dtau)
        kl_rev_pq = _kl(p_est_pq,  p_true_pq, dtau)

        results[method] = dict(
            nll_mean    = float(np.mean(nll_pq)),
            l2_mean     = float(np.mean(l2_pq)),
            kl_fwd_mean = float(np.mean(kl_fwd_pq)),
            kl_rev_mean = float(np.mean(kl_rev_pq)),
        )
    return results


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True,
                    choices=['IHDP', 'ACIC', 'CPS', 'PSID', 'PSID_bal'])
    ap.add_argument('--root-1d', required=True)
    ap.add_argument('--root-2d', required=True)
    ap.add_argument('--causalpfn', required=True)
    ap.add_argument('--repo', required=True)
    ap.add_argument('--methods-1d', nargs='+', default=['cpfn1d', 'dopfn', 'uwyk1d'])
    ap.add_argument('--methods-2d', nargs='+', default=['cpfn2d', 'graph2d', 'dopfnbb'])
    ap.add_argument('--n-realizations', type=int, default=None,
                    help='Cap on realizations (default: dataset default).')
    ap.add_argument('--T', type=int, default=DEFAULT_T,
                    help='τ-grid resolution (default 4001 — very tight).')
    ap.add_argument('--out-md', default=None)
    args = ap.parse_args()

    for p in (args.repo, os.path.join(args.repo, 'benchmarks', 'l2_ihdp'),
              os.path.join(args.repo, 'benchmarks', 'l2_acic')):
        if p not in sys.path:
            sys.path.insert(0, p)
    sys.path.insert(0, args.causalpfn)
    sys.path.insert(0, os.path.join(args.causalpfn, 'src'))

    # CausalPFN's benchmarks package pulls in causalpfn.causal_estimator →
    # `import faiss`. faiss isn't installed in the CPU venv on some nodes.
    # Stub it out — benchmarks.data.{ihdp,acic2016,realcause} loaders don't
    # actually use faiss, so an empty module bypasses the import cascade.
    try:
        import faiss  # noqa: F401
    except ImportError:
        import types as _types
        sys.modules['faiss'] = _types.ModuleType('faiss')

    if args.dataset == 'IHDP':
        from benchmarks import IHDPDataset
        ds_obj = IHDPDataset()
        n_default = 100
    elif args.dataset == 'ACIC':
        from benchmarks import ACIC2016Dataset
        ds_obj = ACIC2016Dataset()
        n_default = 10
    elif args.dataset in ('CPS',):
        from benchmarks import RealCauseLalondeCPSDataset
        ds_obj = RealCauseLalondeCPSDataset()
        n_default = 100
    elif args.dataset in ('PSID', 'PSID_bal'):
        # PSID and PSID_bal share the loader; balancing (train subsample) is
        # applied by the method-side pipeline, doesn't affect X_test or truth.
        from benchmarks import RealCauseLalondePSIDDataset
        ds_obj = RealCauseLalondePSIDDataset()
        n_default = 100
    else:
        raise NotImplementedError(args.dataset)

    n_realizations = args.n_realizations or n_default
    methods_1d = list(args.methods_1d); methods_2d = list(args.methods_2d)
    all_methods = methods_1d + methods_2d
    print(f'[bootstrap] dataset={args.dataset}  n_realizations={n_realizations}  '
          f'T={args.T}  methods_1d={methods_1d}  methods_2d={methods_2d}', flush=True)

    per_r = []
    t0 = time.time()
    for r in range(n_realizations):
        tr = time.time()
        try:
            res = evaluate_realization(r, args.dataset, args.causalpfn, ds_obj,
                                         args.root_1d, args.root_2d,
                                         methods_1d, methods_2d, T=args.T)
        except Exception as e:
            print(f'  [warn] r={r:03d}: {e}', file=sys.stderr); continue
        per_r.append(res)
        print(f'  [{time.strftime("%H:%M:%S")}] r={r:03d}  ({time.time() - tr:.1f}s)',
              flush=True)

    _truth_kind = ('analytic Gaussian N(μ_1(x)−μ_0(x), 2σ²)'
                    if args.dataset in _ANALYTIC_GAUSSIAN
                    else 'empirical KDE from 100 RealCause CSVs (K=100 MC pairs per unit)')
    lines = [
        f'\nDensity metrics — {args.dataset} — tight T={args.T} raw τ grid',
        '',
        f'(truth = {_truth_kind}; '
        f'metrics per query averaged, then averaged across {n_realizations} '
        f'realizations. Lower = better.)',
        '',
        '| Method | NLL | L2 | KL_fwd (truth‖est) | KL_rev (est‖truth) |',
        '|---|---|---|---|---|',
    ]
    for method in all_methods:
        vals = {k: [] for k in ('nll_mean', 'l2_mean', 'kl_fwd_mean', 'kl_rev_mean')}
        for res in per_r:
            m = res.get(method) if res else None
            if not m: continue
            for k in vals: vals[k].append(m[k])
        cells = [method]
        for k in ('nll_mean', 'l2_mean', 'kl_fwd_mean', 'kl_rev_mean'):
            cells.append(_fmt(*_mean_se(vals[k])))
        lines.append('| ' + ' | '.join(cells) + ' |')

    md = '\n'.join(lines) + '\n'
    print(md)
    if args.out_md:
        with open(args.out_md, 'w') as f: f.write(md)
        print(f'wrote {args.out_md}', flush=True)
    print(f'[done] total={time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
