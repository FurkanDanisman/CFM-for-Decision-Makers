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


# ── Method density loaders (per realization → per-query p_tau on shared grid).

def _load_1d_ptau_raw_on_grid(root_1d, method, dataset, r, tau_grid_raw):
    """1D method: reconstruct p(τ|x_q) from p_y0 / p_y1 marginals via
    outer-product enumeration + rasterization onto the shared tau_grid_raw.
    Uses effective_centers if present (dopfn's tail-adjusted centers)."""
    path = os.path.join(root_1d, method, dataset, f'{dataset}_r{r:03d}.npz')
    if not os.path.isfile(path): return None
    with np.load(path, allow_pickle=True) as z:
        p_y0 = np.asarray(z['p_y0_scaled'], dtype=np.float64)
        p_y1 = np.asarray(z['p_y1_scaled'], dtype=np.float64)
        edges = np.asarray(z['edges'], dtype=np.float64)
        y_scale = float(z['y_scale'])
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
    path = os.path.join(root_2d, method, ds_on_disk,
                         f'malc_ci_{malc_tag}_r{r:03d}.npz')
    if not os.path.isfile(path): return None
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
        # Need tau grid first — get μ, σ to size it.
        # We use a fake preliminary call to get μ_diff and σ.
        from true_ihdp import load_ihdp_truth
        ds = ds_obj[r][0]
        y_train_raw = np.asarray(ds.y_train, dtype=np.float64).reshape(-1)
        truth = load_ihdp_truth(r, causalpfn_dir, y_train_raw)
        scale = truth.y_rng / 2.0
        mu_diff = (truth.mu1_test_scaled - truth.mu0_test_scaled) * scale
        sigma_tau = np.sqrt(2.0) * float(truth.sigma_scaled) * scale
    elif dataset == 'ACIC':
        from true_acic import load_acic_truth
        ds = ds_obj[r][0]
        y_train_raw = np.asarray(ds.y_train, dtype=np.float64).reshape(-1)
        truth = load_acic_truth(r, causalpfn_dir, y_train_raw)
        scale = truth.y_rng / 2.0
        mu_diff = (truth.mu1_test_scaled - truth.mu0_test_scaled) * scale
        sigma_tau = np.sqrt(2.0) * float(truth.sigma_scaled) * scale
    else:
        raise NotImplementedError(
            f'Truth density for {dataset} not yet implemented — '
            f'needs 100-CSV MC-stack pipeline.')

    lo = float(mu_diff.min() - tau_pad_sigmas * sigma_tau)
    hi = float(mu_diff.max() + tau_pad_sigmas * sigma_tau)
    tau_grid = np.linspace(lo, hi, T)
    dtau     = tau_grid[1] - tau_grid[0]

    p_true_pq = np.stack([_gaussian(tau_grid, mu, sigma_tau) for mu in mu_diff])
    p_true_pq = _renorm(p_true_pq, dtau)
    true_cate_pq_raw = mu_diff.copy()          # point true CATE per query

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

    if args.dataset in _REALCAUSE_CSV:
        print(f'[note] {args.dataset} truth density from 100-CSV stack not yet '
              f'implemented; only IHDP + ACIC supported in this version.',
              file=sys.stderr)
        sys.exit(1)

    if args.dataset == 'IHDP':
        from benchmarks.data.ihdp import IHDPDataset
        ds_obj = IHDPDataset()
        n_default = 100
    elif args.dataset == 'ACIC':
        from benchmarks.data.acic2016 import ACIC2016Dataset
        ds_obj = ACIC2016Dataset()
        n_default = 10
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

    lines = [
        f'\nDensity metrics — {args.dataset} — tight T={args.T} raw τ grid',
        '',
        f'(truth = analytic Gaussian N(μ_1(x)−μ_0(x), 2σ²); '
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
