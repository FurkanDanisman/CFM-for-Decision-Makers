"""Density-L2 / KL metrics for RealCause IHDP: method p(τ|x_q) vs analytic truth.

Truth (per query, per realization): under IHDP's DGP,
    τ | x_q ~ N(μ_1(x_q) − μ_0(x_q),  2σ²)
with μ_0, μ_1 shipped in ihdp_npci_1-100 NPZs and σ estimated from
training-side factual residuals (see benchmarks/l2_ihdp/true_ihdp.py).

For each realization r:
  1. Load truth (μ_1 − μ_0 per query, σ scalar) in RAW Y units.
  2. Build a shared tight raw τ grid centered on 0 wide enough to cover
     every query's ±6σ_τ tail.
  3. Evaluate p_true(τ|x_q) as Gaussian on that grid.
  4. Also compute the truth ATE density as N(mean(μ_1 − μ_0), 2σ²) — the
     closed-form W2 barycenter of same-σ Gaussians.
  5. For each method, load its saved per-query p(τ|x_q) (1D: reconstruct
     via marginals convolution; 2D-MALC: read p_taus_scaled from
     malc_ci_B500 NPZ), interpolate onto the shared raw τ grid,
     renormalize.
  6. Compute per-query L2, KL_fwd (truth ‖ est), KL_rev (est ‖ truth).
  7. For ATE: form the method's ATE via 1D W2 barycenter over queries,
     compare to truth ATE density on the same grid.
  8. Aggregate to per-realization means.

Prints per-method markdown table (CATE + ATE columns of L2, KL_fwd, KL_rev
each averaged across realizations with SE).

Usage:
    python realcause_eval/eval_density_metrics_ihdp.py \\
        --root-1d /scratch/.../results_1d_density \\
        --root-2d /scratch/.../results_rc_2d_density \\
        --causalpfn /scratch/.../external/causalpfn \\
        --repo /scratch/.../R-PFN --out-md ihdp_density_metrics.md
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np


# ── Shared grid config ───────────────────────────────────────────────────
DEFAULT_T = 2001     # τ-grid resolution (tight)
_EPS = 1e-12


def _import_repo(repo):
    for p in (repo, os.path.join(repo, 'MALC', 'Optimal_Transport'),
              os.path.join(repo, 'benchmarks', 'l2_ihdp')):
        if p not in sys.path:
            sys.path.insert(0, p)


def _gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def _renorm(p, dx):
    """Normalize each row of p (last axis) to integrate to 1 on step dx."""
    p = np.clip(p, 0.0, None)
    s = p.sum(axis=-1, keepdims=True) * dx
    return p / np.where(s > 0, s, 1.0)


def _l2(f, g, dx):
    return np.sqrt(np.sum((f - g) ** 2, axis=-1) * dx)


def _kl(f, g, dx):
    f_ = np.clip(f, _EPS, None); g_ = np.clip(g, _EPS, None)
    return np.sum(f_ * np.log(f_ / g_), axis=-1) * dx


def _mean_se(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float('nan'), float('nan')
    m  = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else float('nan')
    return m, se


def _fmt(m, se):
    if not np.isfinite(m): return '—'
    return f'{m:.4f} ± {se:.4f}' if np.isfinite(se) else f'{m:.4f}'


# ── Method-density loaders (per realization) ─────────────────────────────

def _load_1d_ptau_raw_on_grid(root_1d, method, r, tau_grid_raw):
    """1D method: reconstruct p(τ|x_q) from p_y0 / p_y1 marginals via
    convolution using effective_centers if present, then rasterize onto
    the shared tau_grid_raw (nearest-bin assignment)."""
    path = os.path.join(root_1d, method, 'IHDP', f'IHDP_r{r:03d}.npz')
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
    centers_raw = centers_scaled * y_scale                           # (K,)
    tau_pairs = (centers_raw[None, :] - centers_raw[:, None]).ravel()  # (K²,)
    T = tau_grid_raw.size
    dtau = float(tau_grid_raw[1] - tau_grid_raw[0])
    tau_min = float(tau_grid_raw[0])
    idx = np.clip(np.round((tau_pairs - tau_min) / dtau).astype(int), 0, T - 1)
    N_q = p_y0.shape[0]
    p_tau_mass = np.zeros((N_q, T), dtype=np.float64)
    for q in range(N_q):
        mass = np.outer(p_y0[q], p_y1[q]).ravel()
        np.add.at(p_tau_mass[q], idx, mass)
    return p_tau_mass / dtau     # mass → density


def _load_2d_malc_ptau_raw_on_grid(root_2d, method, r, tau_grid_raw, malc_tag='B500'):
    """2D method (MALC-smoothed): load p_taus_scaled from
    malc_ci_{tag}_r<###>.npz, convert to raw, interpolate onto shared grid."""
    path = os.path.join(root_2d, method, 'IHDP', f'malc_ci_{malc_tag}_r{r:03d}.npz')
    if not os.path.isfile(path): return None
    with np.load(path, allow_pickle=True) as z:
        p_tau_scaled = np.asarray(z['p_taus_scaled'], dtype=np.float64)   # (N_q, 401)
        tau_scaled   = np.asarray(z['tau_scaled'],    dtype=np.float64)
        y_scale      = float(z['y_scale'])
    tau_native_raw = tau_scaled * y_scale
    p_tau_native_raw = p_tau_scaled / max(y_scale, _EPS)
    # Linear interp per query onto tau_grid_raw (both grids are uniform).
    N_q = p_tau_scaled.shape[0]
    out = np.empty((N_q, tau_grid_raw.size), dtype=np.float64)
    for q in range(N_q):
        out[q] = np.interp(tau_grid_raw, tau_native_raw, p_tau_native_raw[q],
                            left=0.0, right=0.0)
    return out


# ── Main pipeline ─────────────────────────────────────────────────────────

def evaluate_realization(r, causalpfn_dir, ihdp_dataset,
                          root_1d, root_2d, methods_1d, methods_2d,
                          bary_fn, T=DEFAULT_T, tau_pad_sigmas=6.0):
    """Return dict method → {cate: dict, ate: dict} of metric per-realization means."""
    from true_ihdp import load_ihdp_truth
    # Load y_train_full for this realization to fix the y_rng scaling.
    ds = ihdp_dataset[r][0]
    y_train_raw = np.asarray(ds.y_train, dtype=np.float64).reshape(-1)
    truth = load_ihdp_truth(r, causalpfn_dir, y_train_raw)

    # Convert truth from SCALED to RAW: raw = (scaled + 1) * y_rng/2 + y_min.
    # For a difference (mu_1 - mu_0), the shift cancels: diff_raw = diff_scaled * y_rng/2.
    scale_s2r = truth.y_rng / 2.0
    mu_diff_raw = (truth.mu1_test_scaled - truth.mu0_test_scaled) * scale_s2r
    sigma_raw   = float(truth.sigma_scaled) * scale_s2r
    tau_sigma_raw = np.sqrt(2.0) * sigma_raw

    # Shared raw τ grid: cover every query's mean ± tau_pad_sigmas * σ_τ.
    lo = float(mu_diff_raw.min() - tau_pad_sigmas * tau_sigma_raw)
    hi = float(mu_diff_raw.max() + tau_pad_sigmas * tau_sigma_raw)
    tau_grid = np.linspace(lo, hi, T)
    dtau     = tau_grid[1] - tau_grid[0]

    # Truth per query on grid.
    p_true_pq = np.stack([_gaussian(tau_grid, mu, tau_sigma_raw) for mu in mu_diff_raw])
    p_true_pq = _renorm(p_true_pq, dtau)                              # (N_q, T)
    # Truth ATE = W2-barycenter of same-σ Gaussians = N(mean(mu_diff), σ_τ).
    p_true_ate = _gaussian(tau_grid, mu_diff_raw.mean(), tau_sigma_raw)
    p_true_ate = _renorm(p_true_ate[None, :], dtau)[0]

    results = {}
    for method, kind in [(m, '1d') for m in methods_1d] + [(m, '2d') for m in methods_2d]:
        try:
            if kind == '1d':
                p_est_pq = _load_1d_ptau_raw_on_grid(root_1d, method, r, tau_grid)
            else:
                p_est_pq = _load_2d_malc_ptau_raw_on_grid(root_2d, method, r, tau_grid, malc_tag='B500')
        except Exception as e:
            print(f'  [warn] r={r:03d} {method}: {e}', file=sys.stderr)
            p_est_pq = None
        if p_est_pq is None:
            results[method] = None
            continue
        p_est_pq = _renorm(p_est_pq, dtau)                            # (N_q, T)

        # ── CATE per-query metrics
        l2_pq     = _l2(p_true_pq, p_est_pq, dtau)                    # (N_q,)
        kl_fwd_pq = _kl(p_true_pq, p_est_pq, dtau)
        kl_rev_pq = _kl(p_est_pq,  p_true_pq, dtau)

        # ── ATE: W2 barycenter of method's per-query densities → compare to truth ATE.
        p_est_ate = bary_fn(p_est_pq, tau_grid, n_tau=4001)
        p_est_ate = _renorm(p_est_ate[None, :], dtau)[0]
        l2_ate     = float(_l2(p_true_ate[None, :], p_est_ate[None, :], dtau)[0])
        kl_fwd_ate = float(_kl(p_true_ate[None, :], p_est_ate[None, :], dtau)[0])
        kl_rev_ate = float(_kl(p_est_ate[None, :],  p_true_ate[None, :], dtau)[0])

        results[method] = dict(
            cate_l2_mean     = float(l2_pq.mean()),
            cate_kl_fwd_mean = float(kl_fwd_pq.mean()),
            cate_kl_rev_mean = float(kl_rev_pq.mean()),
            ate_l2           = l2_ate,
            ate_kl_fwd       = kl_fwd_ate,
            ate_kl_rev       = kl_rev_ate,
        )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root-1d', required=True)
    ap.add_argument('--root-2d', required=True)
    ap.add_argument('--causalpfn', required=True,
                    help='CausalPFN repo root (needs benchmarks/IHDP/ihdp_npci_1-100.*.npz).')
    ap.add_argument('--repo', required=True,
                    help='R-PFN repo root.')
    ap.add_argument('--methods-1d', nargs='+', default=['cpfn1d', 'dopfn', 'uwyk1d'])
    ap.add_argument('--methods-2d', nargs='+', default=['cpfn2d', 'graph2d', 'dopfnbb'])
    ap.add_argument('--n-realizations', type=int, default=100)
    ap.add_argument('--T', type=int, default=DEFAULT_T,
                    help='τ-grid resolution (default 2001).')
    ap.add_argument('--out-md', default=None)
    args = ap.parse_args()

    _import_repo(args.repo)
    # Also expose IHDPDataset via benchmarks.data (CausalPFN convention).
    sys.path.insert(0, args.causalpfn)
    sys.path.insert(0, os.path.join(args.causalpfn, 'src'))
    from benchmarks.data.ihdp import IHDPDataset
    from ot_barycenter import wasserstein_barycenter_1d

    ihdp_ds = IHDPDataset()
    methods_1d = list(args.methods_1d)
    methods_2d = list(args.methods_2d)
    all_methods = methods_1d + methods_2d
    print(f'[bootstrap] n_realizations={args.n_realizations}  T={args.T}  '
          f'methods_1d={methods_1d}  methods_2d={methods_2d}', flush=True)

    per_r = []           # list of dicts (method → metrics)
    t0 = time.time()
    for r in range(args.n_realizations):
        tr = time.time()
        res = evaluate_realization(r, args.causalpfn, ihdp_ds,
                                     args.root_1d, args.root_2d,
                                     methods_1d, methods_2d,
                                     wasserstein_barycenter_1d,
                                     T=args.T)
        per_r.append(res)
        print(f'  [{time.strftime("%H:%M:%S")}] r={r:03d}  ({time.time() - tr:.1f}s)',
              flush=True)

    # ── Aggregate: per method, mean±SE across realizations.
    lines = [
        '\nIHDP density metrics — method p(τ|x_q) vs analytic truth N(μ_1−μ_0, 2σ²)',
        '',
        '(all numbers averaged over 100 realizations; per-query means for CATE, '
        'single value per realization for ATE via W2 barycenter over queries)',
        '',
        '| Method | CATE L2 | CATE KL_fwd | CATE KL_rev | ATE L2 | ATE KL_fwd | ATE KL_rev |',
        '|---|---|---|---|---|---|---|',
    ]
    for method in all_methods:
        vals = {k: [] for k in ('cate_l2_mean', 'cate_kl_fwd_mean', 'cate_kl_rev_mean',
                                 'ate_l2', 'ate_kl_fwd', 'ate_kl_rev')}
        for res in per_r:
            m = res.get(method) if res else None
            if not m: continue
            for k in vals: vals[k].append(m[k])
        cells = [method]
        for k in ('cate_l2_mean', 'cate_kl_fwd_mean', 'cate_kl_rev_mean',
                  'ate_l2', 'ate_kl_fwd', 'ate_kl_rev'):
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
