"""Plot truth vs each method's estimated p(τ|x_q) for a few example queries.

Generates one PNG per dataset with a 6-row × N-col grid: one row per method
(cpfn1d, dopfn, uwyk1d, cpfn2d, graph2d, dopfnbb), one column per query.
For each subplot: truth density overlaid on the method's estimated density
(both on the same tight τ grid, raw Y units). Vertical dashed line at each
query's true CATE.

Truth depends on dataset:
  IHDP / ACIC : analytic Gaussian N(μ_1(x)−μ_0(x), 2σ²).
  CPS / PSID / PSID_bal : empirical KDE from the 100 RealCause CSV stack
                          (K=100 (Y_0, Y_1) pairs per query; Silverman-rule
                          bandwidth). Requires ITE-match plumbing shared
                          with eval_density_metrics.py.

Usage:
    python realcause_eval/plot_density_examples.py \\
        --dataset IHDP \\
        --root-1d $OUT_1D --root-2d $OUT_2D --causalpfn $CAUSALPFN \\
        --repo $REPO --realization 0 --n-queries 6 \\
        --out plot_densities_IHDP.png
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


DEFAULT_T = 4001
_EPS = 1e-12


def _stub_faiss():
    try:
        import faiss  # noqa: F401
    except ImportError:
        import types as _types
        sys.modules['faiss'] = _types.ModuleType('faiss')


def _import_paths(repo, causalpfn):
    for p in (repo, os.path.join(repo, 'benchmarks', 'l2_ihdp'),
              os.path.join(repo, 'benchmarks', 'l2_acic'),
              causalpfn, os.path.join(causalpfn, 'src')):
        if p and p not in sys.path:
            sys.path.insert(0, p)


def _gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def _renorm(p, dx):
    p = np.clip(p, 0.0, None)
    s = p.sum(axis=-1, keepdims=True) * dx
    return p / np.where(s > 0, s, 1.0)


def _load_1d_ptau_raw(root, method, dataset, r, tau_grid):
    """Reconstruct p_est(τ|x_q) for a 1D method on tau_grid."""
    for pad in (3, 2):
        rstr = f'{r:0{pad}d}'
        path = os.path.join(root, method, dataset, f'{dataset}_r{rstr}.npz')
        if os.path.isfile(path): break
    else:
        return None
    with np.load(path, allow_pickle=True) as z:
        p_y0 = np.asarray(z['p_y0_scaled'], dtype=np.float64)
        p_y1 = np.asarray(z['p_y1_scaled'], dtype=np.float64)
        edges = np.asarray(z['edges'], dtype=np.float64)
        y_scale = float(z['y_scale'])
        K = p_y0.shape[1]
        if K == edges.size + 1:
            p_y0 = p_y0[:, 1:-1]; p_y1 = p_y1[:, 1:-1]; K = p_y0.shape[1]
        centers_scaled = (np.asarray(z['effective_centers'], dtype=np.float64)
                          if 'effective_centers' in z.files
                          else 0.5 * (edges[:-1] + edges[1:]))
    p_y0 = p_y0 / p_y0.sum(axis=-1, keepdims=True).clip(min=_EPS)
    p_y1 = p_y1 / p_y1.sum(axis=-1, keepdims=True).clip(min=_EPS)
    centers_raw = centers_scaled * y_scale
    T = tau_grid.size; dtau = float(tau_grid[1] - tau_grid[0])
    tau_min = float(tau_grid[0])
    tau_pairs = (centers_raw[None, :] - centers_raw[:, None]).ravel()
    idx = np.clip(np.round((tau_pairs - tau_min) / dtau).astype(int), 0, T - 1)
    N_q = p_y0.shape[0]
    p_tau = np.zeros((N_q, T), dtype=np.float64)
    for q in range(N_q):
        mass = np.outer(p_y0[q], p_y1[q]).ravel()
        np.add.at(p_tau[q], idx, mass)
    return _renorm(p_tau / dtau, dtau)


def _load_2d_malc_ptau_raw(root, method, dataset, r, tau_grid, malc_tag='B500'):
    ds_on_disk = dataset
    if method == 'dopfnbb' and dataset == 'PSID_bal':
        ds_on_disk = 'PSIDbal'
    for pad in (3, 2):
        rstr = f'{r:0{pad}d}'
        path = os.path.join(root, method, ds_on_disk,
                             f'malc_ci_{malc_tag}_r{rstr}.npz')
        if os.path.isfile(path): break
    else:
        return None
    with np.load(path, allow_pickle=True) as z:
        p_tau_scaled = np.asarray(z['p_taus_scaled'], dtype=np.float64)
        tau_scaled   = np.asarray(z['tau_scaled'],    dtype=np.float64)
        y_scale      = float(z['y_scale'])
    tau_native_raw = tau_scaled * y_scale
    p_tau_native_raw = p_tau_scaled / max(y_scale, _EPS)
    N_q = p_tau_scaled.shape[0]
    out = np.empty((N_q, tau_grid.size), dtype=np.float64)
    for q in range(N_q):
        out[q] = np.interp(tau_grid, tau_native_raw, p_tau_native_raw[q],
                            left=0.0, right=0.0)
    dtau = float(tau_grid[1] - tau_grid[0])
    return _renorm(out, dtau)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True,
                    choices=['IHDP', 'ACIC', 'CPS', 'PSID', 'PSID_bal'])
    ap.add_argument('--root-1d', required=True)
    ap.add_argument('--root-2d', required=True)
    ap.add_argument('--causalpfn', required=True)
    ap.add_argument('--repo', required=True)
    ap.add_argument('--realization', type=int, default=0)
    ap.add_argument('--n-queries', type=int, default=6,
                    help='Sample queries per method row.')
    ap.add_argument('--query-indices', nargs='+', type=int, default=None,
                    help='Explicit query indices (overrides --n-queries even spacing).')
    ap.add_argument('--methods-1d', nargs='+', default=['cpfn1d', 'dopfn', 'uwyk1d'])
    ap.add_argument('--methods-2d', nargs='+', default=['cpfn2d', 'graph2d', 'dopfnbb'])
    ap.add_argument('--T', type=int, default=DEFAULT_T)
    ap.add_argument('--out', required=True, help='Output PNG path.')
    args = ap.parse_args()

    _stub_faiss()
    _import_paths(args.repo, args.causalpfn)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    r = args.realization
    _ANALYTIC = {'IHDP', 'ACIC'}
    _KDE      = {'CPS', 'PSID', 'PSID_bal'}
    keep_mask = None                        # None → all queries used (IHDP/ACIC)

    if args.dataset == 'IHDP':
        from benchmarks import IHDPDataset
        from true_ihdp import load_ihdp_truth
        ds_obj = IHDPDataset()
        ds = ds_obj[r][0]
        y_train_raw = np.asarray(ds.y_train, dtype=np.float64).reshape(-1)
        truth = load_ihdp_truth(r, args.causalpfn, y_train_raw)
    elif args.dataset == 'ACIC':
        from benchmarks import ACIC2016Dataset
        from true_acic import load_acic_truth
        ds_obj = ACIC2016Dataset()
        ds = ds_obj[r][0]
        y_train_raw = np.asarray(ds.y_train, dtype=np.float64).reshape(-1)
        truth = load_acic_truth(r, y_train_raw)
    else:
        # Lalonde — KDE truth path.
        if args.dataset == 'CPS':
            from benchmarks import RealCauseLalondeCPSDataset
            ds_obj = RealCauseLalondeCPSDataset()
        else:
            from benchmarks import RealCauseLalondePSIDDataset
            ds_obj = RealCauseLalondePSIDDataset()
        ds = ds_obj[r][0]

    if args.dataset in _ANALYTIC:
        scale = truth.y_rng / 2.0
        mu_diff = (truth.mu1_test_scaled - truth.mu0_test_scaled) * scale
        sigma_tau = np.sqrt(2.0) * float(truth.sigma_scaled) * scale
        lo = float(mu_diff.min() - 6 * sigma_tau)
        hi = float(mu_diff.max() + 6 * sigma_tau)
        tau_grid = np.linspace(lo, hi, args.T)
        dtau = tau_grid[1] - tau_grid[0]
        # Truth per query — narrow Gaussians.
        p_true_pq = np.stack([_gaussian(tau_grid, mu, sigma_tau) for mu in mu_diff])
        p_true_pq = _renorm(p_true_pq, dtau)
        true_cate_pq = mu_diff.copy()
    else:
        # KDE truth on Lalonde. Reuse helpers from eval_density_metrics.
        sys.path.insert(0, os.path.join(args.repo, 'realcause_eval'))
        from eval_density_metrics import _load_realcause_stack, _match_by_ite
        _, y0_all, y1_all = _load_realcause_stack(args.dataset, args.causalpfn)
        tau_all = y1_all - y0_all               # (N_pool, K=100)
        test_idx_full = _match_by_ite(args.dataset, r, args.causalpfn, ds)
        keep_mask = test_idx_full >= 0
        if not keep_mask.all():
            print(f'  [warn] {args.dataset} r={r}: '
                  f'{int((~keep_mask).sum())}/{len(test_idx_full)} unmatched',
                  file=sys.stderr)
        test_idx = test_idx_full[keep_mask]
        tau_samples = tau_all[test_idx]         # (N_q_kept, K)
        true_cate_pq = np.asarray(ds.true_cate, dtype=np.float64).reshape(-1)[keep_mask]
        # Grid ±3σ of the stacked sample cloud.
        s_all = tau_samples.std(ddof=1)
        lo = float(tau_samples.min() - 3.0 * s_all)
        hi = float(tau_samples.max() + 3.0 * s_all)
        tau_grid = np.linspace(lo, hi, args.T)
        dtau = tau_grid[1] - tau_grid[0]
        # KDE per matched query — same recipe as the metrics script.
        N_q_m, K = tau_samples.shape
        sigmas = tau_samples.std(axis=1, ddof=1)
        sigma_floor = 1e-3 * (tau_grid[-1] - tau_grid[0])
        h = np.maximum(1.06 * sigmas * (K ** (-1.0 / 5.0)), sigma_floor)
        p_true_pq = np.empty((N_q_m, args.T), dtype=np.float64)
        for q in range(N_q_m):
            z = (tau_grid[:, None] - tau_samples[q, None, :]) / h[q]
            p_true_pq[q] = np.exp(-0.5 * z ** 2).sum(axis=1) / (K * h[q] * np.sqrt(2.0 * np.pi))
        p_true_pq = _renorm(p_true_pq, dtau)
        # For grid-labels only — pick a σ scale we can display in title.
        sigma_tau = float(np.mean(sigmas)) * np.sqrt(2.0)
        mu_diff = true_cate_pq       # for query selection + labeling

    N_q = mu_diff.size
    if args.query_indices is not None:
        qi = args.query_indices
    else:
        qi = np.linspace(0, N_q - 1, args.n_queries).astype(int).tolist()
    print(f'[bootstrap] dataset={args.dataset}  realization={r}  '
          f'queries={qi}  N_q_kept={N_q}', flush=True)

    # Method densities. Apply keep_mask to align with truth's query subset
    # (Lalonde only — IHDP/ACIC use all queries).
    method_densities = {}
    for m in args.methods_1d:
        p = _load_1d_ptau_raw(args.root_1d, m, args.dataset, r, tau_grid)
        if p is not None and keep_mask is not None:
            p = p[keep_mask]
        method_densities[m] = p
        print(f'  loaded 1D {m}: {"OK" if p is not None else "MISSING"}', flush=True)
    for m in args.methods_2d:
        p = _load_2d_malc_ptau_raw(args.root_2d, m, args.dataset, r, tau_grid)
        if p is not None and keep_mask is not None:
            p = p[keep_mask]
        method_densities[m] = p
        print(f'  loaded 2D {m}: {"OK" if p is not None else "MISSING"}', flush=True)

    methods = args.methods_1d + args.methods_2d
    n_rows, n_cols = len(methods), len(qi)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.2 * n_rows),
                              sharex=False, squeeze=False)

    for row, method in enumerate(methods):
        p_est = method_densities[method]
        for col, q in enumerate(qi):
            ax = axes[row, col]
            # Zoom x-axis around truth mean ± 6σ_τ for readability.
            xlo = float(mu_diff[q] - 6 * sigma_tau)
            xhi = float(mu_diff[q] + 6 * sigma_tau)
            mask = (tau_grid >= xlo) & (tau_grid <= xhi)
            ax.plot(tau_grid[mask], p_true_pq[q, mask], 'k-', lw=1.4,
                    label='truth' if col == 0 else None, alpha=0.75)
            if p_est is not None:
                ax.plot(tau_grid[mask], p_est[q, mask], 'C0-', lw=1.4,
                        label='est' if col == 0 else None)
            ax.axvline(mu_diff[q], color='r', ls='--', lw=0.9,
                       label='true τ' if col == 0 else None)
            if col == 0:
                ax.set_ylabel(f'{method}\np(τ)', fontsize=9)
                ax.legend(fontsize=7, loc='upper right')
            if row == 0:
                ax.set_title(f'q={q}\nμ_1−μ_0={mu_diff[q]:+.2f}', fontsize=9)
            if row == n_rows - 1:
                ax.set_xlabel('τ (raw Y)', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_xlim(xlo, xhi)

    fig.suptitle(f'{args.dataset} — realization {r} — truth (black) vs '
                  f'estimate (blue) per query.  σ_τ (truth) = {sigma_tau:.3f}',
                  fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=140, bbox_inches='tight')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
