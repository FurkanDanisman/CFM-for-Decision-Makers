"""L2 per-bin-probability at a configurable target J.

Reads existing per-realization shards (produced by eval_realization.py) and
re-aggregates each method's native-grid p_y0 / p_y1 / p_tau (stored on
true_ihdp.Y_CENTERS / TAU_CENTERS — a 100-bin fine grid over [-1.5, 1.5]
for Y and 600 bins over [-3, 3] for τ) to a coarser J-bin target grid
over [-1, 1]. Truth is derived directly from the analytic Gaussian per
query at the same target J.

Usage:
  python l2_per_bin_target_j.py \\
      --shards-glob '<dir>/*.npz' \\
      --dataset {ihdp,acic} \\
      --j-target 32 \\
      --methods ours_fn50,ours_dopfn_bb,ours_graph2d,uwyk_noanc,uwyk_anc \\
      --repo <repo>
"""
from __future__ import annotations
import argparse
import glob
import os
import sys
import numpy as np


def _method_to_label(m: str) -> str:
    return {
        'ours_fn50':      'fn=50',
        'ours_fn10':      'fn=10',
        'ours_dopfn_bb':  'Do-PFN-bb',
        'ours_graph2d':   'graph2d',
        'uwyk_noanc':     'UWYK No-Anc',
        'uwyk_anc':       'UWYK Anc',
        'uwyk_predictive':'UWYK Predictive',
        'dopfn':          'Do-PFN',
    }.get(m, m)


def _rebin_1d(p_src, src_centers, dst_edges):
    """Sum src probabilities into dst bins (both on same [-1, 1]-ish scale).

    p_src: (n_test, len(src_centers)) or (len(src_centers),)  probabilities
    src_centers: (N_src,) — sample points where each p_src[i] value lives
    dst_edges: (J+1,) — target bin edges
    Returns: (n_test, J)  probability per dst bin (summed src mass).
    """
    if p_src.ndim == 1:
        p_src = p_src[None, :]
    n, N_src = p_src.shape
    J = len(dst_edges) - 1
    idx = np.clip(np.searchsorted(dst_edges, src_centers, side='right') - 1, 0, J - 1)
    out = np.zeros((n, J), dtype=np.float64)
    for j in range(J):
        cols = np.where(idx == j)[0]
        if cols.size:
            out[:, j] = p_src[:, cols].sum(axis=1)
    # Renormalise per row (some src mass may fall outside dst range)
    row_sum = out.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum > 0, row_sum, 1.0)
    return out / row_sum


def _truth_probs_gaussian(edges, mu, sigma):
    """Per-query truth: bin probabilities from Gaussian(mu, sigma)."""
    from scipy.stats import norm
    mu = np.atleast_1d(mu); sigma = np.atleast_1d(sigma)
    if sigma.size == 1:
        sigma = np.full_like(mu, float(sigma[0]))
    n = mu.shape[0]; J = len(edges) - 1
    cdf = norm.cdf(edges[None, :], loc=mu[:, None], scale=np.clip(sigma[:, None], 1e-8, None))
    return cdf[:, 1:] - cdf[:, :-1]           # (n, J)


def _l2(p, t, bw):
    """L2 = sqrt( Σ_bin (p_method - p_truth)² / bin_w ) per query, then mean+std."""
    err = np.sqrt(((p - t) ** 2).sum(axis=1) / bw)
    return float(err.mean()), float(err.std(ddof=1) / np.sqrt(err.size)), int(err.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards-glob', required=True)
    ap.add_argument('--dataset',     required=True, choices=['ihdp', 'acic'])
    ap.add_argument('--j-target',    type=int, default=32)
    ap.add_argument('--methods',     default='ours_fn50,ours_dopfn_bb,ours_graph2d,uwyk_noanc,uwyk_anc',
                    help='comma-separated ordered list of method keys to report as rows.')
    ap.add_argument('--repo',        required=True)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(args.repo, 'benchmarks', 'l2_ihdp'))
    if args.dataset == 'ihdp':
        from true_ihdp import load_ihdp_truth, Y_CENTERS, TAU_CENTERS
    else:
        from true_acic import load_acic_truth as load_ihdp_truth
        from true_ihdp import Y_CENTERS, TAU_CENTERS   # shared grids

    # Target grid at J-bins over [-1, 1]
    J = args.j_target
    edges_Y   = np.linspace(-1.0, 1.0, J + 1)
    bin_w_Y   = float(edges_Y[1] - edges_Y[0])
    n_tau     = 2 * J                                   # match paper convention
    edges_tau = np.linspace(-2.0, 2.0, n_tau + 1)
    bin_w_tau = float(edges_tau[1] - edges_tau[0])

    files = sorted(glob.glob(args.shards_glob))
    if not files:
        print(f'no shards found: {args.shards_glob}'); return

    # per-method accumulators
    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    per_m = {m: {'y0': [], 'y1': [], 'tau': []} for m in methods}

    for fn in files:
        f = np.load(fn, allow_pickle=True)
        r = int(f['realization']) if 'realization' in f.files else -1
        try:
            truth = load_ihdp_truth(r) if args.dataset == 'ihdp' else load_ihdp_truth(r)
        except Exception as e:
            print(f'[warn] r={r}: cannot load truth: {e}'); continue

        # Truth marginals on the J-target grid
        mu0 = np.asarray(truth.mu0_test_scaled)
        mu1 = np.asarray(truth.mu1_test_scaled)
        sig = float(truth.sigma_scaled)
        p_y0_true = _truth_probs_gaussian(edges_Y, mu0, sig)
        p_y1_true = _truth_probs_gaussian(edges_Y, mu1, sig)
        tau_mu    = mu1 - mu0
        tau_sig   = float(np.sqrt(2.0) * sig)
        p_tau_true = _truth_probs_gaussian(edges_tau, tau_mu, tau_sig)

        for m in methods:
            k_y0  = f'{m}__p_y0'
            k_y1  = f'{m}__p_y1'
            k_tau = f'{m}__p_tau'
            if k_y0 not in f.files or k_y1 not in f.files:
                continue
            p_y0_agg = _rebin_1d(np.asarray(f[k_y0]), Y_CENTERS,   edges_Y)
            p_y1_agg = _rebin_1d(np.asarray(f[k_y1]), Y_CENTERS,   edges_Y)
            per_m[m]['y0'].append(_l2(p_y0_agg, p_y0_true, bin_w_Y))
            per_m[m]['y1'].append(_l2(p_y1_agg, p_y1_true, bin_w_Y))
            if k_tau in f.files:
                p_tau_agg = _rebin_1d(np.asarray(f[k_tau]), TAU_CENTERS, edges_tau)
                per_m[m]['tau'].append(_l2(p_tau_agg, p_tau_true, bin_w_tau))

    print()
    print(f'══ {args.dataset.upper()} — per-bin L2 at J_target={J} (Y bins over [-1,1], '
          f'τ bins over [-2,2] with {n_tau} bins) ══')
    print(f'{"method":<20} {"L2(y0)":>18} {"L2(y1)":>18} {"L2(τ)":>18}')
    print('-' * 78)
    def _agg_across_realizations(items):
        if not items: return '     — '
        # items is list of (mean, sem, n) per realization
        means = np.array([x[0] for x in items])
        return f'{means.mean():8.4f} ± {means.std(ddof=1)/np.sqrt(len(means)):6.4f} (n={len(means):>3d})'
    for m in methods:
        y0 = _agg_across_realizations(per_m[m]['y0'])
        y1 = _agg_across_realizations(per_m[m]['y1'])
        ta = _agg_across_realizations(per_m[m]['tau'])
        print(f'{_method_to_label(m):<20} {y0:>18} {y1:>18} {ta:>18}')


if __name__ == '__main__':
    main()
