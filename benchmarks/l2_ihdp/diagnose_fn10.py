"""Diagnose why Ours(fn=10) is worse than Do-PFN on density-L2 but wins on PEHE.

For each realization, compute per-query CATE mean and variance from the
saved CATE densities and compare against the analytical truth
(mean = mu1 - mu0, variance = 2 sigma^2). Aggregate across realizations.

If Ours(fn=10)'s CATE MEAN is close to truth (i.e. matches Table 3's
sqrt-PEHE) while its VARIANCE is systematically too large, the L2 gap
is a density-shape issue rather than a point-estimate bug. If the mean
is also off, something is wrong with the pipeline.

Usage
-----
    python R-PFN/benchmarks/l2_ihdp/diagnose_fn10.py \\
        --shards-glob "$DEPLOY_ROOT/l2_ihdp/out.r*.npz"
"""
from __future__ import annotations

import argparse
import glob
import sys

import numpy as np


Y_EDGES = np.linspace(-1.5, 1.5, 101)
Y_CENTERS = 0.5 * (Y_EDGES[:-1] + Y_EDGES[1:])
Y_BIN = float(Y_CENTERS[1] - Y_CENTERS[0])

TAU_EDGES = np.linspace(-3.0, 3.0, 601)
TAU_CENTERS = 0.5 * (TAU_EDGES[:-1] + TAU_EDGES[1:])
TAU_BIN = float(TAU_CENTERS[1] - TAU_CENTERS[0])

METHODS = ['ours_fn50', 'ours_fn10', 'dopfn', 'uwyk_noanc']
LABELS = {
    'ours_fn50':  'Ours(fn=50)',
    'ours_fn10':  'Ours(fn=10)',
    'dopfn':      'Do-PFN',
    'uwyk_noanc': 'UWYK-NoAnc',
}


def _mean_var(density_2d):
    """Per-query mean and variance from densities on TAU_CENTERS."""
    m = (TAU_CENTERS[None, :] * density_2d).sum(axis=1) * TAU_BIN
    m2 = (TAU_CENTERS[None, :] ** 2 * density_2d).sum(axis=1) * TAU_BIN
    v = m2 - m ** 2
    return m, v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards-glob', required=True)
    args = ap.parse_args()

    shards = sorted(glob.glob(args.shards_glob))
    if not shards:
        print(f'[fatal] no shards match {args.shards_glob}'); return 2
    print(f'[load] {len(shards)} shards')

    # per method: list of (per-query mean-err arrays)  and  (per-query variance arrays)
    mean_err = {m: [] for m in METHODS}
    var_est  = {m: [] for m in METHODS}
    var_true = []
    # PEHE-like signal: per-query squared error between predicted-mean and true-mean
    pehe_sq  = {m: [] for m in METHODS}

    for path in shards:
        with np.load(path) as f:
            p_tau_true = f['p_tau_true']
            true_mean, true_var = _mean_var(p_tau_true)
            var_true.append(true_var)
            for m in METHODS:
                key = f'{m}__p_tau'
                if key not in f: continue
                mu, va = _mean_var(f[key])
                mean_err[m].append(mu - true_mean)
                var_est[m].append(va)
                pehe_sq[m].append((mu - true_mean) ** 2)

    var_true = np.concatenate(var_true)
    print()
    print(f'{"method":12s} {"n":>6s}   '
          f'{"mean_bias":>10s} {"|mean_err|":>10s}   '
          f'{"var_pred":>9s} {"var_true":>9s} '
          f'{"var_pred/var_true":>18s}   '
          f'{"pseudo_pehe":>12s}')
    print('-' * 100)
    for m in METHODS:
        if not mean_err[m]:
            continue
        me = np.concatenate(mean_err[m])
        ve = np.concatenate(var_est[m])
        pe = np.concatenate(pehe_sq[m])
        print(f'{LABELS[m]:12s} {me.size:>6d}   '
              f'{me.mean():>+10.4f} {np.abs(me).mean():>10.4f}   '
              f'{ve.mean():>9.4f} {var_true.mean():>9.4f} '
              f'{(ve.mean() / var_true.mean()):>18.3f}   '
              f'{np.sqrt(pe.mean()):>12.4f}')

    print()
    print('Reading:')
    print('  mean_bias     = mean(pred_mean - true_mean)   -> 0 if unbiased')
    print('  |mean_err|    = mean(|pred_mean - true_mean|) -> smaller = better POINT estimate')
    print('  var_pred      = mean of predicted CATE density variance')
    print('  var_true      = mean of true CATE density variance (should be ~2 sigma^2)')
    print('  var_pred/var_true = 1.0 if predicted density is correctly scaled')
    print('  pseudo_pehe   = sqrt(mean((pred_mean - true_mean)^2))   in scaled Y units')
    print()
    print('If a method has small |mean_err| (~= Table 3 winner) but var_pred/var_true')
    print('>> 1, its density is too wide -> good PEHE but bad L2.  That is a shape')
    print('issue, not a bug.  If mean_bias is large, the point estimate is off ->')
    print('bug or model mis-specification.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
