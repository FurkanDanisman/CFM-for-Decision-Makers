"""Aggregator for per-realization ATE density W2-barycenter results.

Reads ate_w2_{tag}_r<###>.npz files produced by compute_ate_density_w2_cell.py
and emits a markdown table with per-cell:
  ATE_mean  ± SE  (average of per-realization ATE means from p_ATE)
  ATE_bias        (mean of ATE_mean - true_ATE across realizations)
  Coverage  ± SE  (fraction of realizations where true_ATE ∈ [τ_lo, τ_hi])
  Length    ± SE  (mean CI width)

Usage:
    python realcause_eval/ate_density_w2_summary.py \\
        --out-root /scratch/.../results_rc_2d_density \\
        --tag malc_B500 \\
        --methods cpfn2d graph2d dopfnbb
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np


DATASETS = ('IHDP', 'ACIC', 'CPS', 'PSID', 'PSID_bal')

_SPLIT_METHODS = {'dopfnbb'}
_DATASET_DIR_ALIASES = {
    'dopfnbb': {'PSID_bal': 'PSIDbal'},
}


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
        m_s  = f'{m:.3f}'
        se_s = f'{se:.3f}' if np.isfinite(se) else '—'
    return f'{m_s} ± {se_s}'


def _fmt_signed(m, big=False):
    if not np.isfinite(m): return '—'
    return f'{m:+,.2f}' if big else f'{m:+.3f}'


def summarize_cell(method_dir, dataset, method, tag):
    ds_dir_name = _DATASET_DIR_ALIASES.get(method, {}).get(dataset, dataset)
    dataset_dir = os.path.join(method_dir, ds_dir_name)
    paths = sorted(glob.glob(os.path.join(dataset_dir, f'ate_w2_{tag}_r*.npz')))
    if not paths:
        return None
    means, biases, lens, covs, true_ates = [], [], [], [], []
    for p in paths:
        try:
            with np.load(p, allow_pickle=True) as z:
                m  = float(z['ate_mean'])
                ta = float(z['true_ate'])
                lo = float(z['tau_lo'])
                hi = float(z['tau_hi'])
                cv = int(z['covered'])
        except Exception as e:
            print(f'  [warn] {p}: {e}', file=sys.stderr); continue
        means.append(m); true_ates.append(ta); biases.append(m - ta)
        lens.append(hi - lo); covs.append(float(cv))
    if not means:
        return None
    return {
        'ate_mean': _mean_se(means),
        'bias':     _mean_se(biases),
        'abs_err':  _mean_se([abs(b) for b in biases]),
        'cov':      _mean_se(covs),
        'len':      _mean_se(lens),
        'n':        len(means),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--tag', required=True,
                    help='Filename tag written by compute_ate_density_w2_cell.py '
                         "(e.g. 'marginals', 'joint', 'malc_B100', 'malc_B500', 'malc_B1000').")
    ap.add_argument('--methods', nargs='+', required=True,
                    help='Methods to include (one row each).')
    ap.add_argument('--datasets', nargs='+', default=None, choices=list(DATASETS))
    ap.add_argument('--out-md', default=None)
    args = ap.parse_args()

    if not os.path.isdir(args.out_root):
        sys.exit(f'FATAL: --out-root not found: {args.out_root}')

    big = {'CPS', 'PSID', 'PSID_bal'}
    dsets = tuple(args.datasets) if args.datasets else DATASETS
    header = '| Method | ' + ' | '.join(dsets) + ' |'
    sep    = '|' + '|'.join(['---'] * (1 + len(dsets))) + '|'
    lines = [
        f'\nATE density (1D W2 barycenter over queries) — tag={args.tag} — {args.out_root}',
        '',
        '(each cell, top → bottom: ATE_mean ± SE, ATE_bias (mean − true_ATE), '
        'Coverage / Length of 95% CI on p_ATE; n = realizations)',
        '',
        header, sep,
    ]

    import time
    for method in args.methods:
        method_dir = os.path.join(args.out_root, method)
        cells = [method]
        for d in dsets:
            _t0 = time.time()
            print(f'[{time.strftime("%H:%M:%S")}] {method:8s} / {d:9s} ...',
                  end='', flush=True)
            got = summarize_cell(method_dir, d, method, args.tag)
            print(f' done in {time.time() - _t0:5.1f}s'
                  + (f' (n={got["n"]})' if got is not None else ' (no data)'),
                  flush=True)
            if got is None:
                cells.append('—'); continue
            ate_s  = _fmt(*got['ate_mean'], big=d in big)
            bias_s = _fmt_signed(got['bias'][0], big=d in big)
            cov_s  = _fmt(*got['cov'], big=False)
            len_s  = _fmt(*got['len'], big=d in big)
            n      = got['n']
            cells.append(
                f'ATE {ate_s}<br>'
                f'bias {bias_s}<br>'
                f'Cov {cov_s}<br>'
                f'Len {len_s} (n={n})'
            )
        lines.append('| ' + ' | '.join(cells) + ' |')

    md = '\n'.join(lines) + '\n'
    print(md)
    if args.out_md:
        with open(args.out_md, 'w') as f:
            f.write(md)
        print(f'wrote {args.out_md}')


if __name__ == '__main__':
    main()
