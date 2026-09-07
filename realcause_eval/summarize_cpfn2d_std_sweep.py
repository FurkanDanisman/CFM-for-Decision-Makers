"""Sweep summary: cpfn2d std_target ∈ {0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1}.

Reads $OUT_ROOT/cpfn2d_std_<sigma>/<DATASET>/<DATASET>_r<###>.npz and prints
one row per sigma. Cells show √PEHE (mean ± SE) on top, ε_ATE below.

Usage:
    python realcause_eval/summarize_cpfn2d_std_sweep.py \\
        --out-root /scratch/.../results_cpfn2d_std_sweep
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np


SIGMAS = (0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1)
DATASETS = ('IHDP', 'ACIC', 'CPS', 'PSID', 'PSID_bal')


def _load(dir_path, dataset, key):
    vals = []
    for p in sorted(glob.glob(os.path.join(dir_path, f'{dataset}_r*.npz'))):
        try:
            with np.load(p, allow_pickle=True) as z:
                if key in z.files:
                    v = z[key]
                    vals.append(float(v.item() if v.shape == () else np.nanmean(v)))
        except Exception as e:
            print(f'  [warn] {p}: {e}', file=sys.stderr)
    return vals


def _mean_se(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float('nan'), float('nan'), 0
    m = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else float('nan')
    return m, se, int(v.size)


def _fmt(m, se, n, big):
    if n == 0 or not np.isfinite(m):
        return '—'
    if big:
        m_s = f'{m:,.2f}'
        se_s = f'{se:,.2f}' if np.isfinite(se) else '—'
    else:
        m_s = f'{m:.3f}'
        se_s = f'{se:.3f}' if np.isfinite(se) else '—'
    return f'{m_s} ± {se_s}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-root', required=True,
                    help='Parent dir; expects subdirs cpfn2d_std_<σ>/<DATASET>/<D>_r<###>.npz.')
    args = ap.parse_args()

    if not os.path.isdir(args.out_root):
        sys.exit(f'FATAL: --out-root not found: {args.out_root}')

    big_cols = {'CPS', 'PSID', 'PSID_bal'}

    header = '| σ | ' + ' | '.join(DATASETS) + ' |'
    sep    = '|' + '|'.join(['---'] * (1 + len(DATASETS))) + '|'

    lines = [f'\ncpfn2d std_target sweep — {args.out_root}', '',
             '(each cell: √PEHE mean±SE on top, ε_ATE mean±SE below)', '',
             header, sep]

    for sigma in SIGMAS:
        cells = [f'{sigma}']
        for d in DATASETS:
            dir_path = os.path.join(args.out_root, f'cpfn2d_std_{sigma}', d)
            pehes = _load(dir_path, d, 'pehe_raw')
            errs  = _load(dir_path, d, 'err_raw')
            pm, pse, pn = _mean_se(pehes)
            em, ese, en = _mean_se(errs)
            big = d in big_cols
            pehe_str = _fmt(pm, pse, pn, big)
            err_str  = _fmt(em, ese, en, False)
            n_tag = f' (n={max(pn, en)})' if max(pn, en) > 0 else ''
            cells.append(f'{pehe_str}<br>{err_str}{n_tag}')
        lines.append('| ' + ' | '.join(cells) + ' |')

    print('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
