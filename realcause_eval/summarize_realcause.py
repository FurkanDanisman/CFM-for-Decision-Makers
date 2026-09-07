"""Aggregate NPZ per-realization outputs from cpfn2d / cpfn1d / graph2d /
Do-PFN-bb RealCause eval into one 5-column table.

Reads --out-root/<DATASET>/<DATASET>_r<###>.npz for
DATASET ∈ {IHDP, ACIC, CPS, PSID, PSID_bal} and prints √PEHE (mean ± SE)
and ε_ATE (mean ± SE) across all 5 datasets. Also picks the metric variant
(pehe_raw / pehe_em / pehe_full) — default raw, matching the paper convention.

Usage:
    python realcause_eval/summarize_realcause.py \\
        --out-root /scratch/.../results_cpfn2d_realcause_step50k

    # ε_ATE from the EM-post-processed CATE instead of raw:
    python realcause_eval/summarize_realcause.py \\
        --out-root /scratch/.../results_cpfn2d_realcause_step50k \\
        --variant em
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np


DATASETS = ('IHDP', 'ACIC', 'CPS', 'PSID', 'PSID_bal')
VARIANTS = ('raw', 'em', 'full')


def _load(out_root: str, dataset: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(out_root, dataset, f'{dataset}_r*.npz'))):
        try:
            with np.load(path, allow_pickle=True) as z:
                rows.append({k: z[k].item() if z[k].shape == () else z[k]
                             for k in z.files})
        except Exception as e:
            print(f'  [warn] {path}: {e}', file=sys.stderr)
    return rows


def _mean_se(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float('nan'), float('nan'), 0
    m = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else float('nan')
    return m, se, int(v.size)


def _fmt(m, se, n, big: bool) -> str:
    if n == 0 or not np.isfinite(m):
        return '—'
    if big:
        m_s = f'{m:,.2f}'
        se_s = f'{se:,.2f}' if np.isfinite(se) else '—'
    else:
        m_s = f'{m:.4f}'
        se_s = f'{se:.4f}' if np.isfinite(se) else '—'
    return f'{m_s} ± {se_s}  (n={n})'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-root', required=True,
                    help='Parent dir that contains one subdir per dataset with '
                         '<DATASET>_r<###>.npz files inside.')
    ap.add_argument('--variant', default='raw', choices=list(VARIANTS),
                    help='pehe_<variant> / err_<variant> to aggregate. Default: raw.')
    ap.add_argument('--tag', default='',
                    help='Extra suffix for graph2d/uwyk1d NPZs which store keys '
                         'as pehe_<variant>_<tag> (e.g. --tag v3b or --tag noanc). '
                         'Leave empty for other methods where keys are un-suffixed.')
    args = ap.parse_args()

    if not os.path.isdir(args.out_root):
        sys.exit(f'FATAL: --out-root not found: {args.out_root}')

    suffix = f'_{args.tag}' if args.tag else ''
    pehe_key = f'pehe_{args.variant}{suffix}'
    err_key  = f'err_{args.variant}{suffix}'

    per_ds = {d: _load(args.out_root, d) for d in DATASETS}

    lines = [f'\nRealCause — {args.out_root}  '
             f'(variant={args.variant}{"  tag=" + args.tag if args.tag else ""})\n']
    for metric_key, label, big_cols in (
        (pehe_key, '√PEHE',   {'CPS', 'PSID', 'PSID_bal'}),
        (err_key,  'ε_ATE',   set()),
    ):
        lines.append(f'## {label} (mean ± SE)')
        lines.append('| ' + ' | '.join(DATASETS) + ' |')
        lines.append('|' + '|'.join(['---'] * len(DATASETS)) + '|')
        cells = []
        for d in DATASETS:
            vals = [row.get(metric_key) for row in per_ds[d]]
            cells.append(_fmt(*_mean_se(vals), big=d in big_cols))
        lines.append('| ' + ' | '.join(cells) + ' |')
        lines.append('')

    print('\n'.join(lines))


if __name__ == '__main__':
    main()
