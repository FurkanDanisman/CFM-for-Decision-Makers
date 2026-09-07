"""One-table aggregator across all 7 methods × 5 RealCause datasets.

Reads the outputs of submit_realcause_all_methods.sbatch — one directory per
(method, dataset) under --out-root, in different formats per method:

    cpfn1d   : $OUT_ROOT/cpfn1d/<D>/<D>_r<###>.npz      keys: pehe_raw, err_raw
    cpfn2d   : $OUT_ROOT/cpfn2d/<D>/<D>_r<###>.npz      keys: pehe_raw, err_raw
    dopfn    : $OUT_ROOT/dopfn/<D>/<D>_r<###>.npz       keys: pehe_dopfn, err_dopfn
    dopfnbb  : $OUT_ROOT/dopfnbb/<D>/summary.npz        keys: pehe[], eps_ate[]
    fn_50    : $OUT_ROOT/fn_50/<D>/rc_all_<D>/<pkl>     keys: pehe, ate_rel_err
    graph2d  : $OUT_ROOT/graph2d/<D>/<D>_r<###>.npz     keys: pehe_raw_v3b/_noanc, err_raw_v3b/_noanc
    uwyk     : $OUT_ROOT/uwyk/<D>/<model>_<D>_<r>       (pkl, from UWYK's own dofm scripts — paper-exact)
    uwyk1d   : $OUT_ROOT/uwyk1d/<D>/<D>_r<###>.npz      keys: pehe_raw_v3b/_noanc, err_raw_v3b/_noanc

graph2d produces TWO rows (noanc + v3b/"full") from one NPZ. uwyk (own scripts)
gives the noanc row; uwyk1d (harness) gives the v3b row. Final table has 9 rows
× 5 columns, each cell shows

    √PEHE mean ± SE
    ε_ATE mean ± SE

Usage:
    python realcause_eval/aggregate_all_methods.py --out-root /scratch/.../results_realcause_all
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys

import numpy as np


DATASETS = ('IHDP', 'ACIC', 'CPS', 'PSID', 'PSID_bal')

# Row order for the final table. Each entry: (row_label, method_dir, loader_key).
ROW_SPEC = (
    ('cpfn1d',       'cpfn1d',  'npz_pehe_raw'),
    ('cpfn2d',       'cpfn2d',  'npz_pehe_raw'),
    ('dopfn',        'dopfn',   'npz_dopfn'),
    ('dopfnbb',      'dopfnbb', 'npz_summary'),
    ('fn_50',        'fn_50',   'pkl_fn50'),
    ('graph2d noanc','graph2d', 'npz_pehe_raw_noanc'),
    ('graph2d v3b',  'graph2d', 'npz_pehe_raw_v3b'),
    ('uwyk noanc',   'uwyk',    'pkl_uwyk'),
    ('uwyk v3b',     'uwyk1d',  'npz_pehe_raw_v3b'),
)

# For each loader_key: (pehe_key, err_key)  — used inside per-realization NPZs.
NPZ_KEYS = {
    'npz_pehe_raw':        ('pehe_raw',       'err_raw'),
    'npz_dopfn':           ('pehe_dopfn',     'err_dopfn'),
    'npz_pehe_raw_v3b':    ('pehe_raw_v3b',   'err_raw_v3b'),
    'npz_pehe_raw_noanc':  ('pehe_raw_noanc', 'err_raw_noanc'),
}


def _load_per_realization_npz(dir_path: str, dataset: str, pehe_key: str, err_key: str):
    """Load all <DATASET>_r<###>.npz in dir_path; return (pehe[], err[])."""
    pehes, errs = [], []
    for p in sorted(glob.glob(os.path.join(dir_path, f'{dataset}_r*.npz'))):
        try:
            with np.load(p, allow_pickle=True) as z:
                if pehe_key in z.files:
                    v = z[pehe_key]
                    pehes.append(float(v.item() if v.shape == () else np.nanmean(v)))
                if err_key in z.files:
                    v = z[err_key]
                    errs.append(float(v.item() if v.shape == () else np.nanmean(v)))
        except Exception as e:
            print(f'  [warn] {p}: {e}', file=sys.stderr)
    return pehes, errs


def _load_summary_npz(dir_path: str):
    """Load $dir/summary.npz which has arrays pehe[], eps_ate[]."""
    p = os.path.join(dir_path, 'summary.npz')
    if not os.path.isfile(p):
        return [], []
    try:
        with np.load(p, allow_pickle=True) as z:
            pehes = list(np.asarray(z['pehe'], dtype=float)) if 'pehe' in z.files else []
            errs  = list(np.asarray(z['eps_ate'], dtype=float)) if 'eps_ate' in z.files else []
        return pehes, errs
    except Exception as e:
        print(f'  [warn] {p}: {e}', file=sys.stderr)
        return [], []


def _load_fn50_pkls(dir_path: str):
    """Walk fn_50 output — pkl files named ours_fn50_<DATASET>_<r> anywhere under it."""
    pehes, errs = [], []
    for p in sorted(glob.glob(os.path.join(dir_path, '**', 'ours_fn50_*'),
                              recursive=True)):
        if not os.path.isfile(p) or p.endswith(('.csv', '.json', '.md')):
            continue
        try:
            with open(p, 'rb') as f:
                d = pickle.load(f)
            if not isinstance(d, dict):
                continue
            if 'pehe' in d and d['pehe'] is not None and np.isfinite(d['pehe']):
                pehes.append(float(d['pehe']))
            if 'ate_rel_err' in d and d['ate_rel_err'] is not None \
                    and np.isfinite(d['ate_rel_err']):
                errs.append(float(d['ate_rel_err']))
        except Exception as e:
            print(f'  [warn] {p}: {e}', file=sys.stderr)
    return pehes, errs


def _load_uwyk_pkls(dir_path: str):
    """Walk uwyk output — pkls named dofm_noclust_<D>_<r> or dofm_psid_balanced_<D>_<r>.
    Keys inside: {'pehe', 'ate_rel_err', 'model', 'dataset', ...}."""
    pehes, errs = [], []
    for p in sorted(glob.glob(os.path.join(dir_path, '**', 'dofm_*'),
                              recursive=True)):
        if not os.path.isfile(p) or p.endswith(('.csv', '.json', '.md')):
            continue
        try:
            with open(p, 'rb') as f:
                d = pickle.load(f)
            if not isinstance(d, dict):
                continue
            if 'pehe' in d and d['pehe'] is not None and np.isfinite(d['pehe']):
                pehes.append(float(d['pehe']))
            if 'ate_rel_err' in d and d['ate_rel_err'] is not None \
                    and np.isfinite(d['ate_rel_err']):
                errs.append(float(d['ate_rel_err']))
        except Exception as e:
            print(f'  [warn] {p}: {e}', file=sys.stderr)
    return pehes, errs


def _mean_se(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float('nan'), float('nan'), 0
    m = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else float('nan')
    return m, se, int(v.size)


def _fmt_pehe(m, se, n, big):
    if n == 0 or not np.isfinite(m):
        return '—'
    return (f'{m:,.2f} ± {se:,.2f}' if big else f'{m:.3f} ± {se:.3f}')


def _fmt_err(m, se, n):
    if n == 0 or not np.isfinite(m):
        return '—'
    return f'{m:.3f} ± {se:.3f}'


def _load_cell(out_root: str, method_dir: str, dataset: str, loader_key: str):
    """→ (pehe_list, err_list). Handles per-realization NPZ, summary NPZ, and
    fn_50 pkls."""
    dir_path = os.path.join(out_root, method_dir, dataset)
    if not os.path.isdir(dir_path):
        return [], []
    if loader_key == 'npz_summary':
        return _load_summary_npz(dir_path)
    if loader_key == 'pkl_fn50':
        return _load_fn50_pkls(dir_path)
    if loader_key == 'pkl_uwyk':
        return _load_uwyk_pkls(dir_path)
    pehe_key, err_key = NPZ_KEYS[loader_key]
    return _load_per_realization_npz(dir_path, dataset, pehe_key, err_key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-root', required=True,
                    help='Parent dir where the 7 method subdirs live.')
    ap.add_argument('--out-md', default=None,
                    help='Optional path to also write the markdown table to.')
    args = ap.parse_args()

    if not os.path.isdir(args.out_root):
        sys.exit(f'FATAL: --out-root not found: {args.out_root}')

    big_pehe = {'CPS', 'PSID', 'PSID_bal'}

    header = '| Method | ' + ' | '.join(DATASETS) + ' |'
    sep    = '|' + '|'.join(['---'] * (1 + len(DATASETS))) + '|'
    lines = [f'\nRealCause — {args.out_root}', '',
             '(each cell: √PEHE mean±SE on top, ε_ATE mean±SE below; n=realizations)',
             '', header, sep]
    for row_label, method_dir, loader_key in ROW_SPEC:
        cells = [row_label]
        for d in DATASETS:
            pehes, errs = _load_cell(args.out_root, method_dir, d, loader_key)
            pm, pse, pn = _mean_se(pehes)
            em, ese, en = _mean_se(errs)
            big = d in big_pehe
            pehe_str = _fmt_pehe(pm, pse, pn, big)
            err_str  = _fmt_err(em, ese, en)
            n_tag = f'(n={max(pn, en)})' if max(pn, en) > 0 else ''
            cells.append(f'{pehe_str}<br>{err_str} {n_tag}'.strip())
        lines.append('| ' + ' | '.join(cells) + ' |')

    md = '\n'.join(lines) + '\n'
    print(md)

    if args.out_md:
        with open(args.out_md, 'w') as f:
            f.write(md)
        print(f'wrote {args.out_md}')


if __name__ == '__main__':
    main()
