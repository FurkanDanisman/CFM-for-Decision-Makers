"""Print a single Table-3-style row (PEHE ± SEM, eps_ATE ± SEM) for one
config across all datasets.

For the paper: `--ckpt "J=10 s150k" --scheme std` gives you the numbers
we settled on (min_max also usable).

Reads the per-realization arrays saved by eval_dopfn_bb_raw.py so the
SEMs are exact — not just the summary means.

Usage:
    python table3_row.py --ckpt "J=10 s150k" --scheme std
    python table3_row.py --ckpt "J=10 s150k" --scheme min_max
"""
from __future__ import annotations
import argparse, glob, os, re, sys
import numpy as np


DATASETS = ['IHDP', 'ACIC', 'CPS', 'PSID', 'PSIDbal', 'law_race', 'sales']


def _shard_path(deploy: str, ckpt: str, ds: str, scheme_tag: str):
    """Map (ckpt, ds, scheme) → shard .npz path.

    Sweeps saved shards under a few different naming conventions across
    submissions:
        rel_${ds}_${tag}.npz              (early J=10 s150k)
        rel_${ds}_${tag}_s${step}.npz     (later J=10 s150k)
        j10_s${step}_${ds}_${tag}.npz     (J=10 at other step)
        fn50_${ds}_${tag}.npz             (fn=50)
    Newest matching file wins (mtime).
    """
    import glob as _glob
    scheme_alias = {'trim5': 'trim5', 'trim10': 'trim10'}.get(scheme_tag, scheme_tag)
    dirpath = os.path.join(deploy, 'eval_dopfn_bb_raw')
    patterns = []
    if ckpt == 'J=10 s150k':
        patterns += [
            f'rel_{ds}_{scheme_alias}.npz',
            f'rel_{ds}_{scheme_alias}_s150000.npz',
        ]
    m = re.match(r'^J=10 s(\d+)k$', ckpt)
    if m:
        step = int(m.group(1)) * 1000
        patterns += [
            f'j10_s{step}_{ds}_{scheme_alias}.npz',
            f'j10_s{step}_{ds}_{scheme_alias}_s{step}.npz',
        ]
    if ckpt == 'fn=50':
        patterns += [f'fn50_{ds}_{scheme_alias}.npz']

    hits = []
    for pat in patterns:
        hits.extend(_glob.glob(os.path.join(dirpath, pat)))
    if not hits:
        return None
    hits.sort(key=os.path.getmtime, reverse=True)
    return hits[0]


def _stats(arr: np.ndarray) -> tuple:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float('nan'), float('nan'), 0
    m = float(arr.mean())
    sem = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return m, sem, arr.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--deploy', default=os.environ.get('DEPLOY_ROOT'))
    ap.add_argument('--ckpt',   default='J=10 s150k',
                    help='Checkpoint label, e.g. "J=10 s150k", "J=10 s200k", "fn=50".')
    ap.add_argument('--scheme', default='std',
                    choices=['min_max', 'std', 'trim5', 'trim10', 'log_transform'])
    ap.add_argument('--full', action='store_true',
                    help='Report PEHE_em and eps_ATE_em (full 9-region) instead of the '
                         'inner-only columns. Default OFF — matches Do-PFN paper convention '
                         'which reports inner-only PEHE.')
    args = ap.parse_args()
    if not args.deploy:
        sys.exit('DEPLOY_ROOT not set; pass --deploy or export it')

    label = 'full 9-region' if args.full else 'inner-only'
    print(f'\n══ Table 3 row: {args.ckpt}  |  scheme={args.scheme}  |  '
          f'metric={label}  ══')
    print(f'{"dataset":10s}  {"n":>5s}   {"PEHE ± SEM":>25s}   {"eps_ATE ± SEM":>18s}')
    print('-' * 75)

    pehe_key = 'pehe_em' if args.full else 'pehe'
    ate_key  = 'eps_ate_em' if args.full else 'eps_ate'

    for ds in DATASETS:
        shard = _shard_path(args.deploy, args.ckpt, ds, args.scheme)
        if shard is None:
            print(f'{ds:10s}  {"—":>5s}   {"(no shard)":>25s}   {"—":>18s}')
            continue
        with np.load(shard, allow_pickle=True) as f:
            if pehe_key not in f.files or ate_key not in f.files:
                print(f'{ds:10s}  {"—":>5s}   {"(missing arrays)":>25s}   {"—":>18s}')
                continue
            pehe_arr = np.asarray(f[pehe_key])
            ate_arr  = np.asarray(f[ate_key])
        pm, ps, n = _stats(pehe_arr)
        am, as_, _ = _stats(ate_arr)
        print(f'{ds:10s}  {n:>5d}   {pm:>13.4f} ± {ps:<9.4f}   {am:>7.4f} ± {as_:.4f}')

    print()


if __name__ == '__main__':
    main()
