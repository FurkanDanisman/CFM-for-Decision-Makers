"""Aggregate graph_eval per-realization *.npz outputs into a summary table.

Expects a directory layout like:

    <root>/
        IHDP/IHDP_r000.npz, IHDP_r001.npz, ...
        ACIC/ACIC_r00.npz,  ACIC_r01.npz,  ...
        CPS/CPS_r000.npz,   CPS_r001.npz,  ...
        PSID/PSID_r000.npz, ...
        PSID_bal/PSID_bal_r000.npz, ...

Prints one row per (dataset, estimator) with mean PEHE and mean err_ATE
across realizations. Robust to any subset of datasets being missing.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np


DATASETS_DEFAULT = ('IHDP', 'ACIC', 'CPS', 'PSID', 'PSID_bal')
ESTIMATORS_DEFAULT = ('raw_anc', 'em_anc', 'raw_noanc', 'em_noanc')


def _mean_se(values):
    v = np.asarray([x for x in values if np.isfinite(x)])
    if v.size == 0:
        return float('nan'), float('nan'), 0
    if v.size == 1:
        return float(v[0]), float('nan'), 1
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)), int(v.size)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True,
                        help='Root dir holding per-dataset subdirs with npz files')
    parser.add_argument('--datasets', nargs='+', default=list(DATASETS_DEFAULT))
    parser.add_argument('--variants', nargs='+', default=None,
                        help='Override estimator keys (default: raw_anc/em_anc/raw_noanc/em_noanc)')
    args = parser.parse_args()

    estimators = tuple(args.variants) if args.variants else ESTIMATORS_DEFAULT

    print(f'{"dataset":<10} {"estimator":<12} {"pehe":>10} {"pehe_se":>8}  {"err":>7} {"err_se":>7}  {"n":>4}')
    print('─' * 68)
    for ds in args.datasets:
        files = sorted(glob.glob(os.path.join(args.root, ds, f'{ds}_r*.npz')))
        if not files:
            continue
        loaded = [np.load(f) for f in files]
        for est in estimators:
            key_pehe = f'pehe_{est}'
            key_err  = f'err_{est}'
            pehe_vals = [float(d[key_pehe]) for d in loaded if key_pehe in d.files]
            err_vals  = [float(d[key_err])  for d in loaded if key_err  in d.files]
            if not pehe_vals:
                continue
            mp, sp, n = _mean_se(pehe_vals)
            me, se, _ = _mean_se(err_vals)
            print(f'{ds:<10} {est:<12} {mp:>10.3f} {sp:>8.3f}  {me:>7.3f} {se:>7.3f}  {n:>4d}')
        print()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
