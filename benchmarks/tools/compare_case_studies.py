"""Side-by-side case-study (or RealCause) comparison across models.

Reads the per-realization npz that eval_cpfn{1,2}d_realcause.py writes at
    <root>/<type>/<DATASET>/<DATASET>_r###.npz
and prints mean +/- SE of a chosen metric for every (dataset, model) pair,
marking the best model per row.

    python benchmarks/tools/compare_case_studies.py \
        --root cpfn2d_armc_30k=$DEPLOY_ROOT/cs_armc_step30000 \
        --root cpfn1d_j1024_50k=$DEPLOY_ROOT/cs_cpfn1d_step50k \
        --type case_studies --metric pehe_raw

Pass --metric err_raw for the ATE column. Datasets missing from a root show
as '--' rather than silently dropping the row, so a half-finished array is
visible instead of looking like a clean sweep.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

CASES = ['Observed_Mediator', 'Observed_Mediator_and_Confounder',
         'Observed_Confounder', 'Backdoor_Criterion',
         'Frontdoor_Criterion', 'Unobserved_Confounder']
REALCAUSE = ['IHDP', 'ACIC', 'CPS', 'PSID', 'PSID_bal']


def load(root, kind, dataset, metric):
    d = os.path.join(root, kind, dataset)
    vals = []
    for p in sorted(glob.glob(os.path.join(d, f'{dataset}_r*.npz'))):
        try:
            with np.load(p, allow_pickle=True) as z:
                if metric in z.files:
                    vals.append(float(np.asarray(z[metric]).reshape(-1)[0]))
        except Exception:
            continue
    a = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if a.size == 0:
        return None
    se = float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
    return float(a.mean()), se, int(a.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', action='append', required=True,
                    help='NAME=PATH, repeatable')
    ap.add_argument('--type', default='case_studies',
                    choices=('case_studies', 'realcause'))
    ap.add_argument('--metric', default='pehe_raw')
    a = ap.parse_args()

    roots = []
    for spec in a.root:
        if '=' not in spec:
            raise SystemExit(f'--root needs NAME=PATH, got {spec!r}')
        name, path = spec.split('=', 1)
        roots.append((name, os.path.expanduser(path)))

    datasets = CASES if a.type == 'case_studies' else REALCAUSE
    w = max(34, max(len(d) for d in datasets) + 2)
    cw = max(22, max(len(n) for n, _ in roots) + 2)

    print(f'\n{a.metric}  ({a.type})  mean +/- SE  [n]   * = best in row\n')
    print(f'{"dataset":<{w}}' + ''.join(f'{n:>{cw}}' for n, _ in roots))
    print('-' * (w + cw * len(roots)))

    wins = {n: 0 for n, _ in roots}
    for ds in datasets:
        got = [load(path, a.type, ds, a.metric) for _, path in roots]
        best = None
        finite = [(i, g[0]) for i, g in enumerate(got) if g is not None]
        if finite:
            best = min(finite, key=lambda t: t[1])[0]
            wins[roots[best][0]] += 1
        cells = []
        for i, g in enumerate(got):
            if g is None:
                cells.append(f'{"--":>{cw}}')
                continue
            m, se, n = g
            star = '*' if i == best else ' '
            cells.append(f'{star}{m:.3f} +/- {se:.3f} [{n}]'.rjust(cw))
        print(f'{ds:<{w}}' + ''.join(cells))

    print('-' * (w + cw * len(roots)))
    print(f'{"rows won":<{w}}' + ''.join(
        f'{wins[n]:>{cw}}' for n, _ in roots))
    print('\nLower is better. A blank (--) means no npz found for that '
          'dataset under that root (job still running or not submitted).')


if __name__ == '__main__':
    main()
