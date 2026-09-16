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


def load_by_real(root, kind, dataset, metric):
    """{realization index: value} so models can be compared PAIRWISE."""
    d = os.path.join(root, kind, dataset)
    out = {}
    for path in sorted(glob.glob(os.path.join(d, f'{dataset}_r*.npz'))):
        base = os.path.splitext(os.path.basename(path))[0]
        digits = ''.join(c for c in base.split('_r')[-1] if c.isdigit())
        if not digits:
            continue
        try:
            with np.load(path, allow_pickle=True) as z:
                if metric in z.files:
                    v = float(np.asarray(z[metric]).reshape(-1)[0])
                    if np.isfinite(v):
                        out[int(digits)] = v
        except Exception:
            continue
    return out


def load(root, kind, dataset, metric):
    vals = load_by_real(root, kind, dataset, metric)
    a = np.asarray(list(vals.values()), dtype=float)
    if a.size == 0:
        return None
    se = float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
    return float(a.mean()), se, int(a.size)


def paired(root_a, root_b, kind, dataset, metric):
    """Mean +/- SE of the per-realization difference (a - b).

    Both models score the SAME realizations, so the paired difference removes
    the realization-to-realization variance that dominates the unpaired SE.
    Two models can look statistically indistinguishable unpaired and be
    separated cleanly once paired -- and vice versa.
    """
    A = load_by_real(root_a, kind, dataset, metric)
    B = load_by_real(root_b, kind, dataset, metric)
    shared = sorted(set(A) & set(B))
    if len(shared) < 2:
        return None
    d = np.asarray([A[r] - B[r] for r in shared], dtype=float)
    se = float(d.std(ddof=1) / np.sqrt(d.size))
    t = float(d.mean() / se) if se > 0 else float('inf')
    return float(d.mean()), se, int(d.size), t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', action='append', required=True,
                    help='NAME=PATH, repeatable')
    ap.add_argument('--type', default='case_studies',
                    choices=('case_studies', 'realcause'))
    ap.add_argument('--metric', default='pehe_raw')
    ap.add_argument('--paired', action='store_true',
                    help='also print the per-realization paired difference of '
                         'the first two roots (the correct test when both '
                         'models scored the same realizations)')
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

    if a.paired and len(roots) >= 2:
        (na, pa), (nb, pb) = roots[0], roots[1]
        print(f'\nPAIRED  {na} - {nb}   (same realizations; negative favours '
              f'{na})\n')
        print(f'{"dataset":<{w}}{"mean diff":>16}{"SE":>10}{"t":>8}{"n":>6}'
              f'   verdict')
        print('-' * (w + 52))
        for ds in datasets:
            r = paired(pa, pb, a.type, ds, a.metric)
            if r is None:
                print(f'{ds:<{w}}{"--":>16}')
                continue
            m, se, n, t = r
            # |t| > 2 is roughly the 5% two-sided threshold at these n
            if abs(t) < 2:
                verdict = 'tie (|t| < 2)'
            else:
                verdict = f'{na} better' if m < 0 else f'{nb} better'
            print(f'{ds:<{w}}{m:>+16.4f}{se:>10.4f}{t:>8.2f}{n:>6}   {verdict}')
        print('-' * (w + 52))


if __name__ == '__main__':
    main()
