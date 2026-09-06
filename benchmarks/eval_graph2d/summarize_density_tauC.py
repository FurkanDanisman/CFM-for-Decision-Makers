#!/usr/bin/env python3
"""Aggregate the Tier-C density npz shards into the comparison table.

Aggregation is mean over queries within a realization (done in the eval), then
mean +/- SE OVER REALIZATIONS here -- not pooled over all queries, which would
treat queries from one realization as independent and understate the SE.

    python benchmarks/eval_graph2d/summarize_density_tauC.py results_density_tauC

Read the rows in this order:
  uwyk_native  -> uwyk_matched   the resolution handicap, isolated
  uwyk_matched -> joint          the model comparison at equal resolution
Do not compare uwyk_native to joint directly; that contrast mixes both.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

METHODS = ('uwyk_native', 'uwyk_matched', 'joint')
LABEL = {'uwyk_native': 'UWYK (x)indep  K=1000',
         'uwyk_matched': 'UWYK (x)indep  J=32',
         'joint': 'Joint-2D       J=32'}
# All four are errors or diagnostics; lower is better except `mass`, which
# should sit at 1.0 and is a grid-coverage check, not a score.
METRICS = ('nll', 'l2', 'kl_fwd', 'kl_rev', 'mass')


def load(results_dir: Path, dataset: str):
    rows = []
    for path in sorted((results_dir / dataset).glob(f'{dataset}_r*.npz')):
        with np.load(path) as z:
            rows.append({k: z[k] for k in z.files})
    return rows


def mean_se(values):
    v = np.asarray([float(x) for x in values], dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float('nan'), float('nan'), 0
    if v.size == 1:
        return float(v[0]), float('nan'), 1
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)), int(v.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results_dir', type=Path)
    ap.add_argument('--datasets', nargs='*', default=['IHDP', 'ACIC'])
    args = ap.parse_args()

    for dataset in args.datasets:
        rows = load(args.results_dir, dataset)
        if not rows:
            print(f'\n== {dataset}: no shards found ==')
            continue
        n_r = len(rows)
        n_q = int(np.mean([float(r['n_queries']) for r in rows]))
        oob = float(np.mean([float(r['frac_tau_outside_grid']) for r in rows]))
        print(f'\n== {dataset}   realizations={n_r}  ~{n_q} queries each  '
              f'anc={rows[0]["anc_tag"]}  |tau*|>3: {oob:.2%} ==')

        head = f'{"method":<24s}' + ''.join(f'{m:>18s}' for m in METRICS)
        print(head)
        print('-' * len(head))
        table = {}
        for m in METHODS:
            cells = []
            for metric in METRICS:
                key = f'{metric}_{m}'
                if key not in rows[0]:
                    cells.append(f'{"--":>18s}')
                    continue
                mu, se, _ = mean_se([r[key] for r in rows])
                table[(m, metric)] = mu
                cells.append(f'{mu:>10.4f}+-{se:<6.4f}')
            print(f'{LABEL[m]:<24s}' + ''.join(cells))

        # The two contrasts the design exists to produce.
        def delta(a, b, metric):
            if (a, metric) in table and (b, metric) in table:
                return table[(b, metric)] - table[(a, metric)]
            return float('nan')

        print()
        print(f'  resolution handicap (uwyk_native -> uwyk_matched):'
              f'  dNLL={delta("uwyk_native","uwyk_matched","nll"):+.4f}'
              f'  dKLrev={delta("uwyk_native","uwyk_matched","kl_rev"):+.4f}')
        print(f'  model gap at equal resolution (uwyk_matched -> joint):'
              f'  dNLL={delta("uwyk_matched","joint","nll"):+.4f}'
              f'  dKLrev={delta("uwyk_matched","joint","kl_rev"):+.4f}')
        print('  (negative = joint better; expect the joint to LOSE slightly '
              'if it carries a spurious rho -- the truth here factorises)')

        bad = [m for m in METHODS
               if (m, 'mass') in table and abs(table[(m, 'mass')] - 1) > 0.01]
        if bad:
            print(f'  WARNING: p(tau) mass off 1.0 by >1% for {bad} -- the tau '
                  f'grid is clipping real density; widen TAU_EDGES.')


if __name__ == '__main__':
    main()
