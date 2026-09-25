"""One table per (tier, dataset) over the same model list.

  --tier marginals   p(Y | do(T=t), x), from eval_density_marginals.py
  --tier tau         p(tau | x), recomputed from the eval_density_tauC.py
                     per-realization NPZs. Reproduces the nll/l2 already in
                     FOR_FURKAN.md to 4 dp and carries `mass` along, which the
                     live tau tables drop into their commented-out blocks.

Aggregation is the same on both tiers: each NPZ holds one realization's mean
over queries, and these are the mean +- SE across realizations.

Usage:
    python benchmarks/eval_graph2d/summarize_density_marginals.py \
        --tier tau --dataset IHDP
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

# (label, marginals run dir, tauC shard, method key in both tiers)
ROWS = [
    ('Do-PFN',                        'dopfn_refresh', 'dopfn_refresh', 'dopfn_native'),
    ('Do-PFN 2D',                     'dopfn_refresh', 'dopfn_refresh', 'dopfn_repro_joint2d'),
    ('UWYK No-Anc',                   'uwyk_noanc', '5312884',  'uwyk_native'),
    ('UWYK No-Anc 2D',                'uwyk_noanc', '5312884',  'joint'),
    ('UWYK Anc (v3a)',                'uwyk_v3a',   '5312882',  'uwyk_native'),
    ('UWYK Anc 2D (v3a)',             'uwyk_v3a',   '5312882',  'joint'),
    ('CausalPFN-C j1024_headrand 1D', 'causalpfn',  '5571187',  'causalpfn_j1024_headrand_1d'),
    ('CausalPFN-C j32 1D',            'causalpfn',  '5571187',  'causalpfn_j32_1d'),
    ('CausalPFN-C botharms 1D',       'causalpfn',  '5571187',  'causalpfn_botharms_1d'),
    ('CausalPFN-C j32_random 2D',     'causalpfn',  '5571187',  'causalpfn_j32_random_2d'),
    ('CausalPFN-C j32_eta0_y01 2D',   'causalpfn',  '5571187',  'causalpfn_j32_eta0_y01_2d'),
]
TIERS = {
    'marginals': dict(root='results_density_marginals',
                      cols=('nll_y0', 'nll_y1', 'l2_y0', 'l2_y1', 'mass_y0', 'mass_y1'),
                      title='marginals p(Y | do(T=t), x)'),
    'tau':       dict(root='results_density_tauC',
                      cols=('nll', 'l2', 'mass'),
                      title='CATE density p(tau | x)'),
}


def collect(pattern, method, cols):
    acc, n = {c: [] for c in cols}, 0
    for f in sorted(glob.glob(pattern)):
        z = np.load(f, allow_pickle=True)
        if f'{cols[0]}_{method}' not in z.files:
            continue
        n += 1
        for c in cols:
            acc[c].append(float(z[f'{c}_{method}']))
    if not n:
        return None, 0
    return {c: (np.mean(v), np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
            for c, v in acc.items()}, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', choices=sorted(TIERS), default='marginals')
    ap.add_argument('--dataset', required=True)
    a = ap.parse_args()
    t = TIERS[a.tier]
    cols = t['cols']

    head = ['method', 'n'] + list(cols)
    print(f'\n### {a.dataset} — {t["title"]}\n')
    print('| ' + ' | '.join(head) + ' |')
    print('| ' + ' | '.join(['---'] + ['---:'] * (len(head) - 1)) + ' |')
    for label, run, shard, method in ROWS:
        key = run if a.tier == 'marginals' else shard
        sub = a.dataset if a.tier == 'marginals' else os.path.join(a.dataset)
        pattern = os.path.join(t['root'], key, sub, f'{a.dataset}_r[0-9]*.npz')
        stats, n = collect(pattern, method, cols)
        if stats is None:
            print(f'| {label} | – |' + ' – |' * len(cols))
            continue
        cells = [f'{stats[c][0]:.4f}±{stats[c][1]:.4f}' for c in cols]
        print(f'| {label} | {n} | ' + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    main()
