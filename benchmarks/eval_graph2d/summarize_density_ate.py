"""Aggregate eval_density_ate.py output into one table per dataset.

`--truth bary` (default) scores against the W2 barycenter of the TRUE per-query
CATE densities -- the same operator applied to the estimate, so it is the
apples-to-apples comparison. `--truth mix` uses the arithmetic mean instead,
the convention realcause_eval/eval_ate_density_metrics.py's docstring names.

`nll` is IDENTICAL under both: it is -log p_est(ATE_true), which reads only the
estimate. The truth convention moves l2, kl_fwd and kl_rev only.

Usage:
    python benchmarks/eval_graph2d/summarize_density_ate.py --dataset IHDP
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from summarize_density_marginals import ROWS

METRICS = ('nll', 'l2', 'kl_rev', 'mass', 'ate_err')


def collect(pattern, method, truth, metrics):
    acc, n = {m: [] for m in metrics}, 0
    for f in sorted(glob.glob(pattern)):
        z = np.load(f, allow_pickle=True)
        if f'nll_{truth}_{method}' not in z.files:
            continue
        n += 1
        for m in metrics:
            acc[m].append(float(z[f'{m}_{truth}_{method}']))
    if not n:
        return None, 0
    return {m: (np.mean(v), np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
            for m, v in acc.items()}, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='results_density_ate')
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--truth', choices=('bary', 'mix'), default='bary')
    a = ap.parse_args()

    head = ['method', 'n'] + list(METRICS)
    print(f'\n### {a.dataset} — ATE density p_ATE(τ)   [truth = {a.truth}]\n')
    print('| ' + ' | '.join(head) + ' |')
    print('| ' + ' | '.join(['---'] + ['---:'] * (len(head) - 1)) + ' |')
    for label, run, _shard, method in ROWS:
        pattern = os.path.join(a.root, run, a.dataset, f'{a.dataset}_r[0-9]*.npz')
        stats, n = collect(pattern, method, a.truth, METRICS)
        if stats is None:
            print(f'| {label} | – |' + ' – |' * len(METRICS))
            continue
        cells = [f'{stats[m][0]:.4f}±{stats[m][1]:.4f}' for m in METRICS]
        print(f'| {label} | {n} | ' + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    main()
