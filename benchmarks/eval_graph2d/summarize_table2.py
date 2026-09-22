"""Table 2 rows: NLL of each density object, IHDP and ACIC side by side.

    f_Y0 + f_Y1   summed arm NLL, -log p(y0 | do(0), x) - log p(y1 | do(1), x)
    f_tau         NLL of p(tau | x)
    f_ATE         NLL of the W2-barycenter ATE density

f_ATE stays '–' until results_density_ate/ is populated by
benchmarks/cluster/submit_density_ate.sbatch.

The arm sum is formed PER REALIZATION and only then averaged, so its SE is the
SE of the summed quantity -- the two arms are scored on the same realizations
and are not independent, so adding the per-arm SEs in quadrature would be wrong.

Usage:
    python benchmarks/eval_graph2d/summarize_table2.py
"""
from __future__ import annotations

import glob
import os

import numpy as np

from summarize_density_marginals import ROWS


def _series(pattern, keys, method):
    """Per-realization values of sum(keys) for one method."""
    vals = []
    for f in sorted(glob.glob(pattern)):
        z = np.load(f, allow_pickle=True)
        if f'{keys[0]}_{method}' not in z.files:
            continue
        vals.append(sum(float(z[f'{k}_{method}']) for k in keys))
    return np.asarray(vals)


def cell(v):
    if v.size == 0:
        return '–'
    se = v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0
    return f'{v.mean():.4f}±{se:.4f}'


def main():
    out = []
    for label, run, shard, method in ROWS:
        row = [label]
        for ds in ('IHDP', 'ACIC'):
            marg = os.path.join('results_density_marginals', run, ds,
                                f'{ds}_r[0-9]*.npz')
            tau = os.path.join('results_density_tauC', shard, ds,
                               f'{ds}_r[0-9]*.npz')
            ate = os.path.join('results_density_ate', run, ds,
                               f'{ds}_r[0-9]*.npz')
            row.append(cell(_series(marg, ('nll_y0', 'nll_y1'), method)))
            row.append(cell(_series(tau, ('nll',), method)))
            # nll_bary == nll_mix by construction (-log p_est(ATE_true) reads
            # only the estimate), so the truth tag here is arbitrary.
            row.append(cell(_series(ate, ('nll_bary',), method)))
        out.append(row)

    head = ['Method', 'IHDP: f_Y0 + f_Y1 ↓', 'IHDP: f_τ ↓', 'IHDP: f_ATE ↓',
            'ACIC: f_Y0 + f_Y1 ↓', 'ACIC: f_τ ↓', 'ACIC: f_ATE ↓']
    w = [max(len(h), *(len(r[i]) for r in out)) for i, h in enumerate(head)]
    print(' | '.join(h.ljust(w[i]) for i, h in enumerate(head)))
    print('-|-'.join('-' * x for x in w))
    for r in out:
        print(' | '.join(c.ljust(w[i]) for i, c in enumerate(r)))


if __name__ == '__main__':
    main()
