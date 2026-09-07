#!/usr/bin/env python3
"""Aggregate the Tier-C density npz shards into the comparison table.

Aggregation is mean over queries within a realization (done in the eval), then
mean +/- SE OVER REALIZATIONS here -- not pooled over all queries, which would
treat queries from one realization as independent and understate the SE.

    python benchmarks/eval_graph2d/summarize_density_tauC.py results_density_tauC

Output is markdown (per-method table + contrast table per dataset), so it can be
pasted straight into LATEST_RESULTS.md; the columns are padded so it stays
readable as plain terminal text too.  The best cell in each column is bolded --
lowest value for the error metrics and the contrasts, closest to 1.0 for `mass`
-- and a column whose entries all tie at display precision gets no bold, since
there is no winner to mark.

The headline contrast is uwyk_native -> joint: each model as it is actually
run, no handicap imposed on either.  uwyk_matched is a bridge benchmark only --
it exists to show that regridding UWYK to J=32 costs it essentially nothing, so
the headline gap cannot be blamed on resolution.  Read it as a control, not as
a competitor.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

METHODS = ('uwyk_native', 'uwyk_matched', 'joint')
LABEL = {'uwyk_native': 'UWYK (x)indep K=1000',
         'uwyk_matched': 'UWYK (x)indep J=32',
         'joint': 'Joint-2D J=32'}
# All four are errors or diagnostics; lower is better except `mass`, which
# should sit at 1.0 and is a grid-coverage check, not a score.
METRICS = ('nll', 'l2', 'kl_fwd', 'kl_rev', 'mass')
MISSING = '--'


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


def score(metric, mu):
    """Lower is better everywhere except `mass`, which is a coverage check
    scored by its distance from 1.0 rather than by being small."""
    return abs(mu - 1.0) if metric == 'mass' else mu


def bold_best(cells, scores, ndigits=4):
    """Bold every cell attaining the lowest score in the column.

    Cells with no score (missing metric) are left alone, and a column whose
    finite scores all agree to `ndigits` is left unbolded -- everything tied at
    the printed precision means there is nothing to single out.
    """
    finite = [round(s, ndigits) for s in scores if s is not None and np.isfinite(s)]
    if not finite or min(finite) == max(finite):
        return list(cells)
    best = min(finite)
    return [f'**{c}**' if s is not None and np.isfinite(s)
            and round(s, ndigits) == best else c
            for c, s in zip(cells, scores)]


def md_table(headers, rows, aligns=None):
    """Render a GitHub-flavoured markdown table, padded to fixed columns."""
    ncol = len(headers)
    aligns = list(aligns) if aligns is not None else ['l'] * ncol
    width = [max(3, len(headers[i]), *(len(r[i]) for r in rows)) if rows
             else max(3, len(headers[i])) for i in range(ncol)]

    def row(cells):
        pad = [cells[i].rjust(width[i]) if aligns[i] == 'r'
               else cells[i].ljust(width[i]) for i in range(ncol)]
        return '| ' + ' | '.join(pad) + ' |'

    sep = ['-' * (width[i] - 1) + ':' if aligns[i] == 'r' else '-' * width[i]
           for i in range(ncol)]
    return '\n'.join([row(headers), '| ' + ' | '.join(sep) + ' |']
                     + [row(r) for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results_dir', type=Path)
    ap.add_argument('--datasets', nargs='*', default=['IHDP', 'ACIC'])
    args = ap.parse_args()

    for dataset in args.datasets:
        rows = load(args.results_dir, dataset)
        print(f'\n### {dataset}\n')
        if not rows:
            print('_no shards found_')
            continue
        n_r = len(rows)
        n_q = int(np.mean([float(r['n_queries']) for r in rows]))
        oob = float(np.mean([float(r['frac_tau_outside_grid']) for r in rows]))
        print(f'realizations={n_r}, ~{n_q} queries each, '
              f'anc={rows[0]["anc_tag"]}, |tau*|>3: {oob:.2%}\n')

        table = {}
        body = [[LABEL[m]] for m in METHODS]
        for metric in METRICS:
            cells, scores = [], []
            for m in METHODS:
                key = f'{metric}_{m}'
                if key not in rows[0]:
                    cells.append(MISSING)
                    scores.append(None)
                    continue
                mu, se, _ = mean_se([r[key] for r in rows])
                table[(m, metric)] = mu
                cells.append(f'{mu:.4f}±{se:.4f}')
                scores.append(score(metric, mu))
            for row_cells, cell in zip(body, bold_best(cells, scores)):
                row_cells.append(cell)
        print(md_table(['method', *METRICS], body,
                       ['l'] + ['r'] * len(METRICS)))

        # The contrasts the design exists to produce.  Paired over realizations:
        # the mean is identical to the difference of the column means, but the
        # SE is the SE of the within-realization difference, which is far
        # tighter than the two column SEs suggest.
        def delta(a, b, metric):
            ka, kb = f'{metric}_{a}', f'{metric}_{b}'
            if ka not in rows[0] or kb not in rows[0]:
                return MISSING, None
            mu, se, _ = mean_se([float(r[kb]) - float(r[ka]) for r in rows])
            return f'{mu:+.4f}±{se:.4f}', mu

        # The two rows are not competitors -- bolding here just marks the
        # larger improvement, which is the headline unless the bridge has
        # stopped being a ~0 control.
        contrasts = [('**HEADLINE**', 'model gap as run',
                      'uwyk_native', 'joint'),
                     ('bridge', 'resolution handicap',
                      'uwyk_native', 'uwyk_matched')]
        cbody = [[kind, f'{what} ({a} -> {b})']
                 for kind, what, a, b in contrasts]
        for metric in ('nll', 'kl_rev'):
            cells, scores = zip(*[delta(a, b, metric)
                                  for _, _, a, b in contrasts])
            for row_cells, cell in zip(cbody, bold_best(cells, scores)):
                row_cells.append(cell)

        print()
        print(md_table(['', 'contrast', 'dNLL', 'dKLrev'], cbody,
                       ['l', 'l', 'r', 'r']))
        print()
        print('_negative = joint better; the bridge row should be ~0, which '
              'is what licenses reading the headline as a model gap and not a '
              'resolution artefact._')

        bad = [m for m in METHODS
               if (m, 'mass') in table and abs(table[(m, 'mass')] - 1) > 0.01]
        if bad:
            print(f'\n**WARNING:** p(tau) mass off 1.0 by >1% for {bad} -- the '
                  f'tau grid is clipping real density; widen TAU_EDGES.')


if __name__ == '__main__':
    main()
