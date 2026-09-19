#!/usr/bin/env python3
"""Aggregate the Tier-C density npz shards into the comparison table.

Aggregation is mean over queries within a realization (done in the eval), then
mean +/- SE OVER REALIZATIONS here -- not pooled over all queries, which would
treat queries from one realization as independent and understate the SE.

    python benchmarks/eval_graph2d/summarize_density_tauC.py results_density_tauC

Output is markdown (one model-family table plus its contrasts per dataset), so
it can be pasted straight into LATEST_RESULTS.md; mixed shards are separated
into UWYK, DoPFN and CausalPFN sections. Use --model to select one family.
DoPFN rows are whatever DOPFN_MODELS listed for the run, read from the shards;
its contrasts pair every 1D row with every joint row. The columns are
padded so they stay readable as plain terminal text too. The best cell is bolded --
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

# DoPFN has no fixed rows: family_methods reads them from the shards.
MODEL_METHODS = {
    'uwyk': ('uwyk_native', 'uwyk_matched', 'joint'),
    'dopfn': (),
    'causalpfn': ('causalpfn_native', 'causalpfn_joint'),
}
MODEL_LABEL = {'uwyk': 'UWYK / g4cfm', 'dopfn': 'DoPFN', 'causalpfn': 'CausalPFN'}
LABEL = {'uwyk_native': 'UWYK (x)indep K=1000',
         'uwyk_matched': 'UWYK (x)indep matched bins',
         'joint': 'UWYK Joint-2D',
         'dopfn_native': 'DoPFN (x)indep native',
         'dopfn_joint': 'DoPFN Joint-2D',
         'causalpfn_native': 'CausalPFN (x)indep native',
         'causalpfn_joint': 'CausalPFN Joint-2D',
         'joint_inner': 'Joint-2D interior mean (raw)'}
# All four are errors or diagnostics; lower is better except `mass`, which
# should sit at 1.0 and is a grid-coverage check, not a score.
METRICS = ('nll', 'l2', 'kl_fwd', 'kl_rev', 'mass')
POINT_METRICS = ('pehe', 'cate_l1', 'ate_abs_err')
MISSING = '--'


def dopfn_methods(rows):
    """DoPFN density rows, in the order the run listed them."""
    found = []
    for r in rows:
        if 'methods' in r:
            names = [str(m) for m in np.atleast_1d(r['methods'])]
        else:       # hand-built rows carry only metric keys
            names = [k.split('_', 1)[1] for k in r if k.startswith(('nll_', 'pehe_'))]
        for m in names:
            if m.startswith('dopfn_') and not m.endswith('_inner') and m not in found:
                found.append(m)
    return found


def family_methods(rows, model):
    methods = dopfn_methods(rows) if model == 'dopfn' else MODEL_METHODS[model]
    return [m for m in methods if any(f'nll_{m}' in r for r in rows)]


def dopfn_kind(rows, method):
    """'1d' or 'joint'. Shards from before DOPFN_MODELS carry no kind_ field."""
    kinds = {str(r[f'kind_{method}']) for r in rows if f'kind_{method}' in r}
    if len(kinds) > 1:
        raise ValueError(f'{method} is {sorted(kinds)} across shards; the name was '
                         'reused for different models. Use separate result directories.')
    if kinds:
        return kinds.pop()
    return 'joint' if method == 'dopfn_joint' else '1d'


def label(rows, method):
    if method in LABEL:
        return LABEL[method]
    if method.endswith('_inner'):
        return f'{label(rows, method[:-len("_inner")])} interior mean (raw)'
    if method.startswith('dopfn_'):
        kind = dopfn_kind(rows, method)
        return (f'DoPFN {method[len("dopfn_"):]} '
                + ('(x)indep' if kind == '1d' else 'Joint-2D'))
    return method


def load(results_dir: Path, dataset: str):
    rows = []
    for path in sorted((results_dir / dataset).glob(f'{dataset}_r*.npz')):
        with np.load(path) as z:
            rows.append({k: z[k] for k in z.files})
    return rows


def mean_se(values):
    v = np.asarray([float(x) for x in values], dtype=np.float64)
    # A finite-support model can legitimately assign zero density to tau*.
    # Never turn an infinite mean NLL into a finite mean by dropping that run.
    if np.isinf(v).any():
        if np.isposinf(v).any() and np.isneginf(v).any():
            return float('nan'), float('nan'), int(v.size)
        return (float('inf') if np.isposinf(v).any() else -float('inf'),
                float('nan'), int(v.size))
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float('nan'), float('nan'), 0
    if v.size == 1:
        return float(v[0]), float('nan'), 1
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)), int(v.size)


def truth_summary(rows):
    """Identify the reference and keep old/new L2 and KL scores separate."""
    sources = {str(r.get('truth_noise_source', 'training_residuals')) for r in rows}
    if len(sources) != 1:
        raise ValueError('Mixed density truth sources in one dataset directory: '
                         f'{sorted(sources)}. Use separate result directories '
                         'or recompute all realizations with the same reference.')
    source = sources.pop()
    if source != 'generator':
        return 'Truth: legacy Gaussian reference with training-residual sigma.'
    sigmas = {float(r['sigma_raw']) for r in rows}
    if len(sigmas) != 1:
        raise ValueError('Generator noise scales differ within one dataset directory')
    residuals = np.asarray([float(r['sigma_residual_raw']) for r in rows])
    return (f'Truth: generator Gaussian, raw sigma={sigmas.pop():g}. '
            f'Training-residual sigma (diagnostic, raw units): '
            f'mean={residuals.mean():.4f}, '
            f'range=[{residuals.min():.4f}, {residuals.max():.4f}].')


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


def point_table(rows, density_methods=None):
    """Point errors from the same logits; old density-only shards stay valid."""
    if density_methods is None:
        density_methods = [*MODEL_METHODS['uwyk'], *dopfn_methods(rows),
                           *MODEL_METHODS['causalpfn']]
    point_methods = [*density_methods, *(m + '_inner' for m in density_methods)]
    if not any(f'pehe_{m}' in r for r in rows for m in point_methods):
        return None
    methods = [m for m in point_methods if any(f'pehe_{m}' in r for r in rows)]
    body = [[label(rows, m)] for m in methods]
    for metric in POINT_METRICS:
        cells, scores = [], []
        for method in methods:
            key = f'{metric}_{method}'
            if not all(key in r for r in rows):
                cells.append(MISSING)
                scores.append(None)
                continue
            mu, se, _ = mean_se([r[key] for r in rows])
            cells.append(f'{mu:.4f}±{se:.4f}')
            scores.append(mu)
        for row, cell in zip(body, bold_best(cells, scores)):
            row.append(cell)
    return md_table(['mean estimator', 'sqrt PEHE', 'CATE L1', 'ATE abs error'],
                    body, ['l', 'r', 'r', 'r'])


def render_family(dataset, rows, model):
    """Render one model family so unrelated backbones never share a table."""
    methods = family_methods(rows, model)
    print(f'\n### {dataset} — {MODEL_LABEL[model]}\n')
    if not methods:
        print(f'_no {MODEL_LABEL[model]} results found_')
        return

    print(truth_summary(rows) + '\n')
    n_r = len(rows)
    n_q = int(np.mean([float(r['n_queries']) for r in rows]))
    oob = float(np.mean([float(r['frac_tau_outside_grid']) for r in rows]))
    graph = f'graph={rows[0]["anc_tag"]}' if model == 'uwyk' else 'graph=none'
    print(f'realizations={n_r}, ~{n_q} queries each, {graph}, '
          f'|tau*|>3: {oob:.2%}\n')
    if model == 'dopfn':
        for method in methods:
            sources = sorted({str(r[f'source_{method}']) for r in rows
                              if f'source_{method}' in r})
            if sources:
                mixed = ' **(MIXED CHECKPOINTS)**' if len(sources) > 1 else ''
                print(f'- `{method}`: {", ".join(sources)}{mixed}')
        print()

    table = {}
    body = [[label(rows, m)] for m in methods]
    for metric in METRICS:
        cells, scores = [], []
        for method in methods:
            key = f'{metric}_{method}'
            if not all(key in r for r in rows):
                cells.append(MISSING)
                scores.append(None)
                continue
            mu, se, _ = mean_se([r[key] for r in rows])
            table[(method, metric)] = mu
            cells.append(f'{mu:.4f}±{se:.4f}')
            scores.append(score(metric, mu))
        for row_cells, cell in zip(body, bold_best(cells, scores)):
            row_cells.append(cell)
    print(md_table(['method', *METRICS], body,
                   ['l'] + ['r'] * len(METRICS)))
    if model == 'causalpfn':
        key = 'frac_zero_density_causalpfn_native'
        if all(key in r for r in rows):
            fraction = np.mean([float(r[key]) for r in rows])
            print(f'\nNative finite support: zero density at tau* in {fraction:.2%} '
                  'of queries (mean over realizations); NLL retains +inf. '
                  'Grid KL uses the shared numerical density floor.')

    points = point_table(rows, methods)
    if points is not None:
        print('\nPoint errors in original outcome units, from the same '
              'predictions. Full-density means except the interior row; '
              'CATE L1 is per-query MAE, ATE error is unnormalised.\n')
        print(points)
        large_grid_mean_gap = []
        for method in methods:
            key = f'grid_mean_max_abs_diff_{method}'
            if all(key in r for r in rows):
                gap = max(float(r[key]) for r in rows)
                print(f'  {method}: max |finite-grid moment - full mean| = {gap:.6g}')
                if gap > 1.0:
                    large_grid_mean_gap.append(method)
        if large_grid_mean_gap:
            print('\n**TAIL NOTE:** The finite tau grid omits distant tail '
                  f'mass for {large_grid_mean_gap}. NLL is evaluated at the '
                  'observed tau and point errors use exact full-density means; '
                  'L2/KL/mass are finite-grid quantities. In particular, '
                  'KL_rev is not the full-support reverse KL when omitted tail '
                  'mass lies far from the truth.')

    def delta(a, b, metric):
        ka, kb = f'{metric}_{a}', f'{metric}_{b}'
        if not all(ka in r and kb in r for r in rows):
            return MISSING, None
        mu, se, _ = mean_se([float(r[kb]) - float(r[ka]) for r in rows])
        return f'{mu:+.4f}±{se:.4f}', mu

    candidates = {
        'uwyk': [('**HEADLINE**', 'model gap as run', 'uwyk_native', 'joint'),
                 ('bridge', 'resolution handicap', 'uwyk_native', 'uwyk_matched')],
        'dopfn': [('as run', 'model gap', a, b)
                  for a in methods if dopfn_kind(rows, a) == '1d'
                  for b in methods if dopfn_kind(rows, b) == 'joint'],
        'causalpfn': [('**HEADLINE**', 'model gap as run',
                       'causalpfn_native', 'causalpfn_joint')],
    }
    contrasts = [c for c in candidates[model] if c[2] in methods and c[3] in methods]
    if contrasts:
        cbody = [[kind, f'{what} ({a} -> {b})']
                 for kind, what, a, b in contrasts]
        contrast_metrics = ('nll', 'kl_rev', 'pehe') if points else ('nll', 'kl_rev')
        for metric in contrast_metrics:
            cells, scores = zip(*[delta(a, b, metric)
                                  for _, _, a, b in contrasts])
            for row_cells, cell in zip(cbody, bold_best(cells, scores)):
                row_cells.append(cell)
        print()
        print(md_table(['', 'contrast', 'dNLL', 'dKLrev']
                       + (['dPEHE'] if points else []), cbody,
                       ['l', 'l'] + ['r'] * len(contrast_metrics)))
        print('\n_negative = destination method has lower error._')
        if model == 'uwyk' and 'uwyk_matched' in methods:
            print('_The bridge measures the effect of rebinning UWYK._')
        if model == 'dopfn':
            print('_DoPFN rows differ in head resolution (and native DoPFN in '
                  'preprocessing); these are as-run comparisons._')

    bad = [m for m in methods
           if (m, 'mass') in table and abs(table[(m, 'mass')] - 1) > 0.01]
    if bad:
        print(f'\n**WARNING:** p(tau) mass off 1.0 by >1% for {bad} -- the '
              f'tau grid is clipping real density; widen TAU_EDGES.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results_dir', type=Path)
    ap.add_argument('--datasets', nargs='*', default=['IHDP', 'ACIC'])
    ap.add_argument(
        '--model', choices=('auto', 'all', *MODEL_METHODS), default='auto',
        help='Model family to display. auto/all render each available family '
             'in a separate table (default).')
    args = ap.parse_args()

    for dataset in args.datasets:
        rows = load(args.results_dir, dataset)
        if not rows:
            print(f'\n### {dataset}\n\n_no shards found_')
            continue
        available = [model for model in MODEL_METHODS
                     if family_methods(rows, model)]
        selected = available if args.model in ('auto', 'all') else [args.model]
        for model in selected:
            render_family(dataset, rows, model)


if __name__ == '__main__':
    main()
