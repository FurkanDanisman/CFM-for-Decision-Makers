"""Audit and refresh the seven density tables in FOR_FURKAN.md.

Run from the repository root with its NumPy/SciPy environment:
    python benchmarks/eval_graph2d/refresh_density_tables.py
    python benchmarks/eval_graph2d/refresh_density_tables.py --fill-missing --workers 4 --write

Only marginal/ATE postprocessing runs; no model inference. Existing derived
files must be complete for the displayed methods. Do-PFN uses a separate
dopfn_refresh output directory so old predictions' scores cannot be reused.
The CausalPFN repair only added missing realizations to its original shard.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import re

# Also limit Apple's Accelerate BLAS when workers run on a Mac.
for _env in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS'):
    os.environ[_env] = '1'

import numpy as np

from summarize_density_marginals import ROWS, TIERS
from summarize_density_ate import METRICS as ATE_METRICS
from density_common import TAU_CENTERS, Y_MARG

DATASETS = {'IHDP': 100, 'ACIC': 10}
COLS = {'marginals': TIERS['marginals']['cols'],
        'tau': TIERS['tau']['cols'],
        'ate': tuple(f'{m}_bary' for m in ATE_METRICS)}
ROOTS = {'marginals': 'results_density_marginals',
         'tau': 'results_density_tauC', 'ate': 'results_density_ate'}
TRUTH_KEYS = ('mu0_scaled', 'mu1_scaled', 'tau_star_scaled', 'sigma_scaled',
              'y_scale', 'y_shift', 'true_cate', 'context_seed', 'n_context',
              'tau_grid', 'n_y0')


def runs():
    grouped = {}
    for _label, run, shard, method in ROWS:
        grouped.setdefault((run, shard), []).append(method)
    return grouped


def audit_inputs():
    """Check exact realization sets, matched query truth/scaling and CATE scores."""
    metadata = {}
    for ds, n in DATASETS.items():
        expected = {f'{ds}_r{r:03d}.npz' for r in range(n)}
        ref = {}
        for (run, shard), methods in runs().items():
            base = Path(ROOTS['tau'], shard, ds)
            for folder in (base, base / 'predictions'):
                actual = {p.name for p in folder.glob(f'{ds}_r*.npz')}
                if actual != expected:
                    raise ValueError(f'{folder}: missing {sorted(expected - actual)}, '
                                     f'extra {sorted(actual - expected)}')
            for name in sorted(expected):
                with np.load(base / 'predictions' / name) as z:
                    truth = {k: z[k] for k in TRUTH_KEYS}
                    if any(not np.isfinite(v).all() for v in truth.values()):
                        raise ValueError(f'{base}/predictions/{name}: nonfinite truth/configuration')
                    r = int(z['realization'])
                    if str(z['dataset']) != ds or name != f'{ds}_r{r:03d}.npz':
                        raise ValueError(f'{base}/predictions/{name}: wrong identity')
                    if not np.array_equal(z['tau_grid'], TAU_CENTERS):
                        raise ValueError(f'{base}/predictions/{name}: unexpected tau grid')
                    if name in ref:
                        for k in TRUTH_KEYS:
                            if (truth[k].shape != ref[name][k].shape or
                                    not np.allclose(truth[k], ref[name][k],
                                                    rtol=1e-9, atol=1e-12)):
                                raise ValueError(f'{base}/{name}: unmatched {k}')
                    else:
                        ref[name] = truth
                    metadata[run, ds, r] = (
                        len(z['mu0_scaled']), float(z['y_scale']),
                        float(z['sigma_scaled']), float(z['y_shift']),
                        float((z['mu1_scaled'] - z['mu0_scaled']).mean()), int(z['n_y0']))
                with np.load(base / name) as z:
                    validate(z, methods, COLS['tau'], ds, r, metadata[run, ds, r])
            print(f'{run}/{ds}: {n}/{n} predictions and CATE scores; matched inputs', flush=True)
    return metadata


def validate(z, methods, cols, ds, r, meta):
    if (str(z['dataset']) != ds or int(z['realization']) != r or
            int(z['n_queries']) != meta[0] or
            not np.isclose(float(z['y_scale']), meta[1], rtol=1e-12, atol=0)):
        raise ValueError(f'{z.zip.filename}: identity/query count/scaling mismatch')
    if 'nll_y0' in cols:
        if (not np.array_equal(z['y_grid'], Y_MARG) or
                not np.isclose(z['sigma_scaled'], meta[2], rtol=1e-12, atol=0) or
                not np.isclose(z['y_shift'], meta[3], rtol=1e-12, atol=1e-12)):
            raise ValueError(f'{z.zip.filename}: marginal grid/truth mismatch')
    if 'nll_bary' in cols:
        if (not np.array_equal(z['tau_grid'], TAU_CENTERS) or
                not np.isclose(z['ate_true_scaled'], meta[4], rtol=1e-12, atol=1e-12) or
                int(z['n_y0']) != meta[5]):
            raise ValueError(f'{z.zip.filename}: ATE grid/truth/quadrature mismatch')
    for method in methods:
        for col in cols:
            value = z[f'{col}_{method}']
            if value.size != 1 or not np.isfinite(value).all():
                raise ValueError(f'{z.zip.filename}: invalid {col}_{method}')


def missing_work(meta):
    work = []
    for (run, shard), methods in runs().items():
        for ds, n in DATASETS.items():
            for tier in ('marginals', 'ate'):
                base = Path(ROOTS[tier], run, ds)
                expected = {f'{ds}_r{r:03d}.npz' for r in range(n)}
                extra = {p.name for p in base.glob(f'{ds}_r*.npz')} - expected
                if extra:
                    raise ValueError(f'{base}: unexpected realizations {sorted(extra)}')
                missing = []
                for r in range(n):
                    name = f'{ds}_r{r:03d}.npz'
                    dest = base / name
                    if dest.exists():
                        with np.load(dest) as z:
                            validate(z, methods, COLS[tier], ds, r, meta[run, ds, r])
                        continue
                    missing.append(r)
                    pred = Path(ROOTS['tau'], shard, ds, 'predictions', name)
                    work.append((tier, str(pred), str(dest), methods))
                print(f'{run}/{ds}/{tier}: {n - len(missing)}/{n}' +
                      (f'; missing {missing}' if missing else ''), flush=True)
    return work


def compute(task):
    tier, pred, dest, methods = task
    if tier == 'marginals':
        from eval_density_marginals import run_realization
        result = run_realization(pred)
    else:
        from eval_density_ate import run_realization
        result = run_realization(pred, only=set(methods))
    # A killed worker must not leave a partial file that a resume trusts.
    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **result)
    tmp.replace(path)
    return f'{tier}: {dest}'


def collect_tables(meta):
    data = {}
    for tier, cols in COLS.items():
        for ds, n in DATASETS.items():
            for label, run, shard, method in ROWS:
                values = []
                for r in range(n):
                    path = Path(ROOTS[tier], shard if tier == 'tau' else run,
                                ds, f'{ds}_r{r:03d}.npz')
                    with np.load(path) as z:
                        validate(z, [method], cols, ds, r, meta[run, ds, r])
                        values.append([float(z[f'{c}_{method}']) for c in cols])
                        if tier == 'ate':
                            for metric in ('nll', 'ate_err', 'mass'):
                                if z[f'{metric}_bary_{method}'] != z[f'{metric}_mix_{method}']:
                                    raise ValueError(f'{path}: {metric} depends on truth convention')
                            if not np.isclose(z[f'mass_bary_{method}'], 1., atol=1e-12, rtol=0):
                                raise ValueError(f'{path}: unnormalised ATE density')
                data[tier, ds, label] = np.asarray(values)
    return data


def cell(values):
    return f'{values.mean():.4f}±{values.std(ddof=1) / np.sqrt(len(values)):.4f}'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] + ['---:'] * (len(headers) - 1)) + ' |'] +
                     ['| ' + ' | '.join(row) + ' |' for row in rows])


def update_document(data, write):
    path = Path('FOR_FURKAN.md')
    original = path.read_text()
    text = original
    replacements = {}
    headers = ['Method'] + [f'{ds}: {metric} ↓' for ds in DATASETS
                            for metric in ('f_Y0 + f_Y1', 'f_τ', 'f_ATE')]
    rows = []
    for label, *_ in ROWS:
        row = [label]
        for ds in DATASETS:
            # Sum arms per realization BEFORE calculating its standard error.
            row.extend([cell(data['marginals', ds, label][:, :2].sum(axis=1)),
                        cell(data['tau', ds, label][:, 0]),
                        cell(data['ate', ds, label][:, 0])])
        rows.append(row)
    replacements['# Summary Table'] = table(headers, rows)
    for tier, title in (('marginals', 'marginals'), ('tau', 'CATE density'), ('ate', 'ATE density')):
        for ds, n in DATASETS.items():
            headers = ['method', 'n'] + list(ATE_METRICS if tier == 'ate' else COLS[tier])
            rows = [[label, str(n)] + [cell(v) for v in data[tier, ds, label].T]
                    for label, *_ in ROWS]
            replacements[f'### {ds} — {title}'] = table(headers, rows)
    changed = 0
    for heading, replacement in replacements.items():
        start = text.index(heading + '\n') + len(heading) + 1
        # Also accept the original summary table without leading pipes.
        match = re.search(r'(?m)^\|?[ \t]*(?:Method|method)[ \t]*\|[^\n]*\n[^\n]*\n(?:[^\n]*\|[^\n]*\n?)+', text[start:])
        if not match:
            raise ValueError(f'No table found under {heading}')
        old = match.group().rstrip('\n')
        a, b = start + match.start(), start + match.start() + len(old)
        if old != replacement:
            changed += 1
            text = text[:a] + replacement + text[b:]
    if write:
        path.write_text(text)
    print(f'{changed} table(s) ' + ('updated.' if write else 'differ from FOR_FURKAN.md.'))
    return changed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fill-missing', action='store_true')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--write', action='store_true', help='replace tables only after every tier passes')
    args = ap.parse_args()
    meta = audit_inputs()
    tasks = missing_work(meta)
    if tasks and not args.fill_missing:
        raise SystemExit(f'{len(tasks)} derived files missing. Run with --fill-missing --write.')
    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(compute, task) for task in tasks]
            for i, future in enumerate(as_completed(futures), 1):
                print(f'[{i}/{len(tasks)}] {future.result()}', flush=True)
    data = collect_tables(meta)
    changed = update_document(data, args.write)
    print('Validated 11 methods × (100 IHDP + 10 ACIC) × 3 tiers; finite displayed '
          'metrics, matched realization sets and ATE truth conventions.')
    if changed and not args.write:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
