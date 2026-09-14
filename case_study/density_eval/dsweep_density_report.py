"""Density version of dsweep_report.py: the combined_cen3 table, but with
coverage / length / WIS / CRPS for CATE and ATE instead of PEHE / L1.

Same grid, same layout, same pooling, same row keys as dsweep_report.py:

    <root>/shift<S>/d<K>/ctx<N>/<model>/<case>/   ->  one row per
    (shift, d, N, case, model)

and --combine-shifts pools per-realization values across shift roots exactly as
the point report does, so `--combine-shifts shift0 shift-2 shift+2
--combine-label cen3` reproduces combined_cen3.csv's 2400 rows with density
columns.

REQUIRES DENSITY DUMPS. The point sweep writes PEHE/L1 only; p(tau|x) is
written solely when the eval runs with DENSITY_DUMP=1. Cells without a dump are
reported as missing rather than silently skipped.

Columns per row (each with _sem / _med as in the point report):
    cate_cov95 cate_len95 cate_wis cate_crps
    ate_cov95  ate_len95  ate_wis  ate_crps  ate_bias
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from interval_metrics import DEFAULT_LEVELS                       # noqa: E402

# Only alpha=0.05 is reported, so only that level is computed.
LEVELS = (0.05,)
from run_density_scm import score_realization, score_arrays        # noqa: E402
from density_truth import scm_true_cate                           # noqa: E402
from density_tauc import DIR_METHOD, load_predictions, is_tauc_prediction  # noqa: E402

# Directory names as 04_submit_density.sh writes them. NOTE these differ from
# dsweep_report.py's point-eval dirs: that sweep writes cpfn2d_pooled /
# cpfn1d_perarm (std mode baked into the name), the density sweep writes plain
# cpfn2d / cpfn1d. Both spellings are accepted so a mixed tree still resolves.
MODELS = ["dopfn_native", "dopfn_bb", "cpfn2d", "cpfn1d",
          "cpfn2d_pooled", "cpfn1d_perarm",
          "graph2d", "uwyk", "uwyk_v3a", "uwyk_noanc"]
# IS = interval (Winkler) score at alpha=0.05:
#     (hi-lo) + (2/alpha) * distance of the truth outside [lo, hi]
# Proper, so it cannot be gamed by widening or narrowing, and it is the single
# alpha the table reports. CRPS and WIS are no longer computed at all.
METRICS = ["cate_cov95", "cate_len95", "cate_is95",
           "ate_cov95", "ate_len95", "ate_is95", "ate_bias"]
_SUFFIX = ["", "_sem", "_med"]
_HEADER = ",".join(["shift", "d", "N", "case", "model"]
                   + [m + s for m in METRICS for s in _SUFFIX] + ["n"]) + "\n"
_A95 = 0.05


def _stats(v):
    a = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if a.size == 0:
        return [float('nan')] * 3
    sem = float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
    return [float(a.mean()), sem, float(np.median(a))]


MAX_Q = int(os.environ.get('MAX_QUERIES', '0'))     # 0 = all


def collect(cell, case, data_root, n_ctx, max_real=None):
    """Per-realization density metrics for one (model, case) dir."""
    model = os.path.basename(os.path.dirname(cell))
    # tauC models keep their DISTRIBUTIONS in predictions/; the top-level npz
    # holds per-method scores (nll/l2/pehe), not densities.
    pred_dir = os.path.join(cell, 'predictions')
    tauc = model in DIR_METHOD
    if tauc:
        # Raw-logit dumps live in predictions/ when that subdir exists, else at
        # the top level of the cell (graph2d/uwyk/dopfn write there). Select the
        # tauC prediction npz (tau_grid + *_logits), never the metrics-scalar
        # siblings -- so graph2d's joint_logits get scored via method 'joint'
        # instead of falling through to score_realization and raising KeyError.
        search = pred_dir if os.path.isdir(pred_dir) else cell
        paths = sorted(p for p in glob.glob(os.path.join(search, '*.npz'))
                       if is_tauc_prediction(p))
    else:
        paths = sorted(p for p in glob.glob(os.path.join(cell, '*.npz'))
                       if 'summary' not in os.path.basename(p)
                       and 'ate_w2' not in os.path.basename(p)
                       and 'malc_ci' not in os.path.basename(p))
    if not paths:
        return None
    os.environ['CASE_STUDY_DATA_ROOT'] = data_root
    os.environ['CASE_STUDY_N'] = str(n_ctx)
    out = {m: [] for m in METRICS}
    got = 0
    for p in paths:
        if max_real and got >= max_real:
            break
        base = os.path.splitext(os.path.basename(p))[0]
        digits = ''.join(c for c in base.split('_')[-1] if c.isdigit())
        if not digits:
            continue
        try:
            if tauc:
                # Cap queries BEFORE building densities -- the per-query
                # density reconstruction is the entire cost here, and coverage
                # is a mean over (shifts x realizations x queries), so the
                # cheapest axis to cut is queries. 20 still leaves 1200
                # samples per cen3 cell (SE ~0.63% on a 95% coverage).
                dens, grid, truth = load_predictions(p, DIR_METHOD[model],
                                                     max_q=MAX_Q or None)
                n_q = min(dens.shape[0], truth.size)
                cate_q, ate_m, _ = score_arrays(dens[:n_q], grid, truth[:n_q],
                                                levels=LEVELS,
                                                with_crps=False,
                                                with_wis=False)
            else:
                truth = scm_true_cate(case, int(digits))
                cate_q, ate_m, _ = score_realization(
                    p, truth, max_q=MAX_Q or None, levels=LEVELS,
                    with_crps=False, with_wis=False)
        except Exception:
            continue
        cq = [q['levels'][_A95] for q in cate_q]
        out['cate_cov95'].append(float(np.mean([c['covered'] for c in cq])))
        out['cate_len95'].append(float(np.mean([c['length'] for c in cq])))
        out['cate_is95'].append(float(np.mean([c['winkler'] for c in cq])))
        a = ate_m['levels'][_A95]
        out['ate_cov95'].append(float(a['covered']))
        out['ate_len95'].append(float(a['length']))
        out['ate_is95'].append(float(a['winkler']))
        out['ate_bias'].append(float(ate_m['ate_bias']))
        got += 1
    return out if got else None


def _row(shift, d, N, case, model, coll):
    n = max((len(coll[m]) for m in METRICS), default=0)
    vals = []
    for m in METRICS:
        vals.extend(_stats(coll[m]))
    return ("%s,%d,%d,%s,%s," + ",".join(["%.6f"] * (3 * len(METRICS))) + ",%d\n") % (
        (shift, d, N, case, model) + tuple(vals) + (n,))


def _score_cell(packed):
    """Score one (d, N, model, case) cell across the shift set. Returns
    (csv_row_or_None, [missing cell dirs])."""
    (d, N, model, case), (root, data_root, shifts, combining, label, max_real, min_n) = packed
    pooled = {m: [] for m in METRICS}
    missing, n_shifts = [], 0
    for s in shifts:
        cell = os.path.join(root, s, f'd{d}', f'ctx{N}', model, case)
        droot = os.path.join(data_root, s, f'd{d}')
        c = collect(cell, case, droot, N, max_real)
        if c is None:
            missing.append(cell)
            continue
        for m in METRICS:
            pooled[m].extend(c[m])
        n_shifts += 1
    if n_shifts == 0:
        return None, missing, None
    if combining and n_shifts != len(shifts):
        return None, missing, None    # same rule as dsweep_report
    tag = label if combining else shifts[0]
    n = max(len(pooled[m]) for m in METRICS)
    flag = '  <-- THIN' if (min_n and n < min_n) else ''
    print(f'  {tag} d{d} N{N} {model:14s} {case:34s} n={n}{flag}', flush=True)
    thin = (tag, d, N, model, case, n) if (min_n and n < min_n) else None
    return _row(tag, d, N, case, model, pooled), missing, thin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='results root (shift<S>/d<K>/ctx<N>/...)')
    ap.add_argument('--data-root', required=True,
                    help='d_variation root (shift<S>/d<K>/<case>/N<n>/*.npz)')
    ap.add_argument('--combine-shifts', nargs='*', default=None)
    ap.add_argument('--combine-label', default='combined')
    ap.add_argument('--cases', nargs='*', default=None)
    ap.add_argument('--models', nargs='*', default=MODELS)
    ap.add_argument('--max-real', type=int, default=None)
    ap.add_argument('--only-d', type=int, nargs='*', default=None,
                    help='restrict to these d values (for array jobs)')
    ap.add_argument('--only-n', type=int, nargs='*', default=None,
                    help='restrict to these context sizes. Cells outside the '
                         'list are never opened, so this cuts scoring time '
                         'proportionally (one N out of five = 5x faster).')
    ap.add_argument('--jobs', type=int, default=1,
                    help='parallel worker processes; the work is embarrassingly '
                         'parallel over (d, N, model, case) cells')
    ap.add_argument('--min-n', type=int, default=0,
                    help='Flag (and list at the end) any pooled cell with fewer '
                         'than this many realizations. 0 = off.')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    shifts = a.combine_shifts or sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(a.root, 'shift*')))
    ds = sorted({int(os.path.basename(p)[1:])
                 for s in shifts
                 for p in glob.glob(os.path.join(a.root, s, 'd*'))
                 if os.path.basename(p)[1:].isdigit()})
    if a.only_d:
        ds = [d for d in ds if d in set(a.only_d)]
    work = []
    for d in ds:
        ctxs = sorted({int(os.path.basename(p)[3:])
                       for s in shifts
                       for p in glob.glob(os.path.join(a.root, s, f'd{d}', 'ctx*'))
                       if os.path.basename(p)[3:].isdigit()})
        if a.only_n:
            ctxs = [n for n in ctxs if n in set(a.only_n)]
        for N in ctxs:
            for model in a.models:
                cases = a.cases or sorted({
                    os.path.basename(p)
                    for s in shifts
                    for p in glob.glob(os.path.join(
                        a.root, s, f'd{d}', f'ctx{N}', model, '*'))
                    if os.path.isdir(p)})
                for case in cases:
                    work.append((d, N, model, case))

    print(f'[density-report] {len(work)} cells to score, jobs={a.jobs}', flush=True)
    args_common = (a.root, a.data_root, shifts, bool(a.combine_shifts),
                   a.combine_label, a.max_real, a.min_n)
    missing, thin, nrows = [], [], 0
    if a.jobs > 1:
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            results = list(ex.map(_score_cell,
                                  [(w, args_common) for w in work], chunksize=1))
    else:
        results = [_score_cell((w, args_common)) for w in work]

    with open(a.out, 'w') as fh:
        fh.write(_HEADER)
        for row, miss, th in results:
            missing.extend(miss)
            if th:
                thin.append(th)
            if row:
                fh.write(row)
                nrows += 1
    print(f'\n[density-report] {nrows} rows -> {a.out}')
    if missing:
        print(f'[density-report] {len(missing)} cells had NO density dump '
              f'(point-eval only). First few:')
        for c in missing[:5]:
            print('   ', c)
    if thin:
        print(f'[density-report] WARNING: {len(thin)} cell(s) below --min-n='
              f'{a.min_n} (thin — noisy stats). First few:')
        for tag, d, N, model, case, n in thin[:10]:
            print(f'    {tag} d{d} N{N} {model} {case}  n={n}')


if __name__ == '__main__':
    main()
