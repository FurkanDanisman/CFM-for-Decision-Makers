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

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from interval_metrics import DEFAULT_LEVELS                       # noqa: E402
from run_density_scm import score_realization                     # noqa: E402
from density_truth import scm_true_cate                           # noqa: E402

# Same model list / dir names as dsweep_report.py.
MODELS = ["dopfn_native", "dopfn_bb", "cpfn2d_pooled", "cpfn1d_perarm",
          "graph2d", "uwyk", "uwyk_v3a", "uwyk_noanc"]
METRICS = ["cate_cov95", "cate_len95", "cate_wis", "cate_crps",
           "ate_cov95", "ate_len95", "ate_wis", "ate_crps", "ate_bias"]
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


def collect(cell, case, data_root, n_ctx, max_real=None):
    """Per-realization density metrics for one (model, case) dir."""
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
            truth = scm_true_cate(case, int(digits))
            cate_q, ate_m, _ = score_realization(p, truth)
        except Exception:
            continue
        cq = [q['levels'][_A95] for q in cate_q]
        out['cate_cov95'].append(float(np.mean([c['covered'] for c in cq])))
        out['cate_len95'].append(float(np.mean([c['length'] for c in cq])))
        out['cate_wis'].append(float(np.mean([q['wis'] for q in cate_q])))
        out['cate_crps'].append(float(np.mean([q['crps'] for q in cate_q])))
        a = ate_m['levels'][_A95]
        out['ate_cov95'].append(float(a['covered']))
        out['ate_len95'].append(float(a['length']))
        out['ate_wis'].append(float(ate_m['wis']))
        out['ate_crps'].append(float(ate_m['crps']))
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
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    shifts = a.combine_shifts or sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(a.root, 'shift*')))
    ds = sorted({int(os.path.basename(p)[1:])
                 for s in shifts
                 for p in glob.glob(os.path.join(a.root, s, 'd*'))
                 if os.path.basename(p)[1:].isdigit()})
    missing, nrows = [], 0
    with open(a.out, 'w') as fh:
        fh.write(_HEADER)
        for d in ds:
            ctxs = sorted({int(os.path.basename(p)[3:])
                           for s in shifts
                           for p in glob.glob(os.path.join(a.root, s, f'd{d}', 'ctx*'))
                           if os.path.basename(p)[3:].isdigit()})
            for N in ctxs:
                for model in a.models:
                    cases = a.cases or sorted({
                        os.path.basename(p)
                        for s in shifts
                        for p in glob.glob(os.path.join(
                            a.root, s, f'd{d}', f'ctx{N}', model, '*'))
                        if os.path.isdir(p)})
                    for case in cases:
                        pooled = {m: [] for m in METRICS}
                        n_shifts = 0
                        for s in shifts:
                            cell = os.path.join(a.root, s, f'd{d}', f'ctx{N}',
                                                model, case)
                            droot = os.path.join(a.data_root, s, f'd{d}')
                            c = collect(cell, case, droot, N, a.max_real)
                            if c is None:
                                missing.append(cell)
                                continue
                            for m in METRICS:
                                pooled[m].extend(c[m])
                            n_shifts += 1
                        if n_shifts == 0:
                            continue
                        if a.combine_shifts and n_shifts != len(shifts):
                            continue          # same rule as dsweep_report
                        label = a.combine_label if a.combine_shifts else shifts[0]
                        fh.write(_row(label, d, N, case, model, pooled))
                        nrows += 1
                        print(f'  {label} d{d} N{N} {model:14s} {case:34s} '
                              f'n={max(len(pooled[m]) for m in METRICS)}', flush=True)
    print(f'\n[density-report] {nrows} rows -> {a.out}')
    if missing:
        print(f'[density-report] {len(missing)} cells had NO density dump '
              f'(point-eval only). First few:')
        for c in missing[:5]:
            print('   ', c)


if __name__ == '__main__':
    main()
