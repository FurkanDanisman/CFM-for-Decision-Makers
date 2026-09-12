"""Score CausalPFN DENSITY_DUMP output with coverage / length / Winkler / CRPS.

Reads the npz a cpfn1d or cpfn2d eval wrote with DENSITY_DUMP=1, rebuilds
p(tau) per query, and scores it against the case-study true CATE (which IS the
per-unit tau — see density_truth.py).

    python run_density_cpfn.py --dump-root <cell dir> --dataset <case> \
        --data-root case_study/data_shift+2 --n 200 --out <out.json>

Units. A pooled dump has one shared scaled axis: p(tau) is built on the scaled
tau grid and `y_scale` converts the metrics to raw. A per-arm dump has no
single scaled axis, so p(tau) is built directly in RAW units and y_scale is 1.
Both paths score against the same raw truth; the tau GRID differs, and that is
recorded in the output as `units`.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from density_cpfn import load_dump, iter_queries, tau_support, tau_support_raw  # noqa: E402
from density_truth import scm_true_cate                                        # noqa: E402
from interval_metrics import DEFAULT_LEVELS, query_metrics, summarize, format_table  # noqa: E402

TAU_MIN, TAU_MAX, TAU_STEP = -3.0, 3.0, 0.0005


def tau_grid(lo, hi, step=TAU_STEP, max_points=400_001):
    """Grid covering the model's ENTIRE tau support.

    Deliberately NOT the fixed [-3, 3] TAU_CENTERS: CausalPFN's support is
    +/-(edges span), routinely wider than 3, and truncating it biases length,
    Winkler and CRPS DOWNWARD -- i.e. flatters diffuse models. We widen instead
    and report the realised span.
    """
    lo = min(lo, TAU_MIN)
    hi = max(hi, TAU_MAX)
    n = int(round((hi - lo) / step)) + 1
    if n > max_points:                       # keep memory bounded
        n = max_points
    return np.linspace(lo, hi, n)


def score_cell(dump_path, dataset, levels=DEFAULT_LEVELS, method='equal-tailed'):
    dump = load_dump(dump_path)
    truth_raw = scm_true_cate(dataset, _realization_of(dump_path))
    per_arm = bool(dump.get('per_arm'))

    if per_arm:
        lo, hi = tau_support_raw(dump['e0_raw'], dump['e1_raw'])
        y_scale, units = 1.0, 'raw'
    else:
        lo, hi = tau_support(dump['edges'])
        y_scale, units = dump['y_scale'], 'scaled'
    tau = tau_grid(lo, hi)

    per_q, outside = [], 0
    for q, dens, extras in iter_queries(dump):
        if q >= truth_raw.size:
            break
        t_raw = float(truth_raw[q])
        t_axis = t_raw if per_arm else t_raw / y_scale
        if not (lo <= t_axis <= hi):
            outside += 1
        per_q.append(query_metrics(dens(tau), tau, t_axis,
                                   levels=levels, y_scale=y_scale, method=method))
    if not per_q:
        raise RuntimeError(f'{dump_path}: no queries scored')

    s = summarize(per_q, levels=levels)
    s.update(dataset=dataset, dump=os.path.basename(dump_path), units=units,
             per_arm=per_arm, y_scale=y_scale, method=method,
             tau_support=[lo, hi], tau_grid_n=int(tau.size),
             outside_support_frac=outside / len(per_q))
    return s


def _realization_of(path):
    base = os.path.splitext(os.path.basename(path))[0]
    for tok in reversed(base.replace('-', '_').split('_')):
        digits = ''.join(c for c in tok if c.isdigit())
        if digits:
            return int(digits)
    raise ValueError(f'cannot parse a realization index from {path!r}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump-root', required=True,
                    help='directory of DENSITY_DUMP npz files for one cell')
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--method', default='equal-tailed',
                    choices=['equal-tailed', 'hpd'])
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    os.environ['CASE_STUDY_DATA_ROOT'] = a.data_root
    os.environ['CASE_STUDY_N'] = str(a.n)

    paths = sorted(glob.glob(os.path.join(a.dump_root, '*.npz')))
    if not paths:
        raise FileNotFoundError(f'no npz under {a.dump_root}')

    rows = []
    for p in paths:
        try:
            rows.append(score_cell(p, a.dataset, method=a.method))
        except Exception as e:                    # one bad realization != lost run
            print(f'[warn] {os.path.basename(p)}: {type(e).__name__}: {e}',
                  flush=True)
    if not rows:
        raise RuntimeError('every realization failed')

    pooled = {
        'dataset': a.dataset, 'n_context': a.n, 'n_realizations': len(rows),
        'units': rows[0]['units'], 'per_arm': rows[0]['per_arm'],
        'method': a.method,
        'crps_mean': float(np.mean([r['crps_mean'] for r in rows])),
        'outside_support_frac': float(np.mean([r['outside_support_frac']
                                               for r in rows])),
        'mass_mean': float(np.mean([r['mass_mean'] for r in rows])),
        'levels': {},
    }
    for lv in DEFAULT_LEVELS:
        pooled['levels'][str(lv)] = {
            k: float(np.mean([r['levels'][lv][k] for r in rows]))
            for k in ('nominal_pct', 'coverage_pct', 'length_mean',
                      'winkler_mean', 'censored_frac')}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump({'pooled': pooled, 'per_realization': rows}, f, indent=2)
    print(format_table(summarize([], levels=DEFAULT_LEVELS)
                       if False else rows[0], title=f'{a.dataset} N={a.n} '
                       f'(realization 0 of {len(rows)})'))
    print(f'\npooled CRPS={pooled["crps_mean"]:.5f}  '
          f'outside_support={100*pooled["outside_support_frac"]:.1f}%  -> {a.out}')


if __name__ == '__main__':
    main()
