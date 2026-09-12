"""CATE + ATE density metrics for the case studies: coverage, length, WIS, CRPS.

Per realization:
  CATE  p(tau | x_q) for each query q, scored against that query's true CATE.
  ATE   1D 2-Wasserstein barycenter of the per-query p(tau | x_q), scored
        against the realization's true ATE = mean_q true_cate[q].

The barycenter is `MALC/Optimal_Transport/ot_barycenter.wasserstein_barycenter_1d`
and the tau projections are imported from
`realcause_eval/compute_ate_density_w2_cell.py`, so this is the SAME optimal
transport and the SAME joint/marginal conventions the Figure-4 ATE pipeline
uses -- nothing is reimplemented:

    2D methods   antidiagonal sums of p_joint_scaled   (no independence assumed)
    1D methods   FFT convolution of p_y0 x p_y1        (Y|do(0) _|_ Y|do(1))

SCORING CONVENTION. Both CATE and ATE use interval_metrics, whose CRPS inserts
the truth as an exact quadrature node. `ate_density_w2_summary.py` instead sums
(F - 1[tau>=y])^2 over a fixed grid; that straddles the jump at y and errs by
about (dtau/2)*|2F(y)-1| -- zero when the truth sits at the predictive median,
saturating near dtau/2 in the tails (measured: 1.5e-3 at dtau=2.5e-3, ~2% of a
typical CRPS). The error is always positive and grows with how far the truth
lies in the model's tail, so it inflates CRPS for badly-centred models
specifically. Numbers here are therefore NOT directly comparable to Figure-4
ATE numbers computed with the old convention.

Usage:
    python run_density_scm.py --dump-root <cell dir> --dataset Observed_Confounder \\
        --data-root case_study/d_variation/shift+2/d10 --n 200 --out metrics.json
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _HERE)

from interval_metrics import (DEFAULT_LEVELS, query_metrics, summarize,   # noqa: E402
                              format_table)
from density_truth import scm_true_cate                                    # noqa: E402

_TRAPZ = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz


def _load_rc_module():
    """Import compute_ate_density_w2_cell for its tau projections + barycenter.
    Read-only: nothing outside case_study/ is modified."""
    p = os.path.join(_REPO, 'realcause_eval', 'compute_ate_density_w2_cell.py')
    if not os.path.isfile(p):
        raise FileNotFoundError(p)
    spec = importlib.util.spec_from_file_location('_rc_ate_w2', p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def p_tau_atoms(z):
    """Per-query tau mass atoms + their SCALED tau positions, from a density npz.

    Returns (p_tau (N_q, T), tau_scaled (T,), y_scale). 2D dumps use the joint
    (no independence); 1D dumps convolve the marginals.
    """
    rc = _load_rc_module()
    edges = np.asarray(z['edges'], dtype=np.float64).reshape(-1)
    bw = float((edges[-1] - edges[0]) / (edges.size - 1))
    if 'p_joint_scaled' in z:
        p = rc._p_tau_from_joint(np.asarray(z['p_joint_scaled'], dtype=np.float64))
        J = int(np.asarray(z['p_joint_scaled']).shape[1])
    else:
        p0 = np.asarray(z['p_y0_scaled'], dtype=np.float64)
        p1 = np.asarray(z['p_y1_scaled'], dtype=np.float64)
        p = rc._p_tau_from_marginals(p0, p1)
        J = p0.shape[1]
    tau_scaled = (np.arange(p.shape[1]) - (J - 1)) * bw
    y_scale = float(np.asarray(z['y_scale']).reshape(-1)[0]) if 'y_scale' in z else 1.0
    return p, tau_scaled, y_scale


def densify(p_atoms, tau_atoms, tau_grid):
    """Atoms -> density on `tau_grid` by linear (tent) splatting.

    Mass-preserving and the continuous limit of the histogram convolution, i.e.
    the same object `_interior_tau` builds in density_common.
    """
    T = tau_grid.size
    dt = float(tau_grid[1] - tau_grid[0])
    x = (tau_atoms - tau_grid[0]) / dt
    i0 = np.floor(x).astype(int)
    w1 = x - i0
    out = np.zeros((p_atoms.shape[0], T), dtype=np.float64)
    for idx, w in ((i0, 1.0 - w1), (i0 + 1, w1)):
        ok = (idx >= 0) & (idx < T)
        if ok.any():
            np.add.at(out, (slice(None), idx[ok]), p_atoms[:, ok] * w[ok])
    return out / dt


def score_realization(path, true_cate, levels=DEFAULT_LEVELS, n_grid=8001,
                      method='equal-tailed', n_tau_bary=4001):
    rc = _load_rc_module()
    barycenter = rc._import_barycenter(_REPO)

    with np.load(path, allow_pickle=True) as z:
        p_atoms, tau_scaled, y_scale = p_tau_atoms(z)
    tau_raw_atoms = tau_scaled * y_scale
    n_q = min(p_atoms.shape[0], true_cate.size)
    p_atoms, true_cate = p_atoms[:n_q], true_cate[:n_q]

    lo, hi = float(tau_raw_atoms[0]), float(tau_raw_atoms[-1])
    pad = 0.02 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, n_grid)
    dens = densify(p_atoms, tau_raw_atoms, grid)
    dens /= (dens.sum(axis=1, keepdims=True) * (grid[1] - grid[0])).clip(min=1e-300)

    cate_q = [query_metrics(dens[q], grid, float(true_cate[q]),
                            levels=levels, y_scale=1.0, method=method)
              for q in range(n_q)]

    p_ate = np.asarray(barycenter(dens, grid, n_tau=n_tau_bary), dtype=np.float64)
    s = float(_TRAPZ(p_ate, grid))
    if s > 0:
        p_ate = p_ate / s
    true_ate = float(np.mean(true_cate))
    ate_m = query_metrics(p_ate, grid, true_ate, levels=levels,
                          y_scale=1.0, method=method)
    ate_m['ate_mean_pred'] = float(_TRAPZ(p_ate * grid, grid))
    ate_m['true_ate'] = true_ate
    ate_m['ate_bias'] = ate_m['ate_mean_pred'] - true_ate
    return cate_q, ate_m, {'n_q': n_q, 'tau_support': [lo, hi]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump-root', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--method', default='equal-tailed',
                    choices=['equal-tailed', 'hpd'])
    ap.add_argument('--pattern', default='*.npz')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    os.environ['CASE_STUDY_DATA_ROOT'] = a.data_root
    os.environ['CASE_STUDY_N'] = str(a.n)

    paths = sorted(p for p in glob.glob(os.path.join(a.dump_root, a.pattern))
                   if 'ate_w2' not in os.path.basename(p)
                   and 'malc_ci' not in os.path.basename(p))
    if not paths:
        raise FileNotFoundError(f'no density npz under {a.dump_root}')

    cate_all, ate_all, skipped = [], [], []
    for p in paths:
        base = os.path.splitext(os.path.basename(p))[0]
        try:
            r = int(''.join(c for c in base.split('_')[-1] if c.isdigit()))
            truth = scm_true_cate(a.dataset, r)
            cq, am, _ = score_realization(p, truth, method=a.method)
        except Exception as e:
            skipped.append((os.path.basename(p), f'{type(e).__name__}: {e}'))
            continue
        cate_all.extend(cq)
        ate_all.append(am)

    if not cate_all:
        raise RuntimeError(f'every realization failed; first: {skipped[:2]}')

    out = {'dataset': a.dataset, 'n_context': a.n, 'method': a.method,
           'crps_convention': 'node-insertion (truth as exact quadrature node)',
           'n_realizations': len(ate_all), 'n_skipped': len(skipped),
           'skipped': skipped[:10],
           'cate': summarize(cate_all, levels=DEFAULT_LEVELS),
           'ate': summarize(ate_all, levels=DEFAULT_LEVELS)}
    out['ate']['bias_mean'] = float(np.mean([m['ate_bias'] for m in ate_all]))
    out['ate']['abs_err_mean'] = float(np.mean([abs(m['ate_bias']) for m in ate_all]))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=2)
    print(format_table(out['cate'], title=f'CATE  {a.dataset} N={a.n}'))
    print()
    print(format_table(out['ate'], title=f'ATE (W2 barycenter)  {a.dataset} N={a.n}'))
    print(f"  ATE bias={out['ate']['bias_mean']:+.4f}  "
          f"|bias|={out['ate']['abs_err_mean']:.4f}  "
          f"realizations={len(ate_all)}  skipped={len(skipped)}")
    print(f'-> {a.out}')


if __name__ == '__main__':
    main()
