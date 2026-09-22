"""ATE-density eval: p_ATE(tau), the W2 barycenter of per-query p(tau | x).

Third tier alongside eval_density_marginals.py (arms) and eval_density_tauC.py
(CATE). No shortcuts: the per-query tau densities are rebuilt with the SAME
exact routines and the SAME TAU_CENTERS grid the tau tier used, from the same
prediction dumps, so f_ATE sits on identical densities to f_tau. In particular
the 2D heads go through joint_tau_density's full-mixture tail quadrature at the
dump's own n_y0 -- not an anti-diagonal sum of the interior, and not a grid
convolution of the marginals.

  Estimated p_ATE  = W2 barycenter over queries of the model's p(tau | x_q).

Two truth conventions exist in this repo and they are NOT the same object, so
both are scored and both are stored:

  truth_bary  W2 barycenter of the true per-query p(tau | x_q).
              benchmarks/l2_ihdp/true_ihdp.py::true_ate_barycenter.
              Apples-to-apples: same operator as the estimate.
  truth_mix   arithmetic mean of the true per-query densities.
              realcause_eval/eval_ate_density_metrics.py's docstring.
              For N(mu_q, 2 sigma^2) this is a MIXTURE of width
              ~sqrt(2 sigma^2 + var(mu_q)); the barycenter is N(mean mu_q,
              2 sigma^2) of width sqrt(2) sigma. They coincide only when the
              CATE is homogeneous, so the two NLLs can differ a lot.

Metrics per realization, on the scaled tau axis (the axis f_tau lives on;
multiply a density by 1/y_scale for raw units, and NLL shifts by log y_scale):

  nll      -log p_est(ATE_true)   ATE_true = mean_q (mu1 - mu0), scaled
  l2       ||p_true - p_est||_2
  kl_fwd   KL(truth || est)
  kl_rev   KL(est || truth)
  mass     int p_est dtau
  ate_err  |E[p_est] - ATE_true|

Usage:
    python benchmarks/eval_graph2d/eval_density_ate.py \\
        --predictions results_density_tauC/5312882/IHDP/predictions \\
        --out results_density_ate/uwyk_v3a/IHDP --workers 8
"""
from __future__ import annotations

import argparse
import functools
import glob
import math
import multiprocessing as mp
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO, 'MALC', 'Optimal_Transport'))

from density_common import (                                        # noqa: E402
    TAU_CENTERS, uwyk_tau_density, dopfn_tau_density, causalpfn_tau_density,
    joint_tau_density, truth_tau_density, l2_distance, kl, mass,
)
from eval_density_marginals import build_objects                    # noqa: E402
from ot_barycenter import wasserstein_barycenter_1d                 # noqa: E402

TAU_1D = {'uwyk': uwyk_tau_density, 'dopfn': dopfn_tau_density,
          'causalpfn': causalpfn_tau_density}
METRICS = ('nll', 'l2', 'kl_fwd', 'kl_rev', 'mass', 'ate_err')


def tau_densities(family, kind, per_query, n_y0):
    """(n_q, len(TAU_CENTERS)) -- exactly the tau tier's construction."""
    if kind == 'joint':
        return np.stack([joint_tau_density(jt, TAU_CENTERS, n_y0=n_y0)
                         for jt in per_query])
    fn = TAU_1D[family]
    if family == 'uwyk':                      # only this one takes n_y0
        return np.stack([fn(f0, f1, TAU_CENTERS, n_y0=n_y0)
                         for f0, f1 in per_query])
    return np.stack([fn(f0, f1, TAU_CENTERS) for f0, f1 in per_query])


def _normalise(p):
    s = float(np.trapezoid(p, TAU_CENTERS))
    return p / s if s > 0 else p


def score(p_est, p_true, ate_true):
    at = float(np.interp(ate_true, TAU_CENTERS, p_est))
    return dict(
        nll=float('inf') if at <= 0 else float(-math.log(at)),
        l2=l2_distance(p_true, p_est, TAU_CENTERS),
        kl_fwd=kl(p_true, p_est, TAU_CENTERS),
        kl_rev=kl(p_est, p_true, TAU_CENTERS),
        mass=mass(p_est, TAU_CENTERS),
        ate_err=abs(float(np.trapezoid(TAU_CENTERS * p_est, TAU_CENTERS))
                    - ate_true),
    )


def run_realization(path, only=None, save_densities=False):
    z = np.load(path, allow_pickle=True)
    mu0, mu1 = np.asarray(z['mu0_scaled'], float), np.asarray(z['mu1_scaled'], float)
    sigma = float(z['sigma_scaled'])
    n_y0 = int(z['n_y0']) if 'n_y0' in z.files else 4096
    ate_true = float((mu1 - mu0).mean())

    p_true_q = np.stack([truth_tau_density(mu0[q], mu1[q], sigma, TAU_CENTERS)
                         for q in range(mu0.size)])
    truths = {'bary': _normalise(wasserstein_barycenter_1d(p_true_q, TAU_CENTERS)),
              'mix': _normalise(p_true_q.mean(axis=0))}

    out = {'dataset': str(z['dataset']), 'realization': int(z['realization']),
           'n_queries': int(mu0.size), 'ate_true_scaled': ate_true,
           'y_scale': float(z['y_scale']), 'n_y0': n_y0,
           'anc_tag': str(z['anc_tag']) if 'anc_tag' in z.files else '',
           'tau_grid': TAU_CENTERS}
    for t, p in truths.items():
        out[f'p_ate_truth_{t}'] = p

    objs = build_objects(z)
    for m, (family, kind, per_query) in objs.items():
        if only and m not in only:
            continue
        p_ate = _normalise(wasserstein_barycenter_1d(
            tau_densities(family, kind, per_query, n_y0), TAU_CENTERS))
        for t, p_true in truths.items():
            for k, v in score(p_ate, p_true, ate_true).items():
                out[f'{k}_{t}_{m}'] = v
        if save_densities:
            out[f'p_ate_{m}'] = p_ate
    out['methods'] = np.asarray(sorted(m for m in objs if not only or m in only))
    return out


def _one(path, out_dir, only, save_densities):
    res = run_realization(path, only=only, save_densities=save_densities)
    dest = os.path.join(out_dir, os.path.basename(path))
    np.savez_compressed(dest, **res)
    return f'r{res["realization"]:03d} ({res["n_queries"]} q, ' \
           f'{len(res["methods"])} methods)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--methods', default='',
                    help='comma-separated subset; default every method in the dump')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--save-densities', action='store_true')
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.predictions, '*_r[0-9]*.npz')))
    if a.limit:
        files = files[:a.limit]
    if not files:
        raise SystemExit(f'no prediction dumps under {a.predictions}')
    os.makedirs(a.out, exist_ok=True)
    only = set(filter(None, a.methods.split(','))) or None

    work = functools.partial(_one, out_dir=a.out, only=only,
                             save_densities=a.save_densities)
    print(f'[ate] {len(files)} realizations, {a.workers} worker(s) -> {a.out}',
          flush=True)
    if a.workers > 1:
        with mp.Pool(a.workers) as pool:
            for i, tag in enumerate(pool.imap_unordered(work, files), 1):
                print(f'[{i}/{len(files)}] {tag}', flush=True)
    else:
        for i, f in enumerate(files, 1):
            print(f'[{i}/{len(files)}] {work(f)}', flush=True)


if __name__ == '__main__':
    main()
