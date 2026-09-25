"""Tier-A density eval: the per-arm marginals p(Y | do(T=t), x) on IHDP / ACIC.

Sibling of eval_density_tauC.py, one tier upstream. That script scores p(tau |
x); this one scores the two arm densities the models actually emit, BEFORE any
tau is formed:

    1D heads (UWYK, DoPFN, CausalPFN-1D)
        two forward passes with T := 0 and T := 1 -> pred0, pred1.
        eval_density_tauC then CONVOLVES them into p(tau). We read pred0/pred1.
    2D heads (graph2d, DoPFN joint, CausalPFN-2D)
        one forward pass -> a joint head over (y0, y1).
        eval_density_tauC integrates p(y0, y0 + tau) over y0, with exact
        interior terms and tail quadrature, to get p(tau).
        We integrate the other arm out instead (density_common.joint_marginals,
        closed form over the same 9-region mixture).

So nothing here is reconstructed from a CATE density: tau is downstream of both
objects. The marginals also carry no arm-independence assumption -- that
assumption enters only in the 1D convolution that produces p(tau).

This runs entirely OFFLINE off the prediction dumps eval_density_tauC already
writes under <OUT>/predictions/ (SAVE_PREDICTIONS=1, the default). No model is
loaded and no GPU is touched.

Metrics per query per arm, on Y_MARG (scaled units, the axis the heads live on):
    nll     -log p_est(mu_t)   at the true conditional mean, evaluated on the
                               density object itself, never read off the grid.
                               (The tau tier takes NLL at a sampled tau*; the
                               dumps carry no sampled y0*/y1*, and this is the
                               convention realcause_eval/eval_density_metrics.py
                               already uses for CATE.)
    l2      ||p_true - p_est||_2
    kl_fwd  KL(truth || est)
    kl_rev  KL(est || truth)
    mass    int p_est dy        diagnostic: how much sits on the grid

Truth is the analytic Gaussian N(mu_t, sigma) the IHDP/ACIC DGPs ship, read
straight from the dump (mu0_scaled, mu1_scaled, sigma_scaled).

Aggregation matches the tau tier: mean over queries here, mean +- SE over
realizations in the summarizer.

Usage:
    python benchmarks/eval_graph2d/eval_density_marginals.py \
        --predictions results_density_tauC/5312882/IHDP/predictions \
        --out results_density_marginals/uwyk_v3a/IHDP
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from density_common import (                                        # noqa: E402
    Joint2D, UWYK1D, DoPFN1D, CausalPFN1D, joint_marginals,
    l2_distance, kl, mass, Y_MARG,
)

ARMS = (0, 1)


def _gauss(x, mu, sigma):
    z = (np.asarray(x, dtype=np.float64) - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


# ---------------------------------------------------------------------------
# Reconstruction. build_objects is shared with eval_density_ate.py so both
# tiers score the *same* density objects the tau tier scored.
# ---------------------------------------------------------------------------
def build_objects(z):
    """{method: (family, kind, per_query)} from a prediction dump.

    family is 'uwyk' | 'dopfn' | 'causalpfn' -- it picks the tau-density
    routine downstream. kind is '1d' (per_query holds (arm0, arm1) pairs) or
    'joint' (per_query holds Joint2D).
    """
    keys = set(z.files)
    out: dict[str, tuple] = {}
    n_q = int(z['mu0_scaled'].shape[0])
    ys, yc = float(z['y_shift']), float(z['y_scale'])

    # -- UWYK 1D + graph2d joint ------------------------------------------
    if 'uwyk_pred0' in keys:
        be, bw = z['bar_edges'], z['bar_widths']
        sL, sR = float(z['base_sL']), float(z['base_sR'])
        p0, p1 = z['uwyk_pred0'], z['uwyk_pred1']
        out['uwyk_native'] = ('uwyk', '1d', [
            (UWYK1D.from_pred(p0[q], be, bw, sL, sR),
             UWYK1D.from_pred(p1[q], be, bw, sL, sR)) for q in range(n_q)])
    if 'joint_logits' in keys:
        J, e2 = int(z['J']), z['edges2d']
        lg = z['joint_logits']
        out['joint'] = ('uwyk', 'joint',
                        [Joint2D.from_pred(lg[q], J, e2) for q in range(n_q)])

    # -- DoPFN -------------------------------------------------------------
    if 'dopfn_pred0' in keys:                      # the library ("native") head
        out['dopfn_native'] = ('dopfn', '1d', [
            (DoPFN1D.from_pred(z['dopfn_pred0'][q], z['dopfn_borders0_raw'],
                               y_shift=ys, y_scale=yc,
                               tail_scales=z['dopfn_tail_scales0_raw']),
             DoPFN1D.from_pred(z['dopfn_pred1'][q], z['dopfn_borders1_raw'],
                               y_shift=ys, y_scale=yc,
                               tail_scales=z['dopfn_tail_scales1_raw']))
            for q in range(n_q)])
    for m, kind in (zip(z['dopfn_methods'], z['dopfn_kinds'])
                    if 'dopfn_methods' in keys else ()):
        m, kind = str(m), str(kind)
        if m == 'dopfn_native':
            continue                                # handled above
        # NB: {m}_y_shift/_y_scale are the model's OWN native scaling, kept for
        # provenance and already folded into {m}_borders_raw / {m}_edges2d.
        # density_dopfn passes the HARNESS affine to from_pred; mirror that.
        # dopfn_kinds spells these 'native' / 'repro_1d' / 'repro_joint'.
        if 'joint' in kind:
            J, e2 = int(z[f'{m}_J']), z[f'{m}_edges2d']
            lg = z[f'{m}_logits']
            out[m] = ('dopfn', 'joint',
                      [Joint2D.from_pred(lg[q], J, e2) for q in range(n_q)])
        else:
            b = z[f'{m}_borders_raw']
            a0, a1 = z[f'{m}_pred0'], z[f'{m}_pred1']
            out[m] = ('dopfn', '1d', [
                (DoPFN1D.from_pred(a0[q], b, y_shift=ys, y_scale=yc),
                 DoPFN1D.from_pred(a1[q], b, y_shift=ys, y_scale=yc))
                for q in range(n_q)])

    # -- CausalPFN ---------------------------------------------------------
    for m, kind in (zip(z['causalpfn_methods'], z['causalpfn_kinds'])
                    if 'causalpfn_methods' in keys else ()):
        m, kind = str(m), str(kind)
        if kind == 'joint':
            J, e2 = int(z[f'{m}_J']), z[f'{m}_edges2d']
            lg = z[f'{m}_logits']
            out[m] = ('causalpfn', 'joint',
                      [Joint2D.from_pred(lg[q], J, e2) for q in range(n_q)])
        else:
            e1 = z[f'{m}_edges1d']
            a0, a1 = z[f'{m}_pred0'], z[f'{m}_pred1']
            out[m] = ('causalpfn', '1d', [
                (CausalPFN1D.from_pred(a0[q], e1),
                 CausalPFN1D.from_pred(a1[q], e1)) for q in range(n_q)])
    return out


def build_methods(z):
    """{method: [evaluate_q, ...]}, evaluate(y) -> (p_y0, p_y1) for one query."""
    out = {}
    for m, (_family, kind, per_query) in build_objects(z).items():
        if kind == '1d':
            out[m] = [(lambda y, o0=o0, o1=o1: (o0.density(y), o1.density(y)))
                      for (o0, o1) in per_query]
        else:
            out[m] = [(lambda y, jt=jt: joint_marginals(jt, y))
                      for jt in per_query]
    return out


# ---------------------------------------------------------------------------
def run_realization(path, save_densities=False):
    z = np.load(path, allow_pickle=True)
    mu = (np.asarray(z['mu0_scaled'], float), np.asarray(z['mu1_scaled'], float))
    sigma = float(z['sigma_scaled'])
    n_q = mu[0].shape[0]
    methods = build_methods(z)
    n = Y_MARG.size
    truth = [np.stack([_gauss(Y_MARG, mu[t][q], sigma) for q in range(n_q)])
             for t in ARMS]

    out = {'dataset': str(z['dataset']), 'realization': int(z['realization']),
           'n_queries': n_q, 'anc_tag': str(z['anc_tag']) if 'anc_tag' in z.files else '',
           'sigma_scaled': sigma, 'y_scale': float(z['y_scale']),
           'y_shift': float(z['y_shift']), 'y_grid': Y_MARG,
           'methods': np.asarray(sorted(methods))}

    for name, per_query in methods.items():
        acc = {t: {k: [] for k in ('nll', 'l2', 'kl_fwd', 'kl_rev', 'mass')}
               for t in ARMS}
        dens = {t: np.empty((n_q, n)) for t in ARMS} if save_densities else None
        for q, evaluate in enumerate(per_query):
            # One call: the grid, then mu0 and mu1 appended so NLL is taken on
            # the density object rather than interpolated off the grid.
            ye = np.concatenate([Y_MARG, [mu[0][q], mu[1][q]]])
            pe = evaluate(ye)
            for t in ARMS:
                p = pe[t][:n]
                at_mu = float(pe[t][n + t])
                acc[t]['nll'].append(float('inf') if at_mu <= 0
                                     else float(-np.log(at_mu)))
                acc[t]['l2'].append(l2_distance(truth[t][q], p, Y_MARG))
                acc[t]['kl_fwd'].append(kl(truth[t][q], p, Y_MARG))
                acc[t]['kl_rev'].append(kl(p, truth[t][q], Y_MARG))
                acc[t]['mass'].append(mass(p, Y_MARG))
                if save_densities:
                    dens[t][q] = p
        for t in ARMS:
            out[f'frac_zero_density_y{t}_{name}'] = float(
                np.mean(np.isposinf(acc[t]['nll'])))
            for k, v in acc[t].items():
                out[f'{k}_y{t}_{name}'] = float(np.mean(v))
            if save_densities:
                out[f'p_y{t}_scaled_{name}'] = dens[t]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', required=True,
                    help='a <OUT>/predictions directory written by eval_density_tauC.py')
    ap.add_argument('--out', required=True)
    ap.add_argument('--save-densities', action='store_true',
                    help='also store p_y{0,1}_scaled per query (feeds the ATE '
                         'barycenter path); large, so off by default')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.predictions, '*_r[0-9]*.npz')))
    if a.limit:
        files = files[:a.limit]
    if not files:
        raise SystemExit(f'no prediction dumps under {a.predictions}')
    os.makedirs(a.out, exist_ok=True)
    for i, f in enumerate(files):
        res = run_realization(f, save_densities=a.save_densities)
        dest = os.path.join(a.out, os.path.basename(f))
        np.savez_compressed(dest, **res)
        print(f'[{i + 1}/{len(files)}] r{res["realization"]:03d} '
              f'{len(res["methods"])} methods -> {dest}', flush=True)


if __name__ == '__main__':
    main()
