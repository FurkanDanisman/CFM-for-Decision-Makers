"""Rebuild p(tau) per method from an eval_density_tauC.py prediction dump.

The tauC runner writes a DIFFERENT artifact from the CausalPFN DENSITY_DUMP:

    <OUT>/<DATASET>_r###.npz               per-method METRICS (nll/l2/pehe)
    <OUT>/predictions/<DATASET>_r###.npz   raw logits + bar geometry + truth

Only the second is usable here; the first holds scores, not distributions. This
module reads it and reconstructs exactly the objects the runner builds
internally (eval_density_tauC.py lines ~341-379), so the densities are the same
ones that produced its nll/l2 -- no new modelling, just re-derivation:

    joint_logits, edges2d, J                  -> Joint2D            'joint'
    uwyk_pred0/1, bar_edges/widths, base_s*   -> UWYK1D x2          'uwyk_native'
        ... rebinned onto edges2d                                   'uwyk_matched'
    dopfn_pred0/1, dopfn_borders*_raw         -> DoPFN1D x2         'dopfn_native'
    dopfn_joint_logits, dopfn_edges2d         -> Joint2D            'dopfn_joint'

ONE RUN EMITS SEVERAL METHODS. A MODEL_FAMILY=uwyk job yields uwyk_native,
uwyk_matched AND joint (= graph2d) from one process; dopfn yields dopfn_native
and dopfn_joint. So the submit-script's directory names do not map one-to-one
onto methods -- DIR_METHOD records which method each output directory is
reporting, and it is also why submitting graph2d and uwyk separately duplicates
work.

UNITS. tau_grid is the SCALED axis; tau_raw = tau_scaled * y_scale (the shift
cancels in a difference). true_cate in the dump is already raw.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from density_common import (                                          # noqa: E402
    Joint2D, UWYK1D, DoPFN1D, joint_tau_density, uwyk_tau_density,
    dopfn_tau_density,
)

# Which method each submit-script output directory is reporting.
DIR_METHOD = {
    'graph2d': 'joint',
    'uwyk': 'uwyk_native',
    'uwyk_v3a': 'uwyk_native',
    'uwyk_noanc': 'uwyk_native',
    'dopfn_native': 'dopfn_native',
    'dopfn_bb': 'dopfn_joint',
}
METHODS = ('joint', 'uwyk_native', 'uwyk_matched', 'dopfn_native', 'dopfn_joint')


def is_tauc_prediction(path):
    try:
        with np.load(path, allow_pickle=True) as z:
            return 'tau_grid' in z.files and (
                'joint_logits' in z.files or 'dopfn_joint_logits' in z.files)
    except Exception:
        return False


def _g(z, k, default=None):
    return z[k] if k in z.files else default


# Tail-quadrature resolution. The tauC eval defaults to 4096, which its own
# sbatch measures at 1.60 s/query (3 methods) vs 0.32 s at 1024 -- 5x -- for
# joint-path mass 0.99990 vs 0.99922. The scorer RENORMALISES each density
# before scoring, so that 8e-4 is absorbed and only the shape matters. At 4096
# the report needs ~24 h for the cen3 grid and dies on the 3 h wall clock.
N_Y0_DEFAULT = int(os.environ.get('TAUC_N_Y0', '1024'))


def load_predictions(path, method, n_y0=None):
    """-> (dens (N_q, T), tau_raw (T,), true_cate_raw (N_q,))  for one method."""
    n_y0 = N_Y0_DEFAULT if n_y0 is None else int(n_y0)
    with np.load(path, allow_pickle=True) as z:
        tau_scaled = np.asarray(z['tau_grid'], dtype=np.float64).reshape(-1)
        y_scale = float(np.asarray(z['y_scale']).reshape(-1)[0])
        true_cate = np.asarray(z['true_cate'], dtype=np.float64).reshape(-1)

        if method in ('joint', 'uwyk_native', 'uwyk_matched'):
            if 'joint_logits' not in z.files:
                raise KeyError(f'{path}: no joint_logits (MODEL_FAMILY=dopfn run?)')
            J = int(np.asarray(z['J']).reshape(-1)[0])
            edges2d = np.asarray(z['edges2d'], dtype=np.float64).reshape(-1)
            if method == 'joint':
                logits = np.asarray(z['joint_logits'], dtype=np.float64)
                dens = np.stack([
                    joint_tau_density(Joint2D.from_pred(logits[q], J, edges2d),
                                      tau_scaled, n_y0=n_y0)
                    for q in range(logits.shape[0])])
            else:
                p0 = np.asarray(z['uwyk_pred0'], dtype=np.float64)
                p1 = np.asarray(z['uwyk_pred1'], dtype=np.float64)
                be = np.asarray(z['bar_edges'], dtype=np.float64).reshape(-1)
                bw = np.asarray(z['bar_widths'], dtype=np.float64).reshape(-1)
                sL = float(np.asarray(z['base_sL']).reshape(-1)[0])
                sR = float(np.asarray(z['base_sR']).reshape(-1)[0])
                out = []
                for q in range(p0.shape[0]):
                    f0 = UWYK1D.from_pred(p0[q], be, bw, sL, sR)
                    f1 = UWYK1D.from_pred(p1[q], be, bw, sL, sR)
                    if method == 'uwyk_matched':
                        f0, f1 = f0.rebin(edges2d), f1.rebin(edges2d)
                    out.append(uwyk_tau_density(f0, f1, tau_scaled, n_y0=n_y0))
                dens = np.stack(out)

        elif method == 'dopfn_joint':
            logits = np.asarray(z['dopfn_joint_logits'], dtype=np.float64)
            J = int(np.asarray(z['dopfn_J']).reshape(-1)[0])
            edges = np.asarray(z['dopfn_edges2d'], dtype=np.float64).reshape(-1)
            dens = np.stack([
                joint_tau_density(Joint2D.from_pred(logits[q], J, edges),
                                  tau_scaled, n_y0=n_y0)
                for q in range(logits.shape[0])])

        elif method == 'dopfn_native':
            y_shift = float(np.asarray(_g(z, 'y_shift', 0.0)).reshape(-1)[0])
            arms = []
            for a in (0, 1):
                lg = np.asarray(z[f'dopfn_pred{a}'], dtype=np.float64)
                bo = np.asarray(z[f'dopfn_borders{a}_raw'], dtype=np.float64)
                sc = _g(z, f'dopfn_tail_scales{a}_raw')
                sc = None if sc is None else np.asarray(sc, dtype=np.float64)
                arms.append((lg, bo, sc))
            n_q = arms[0][0].shape[0]
            out = []
            for q in range(n_q):
                fs = []
                for lg, bo, sc in arms:
                    b = bo[q] if bo.ndim > 1 else bo
                    s = None if sc is None else (sc[q] if sc.ndim > 1 else sc)
                    fs.append(DoPFN1D.from_pred(lg[q], b, y_shift=y_shift,
                                                y_scale=y_scale, tail_scales=s))
                out.append(dopfn_tau_density(fs[0], fs[1], tau_scaled))
            dens = np.stack(out)
        else:
            raise ValueError(f'unknown method {method!r}; pick from {METHODS}')

    # scaled -> raw: tau_raw = tau_scaled * y_scale, density divides by y_scale
    tau_raw = tau_scaled * y_scale
    dens = dens / y_scale
    return dens, tau_raw, true_cate
