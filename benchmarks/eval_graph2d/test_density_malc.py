"""Validation gates for the MALC arm of density_common.py. No model, no GPU.

    python benchmarks/eval_graph2d/test_density_malc.py

Gate 1 is the one that matters most. MALC_2D indexes its matrix as
[y_index, x_index] while ours is p_mat[y0_bin, y1_bin], so a missing
transpose swaps the axes and returns p(-tau) instead of p(tau). Nothing
raises, the mass still integrates to 1, and every symmetric test passes --
which is exactly why it needs an ASYMMETRIC fixture and a dedicated gate.

Gates
  1  axis convention   an asymmetric joint must come back with E[Y0] and
                       E[Y1] on the right axes, and p(tau) peaked at
                       +(E[Y1]-E[Y0]), not at its negative
  2  normalisation     \\int p_malc(tau) dtau = 1 over the full tau axis
  3  region 0 only     MALC and raw must be IDENTICAL where the interior
                       cannot contribute (|tau| > 2), proving the 8 tail
                       regions were not touched
  4  hull diagnostic   p(tau) = 0 exactly outside malc_hull_tau_range
  5  mean override     Joint2D.mean(inner=inner_mean()) == Joint2D.mean()
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from density_common import (                                        # noqa: E402
    Joint2D, TAU_CENTERS, fit_malc_interior, malc_tau_density,
    malc_interior_tau, malc_hull_tau_range, malc_inner_mean, _malc_module,
    joint_tau_density, mass,
)

RESULTS: list[tuple[str, bool, str]] = []


def check(name, ok, detail=''):
    RESULTS.append((name, bool(ok), detail))
    print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f'   {detail}' if detail else ''))


def make_joint(J=32, mu0=-0.5, mu1=+0.5, s0=0.10, s1=0.10, rho_logits=None,
               seed=0):
    """A deliberately ASYMMETRIC Joint2D: E[Y0] != E[Y1], and s0 may != s1."""
    edges = np.linspace(-1.0, 1.0, J + 1)
    c = 0.5 * (edges[:-1] + edges[1:])
    g0 = np.exp(-0.5 * ((c - mu0) / s0) ** 2)
    g1 = np.exp(-0.5 * ((c - mu1) / s1) ** 2)
    p_mat = np.outer(g0 / g0.sum(), g1 / g1.sum())
    pred = np.concatenate([
        np.log(np.maximum(p_mat.reshape(-1), 1e-300)),
        np.array([6.0] + [0.0] * 8),                 # region weights: ~all inner
        np.zeros(4),                                  # tail scales
    ])
    return Joint2D.from_pred(pred, J, edges), edges, float(c @ (g0 / g0.sum())), \
        float(c @ (g1 / g1.sum()))


print(__doc__.splitlines()[0])
print()

# -- gate 1: axis convention -------------------------------------------------
jt, edges, e_y0, e_y1 = make_joint()
fit = fit_malc_interior(jt.p_mat, edges, B=1000, seed=0)
inner = malc_inner_mean(fit, -1.0, 1.0)
ok = (fit is not None and inner is not None
      and abs(inner[0] - e_y0) < 0.05 and abs(inner[1] - e_y1) < 0.05)
check('1a axis: interior mean lands on the right axes', ok,
      f'truth ({e_y0:+.3f}, {e_y1:+.3f})  malc ({inner[0]:+.3f}, {inner[1]:+.3f})')

p = malc_interior_tau(fit, -1.0, 1.0, TAU_CENTERS, w0=1.0)
peak = float(TAU_CENTERS[int(np.argmax(p))])
expect = e_y1 - e_y0
check('1b axis: p(tau) peaks at +(E[Y1]-E[Y0]), not its negative',
      abs(peak - expect) < 0.05,
      f'expect {expect:+.3f}  peak {peak:+.3f}  (a missing .T gives {-expect:+.3f})')

# The failure mode itself, asserted directly.
raw = _malc_module().MALC_2D(jt.p_mat / jt.p_mat.sum(), edges, edges, K=1,
                             B_fit=1000, B_select=1000, parallel=False, seed=0)
bad = malc_inner_mean(raw, -1.0, 1.0)
check('1c axis: the untransposed call really does swap them (guard is needed)',
      bad is not None and abs(bad[0] - e_y1) < 0.05,
      f'untransposed gives ({bad[0]:+.3f}, {bad[1]:+.3f})')

# -- gate 2: normalisation ---------------------------------------------------
p_full = malc_tau_density(fit, jt.density, TAU_CENTERS, w0=float(jt.w[0]),
                          lo=-1.0, hi=1.0, pad=8.0 * jt.max_scale,
                          align_bins=jt.p_mat.shape[0])
m = mass(p_full, TAU_CENTERS)
check('2  normalisation: int p_malc(tau) dtau = 1', abs(m - 1.0) < 2e-3,
      f'mass = {m:.6f}')

# -- gate 3: region 0 only ---------------------------------------------------
# The interior diagonal spans tau in [-2, 2]; beyond that only the 8 tail
# regions can contribute, and those are supposed to be untouched.
p_raw = joint_tau_density(jt, TAU_CENTERS, n_y0=4096)
far = np.abs(TAU_CENTERS) > 2.0
d_far = float(np.abs(p_full[far] - p_raw[far]).max())
check('3  region 0 only: MALC == raw where the interior cannot reach',
      d_far < 1e-12, f'max|malc - raw| over |tau|>2 is {d_far:.3e}')
i_far = float(np.abs(malc_interior_tau(fit, -1.0, 1.0, TAU_CENTERS[far],
                                       w0=1.0)).max())
check('3b region 0 only: the MALC interior itself is 0 beyond |tau|=2',
      i_far == 0.0, f'max interior term = {i_far:.3e}')

# -- gate 4: hull diagnostic -------------------------------------------------
t_lo, t_hi = malc_hull_tau_range(fit)
probe = np.array([t_lo - 0.05, t_lo + 0.05, 0.5 * (t_lo + t_hi),
                  t_hi - 0.05, t_hi + 0.05])
vals = malc_interior_tau(fit, -1.0, 1.0, probe, w0=1.0)
outside = (probe < t_lo) | (probe > t_hi)
check('4  hull: interior density is 0 outside the hull tau-range and >0 inside',
      bool(np.all(vals[outside] == 0.0) and np.all(vals[~outside] > 0.0)),
      f'range [{t_lo:+.3f}, {t_hi:+.3f}]  values ' +
      ' '.join(f'{v:.2e}' for v in vals))

# -- gate 5: mean override ---------------------------------------------------
a = jt.mean()
b = jt.mean(inner=jt.inner_mean())
check('5  mean override: passing inner_mean() back reproduces mean()',
      max(abs(a[0] - b[0]), abs(a[1] - b[1])) < 1e-12,
      f'{a} vs {b}')

print()
n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
print(f'{len(RESULTS) - n_fail}/{len(RESULTS)} gates passed')
sys.exit(1 if n_fail else 0)
