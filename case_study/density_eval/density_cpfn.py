"""CausalPFN (cpfn1d / cpfn2d) tau densities — the piece `density_common.py`
never had.

`density_common.py` implements `UWYK1D`, `DoPFN1D` and `Joint2D`, all of which
unpack RAW HEAD LOGITS: K bars plus half-Gaussian tail masses and scales. The
CausalPFN evals do not expose logits. What they expose is `DENSITY_DUMP=1`,
which writes already-softmaxed bin probabilities:

    1D  (eval_causalpfn_v0_realcause.py)   edges (J+1,)
                                           p_y0_scaled (N_q, J)
                                           p_y1_scaled (N_q, J)
                                           y_shift, y_scale
    2D  (eval_cpfn2d_realcause.py)         + p_joint_scaled (N_q, J, J)
                                           + true_cate_per_query (N_q,)

So a CausalPFN prediction is a PURE HISTOGRAM: no tail parameters exist in the
dump, and none can be recovered from it.

CONSEQUENCE — COMPACT SUPPORT. p(y) is exactly 0 outside [edges[0], edges[-1]],
hence p(tau) is exactly 0 outside +/-(edges[-1] - edges[0]). A true tau landing
outside can never be covered at any level and gets NLL = +inf. That is a real
property of the estimator, the same one `density_common` documents for MALC,
not a bug to paper over. `support_diagnostics` reports the fraction affected;
report it next to coverage or the coverage number is not interpretable.

BECAUSE THERE ARE NO TAILS, the tau density is EXACT and needs no quadrature.
`joint_tau_density` / `uwyk_tau_density` split p(tau) into an exact interior
diagonal term plus a quadrature over the 8 outside regions. Here regions 1-8
carry zero mass, so only the interior term survives:

    p(tau) = (1/bw) * interp_bw( sum_i p[i, i+k] )

which is the same `_interior_tau` those functions call. We call it directly:
same numbers, no wasted quadrature, and no dependence on tail scales that do
not exist. `to_joint2d()` / `to_uwyk1d()` adapters are provided anyway so the
generic functions still work (they return the identical answer, just slower) —
`test_density_cpfn.py` asserts that equivalence.

UNITS. Densities live on the SCALED y axis. tau = Y1 - Y0, so y_shift cancels
and tau_raw = tau_scaled * y_scale exactly. Evaluate on the scaled tau grid and
pass `y_scale` to `interval_metrics.query_metrics`, exactly as
`density_common.point_metrics(cate_scaled, true_cate, y_scale)` does. Convert a
raw truth with `scaled_truth = true_cate / y_scale`.

Nothing outside `case_study/` is imported for modification; the reusable
primitives come from the LOCAL copy of density_common.py in this folder.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from density_common import (          # LOCAL copy in case_study/density_eval/
    Joint2D, UWYK1D, _diag_sums, _diag_sums_product, _interior_tau,
)

_TRAPZ = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
_SUM_TOL = 1e-3          # float32 dumps: bin probs rarely sum to 1 exactly


def _check_edges(edges):
    edges = np.asarray(edges, dtype=np.float64).reshape(-1)
    if edges.size < 2:
        raise ValueError('edges must have at least 2 entries')
    w = np.diff(edges)
    if np.any(w <= 0):
        raise ValueError('edges must be strictly increasing')
    bw = float(w.mean())
    rel = float(np.abs(w - bw).max() / bw)
    if rel > 1e-4:
        # The exact interior formula assumes a uniform grid, same requirement
        # density_common.uwyk_tau_density enforces.
        raise ValueError(f'CausalPFN bins must be uniform; max relative width '
                         f'deviation {rel:.2e} > 1e-4')
    return edges, bw


def _norm(p, what):
    p = np.asarray(p, dtype=np.float64)
    if np.any(p < 0):
        raise ValueError(f'{what} has negative probabilities')
    s = float(p.sum())
    if s <= 0:
        raise ValueError(f'{what} sums to {s}')
    if abs(s - 1.0) > _SUM_TOL:
        raise ValueError(f'{what} sums to {s:.6f}, not 1 (tol {_SUM_TOL}); '
                         'the dump is not a normalised histogram')
    return p / s


# ── one-query containers ─────────────────────────────────────────────────────
@dataclass
class CPFN1D:
    """One arm of a cpfn1d prediction: a normalised histogram on `edges`."""
    p: np.ndarray              # (J,) sums to 1
    edges: np.ndarray          # (J+1,)

    @classmethod
    def from_dump(cls, p_row, edges):
        edges, _ = _check_edges(edges)
        p = _norm(np.asarray(p_row, dtype=np.float64).reshape(-1), 'p_y*_scaled')
        if p.size != edges.size - 1:
            raise ValueError(f'{p.size} bins vs {edges.size - 1} implied by edges')
        return cls(p=p, edges=edges)

    @property
    def bw(self):
        return float((self.edges[-1] - self.edges[0]) / self.p.size)

    @property
    def support(self):
        return float(self.edges[0]), float(self.edges[-1])

    def density(self, y):
        y = np.asarray(y, dtype=np.float64)
        out = np.zeros(np.shape(y), dtype=np.float64)
        inside = (y >= self.edges[0]) & (y < self.edges[-1])
        if np.any(inside):
            k = np.clip(np.searchsorted(self.edges[1:-1], y[inside], side='right'),
                        0, self.p.size - 1)
            out[inside] = self.p[k] / self.bw
        return out

    def mean(self):
        centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        return float(self.p @ centers)

    def to_uwyk1d(self) -> UWYK1D:
        """Adapter: zero mass in both half-Gaussian tails.

        The scales must stay finite and positive (they set the quadrature pad
        in uwyk_tau_density and would produce 0*inf if driven to 0), but with
        log_p{L,R} = -inf they contribute exactly zero density.
        """
        return UWYK1D(
            log_pL=-np.inf, log_pBars=np.log(np.maximum(self.p, 1e-300)),
            log_pR=-np.inf, sL=self.bw, sR=self.bw,
            edges=self.edges, widths=np.diff(self.edges),
        )


@dataclass
class CPFN2D:
    """One cpfn2d prediction: a normalised joint histogram on edges x edges."""
    p_mat: np.ndarray          # (J, J) sums to 1
    edges: np.ndarray          # (J+1,)

    @classmethod
    def from_dump(cls, p_joint, edges):
        edges, _ = _check_edges(edges)
        p = np.asarray(p_joint, dtype=np.float64)
        if p.ndim != 2 or p.shape[0] != p.shape[1]:
            raise ValueError(f'p_joint must be square (J, J), got {p.shape}')
        if p.shape[0] != edges.size - 1:
            raise ValueError(f'{p.shape[0]} bins vs {edges.size - 1} from edges')
        return cls(p_mat=_norm(p, 'p_joint_scaled'), edges=edges)

    @property
    def bw(self):
        return float((self.edges[-1] - self.edges[0]) / self.p_mat.shape[0])

    @property
    def support(self):
        return float(self.edges[0]), float(self.edges[-1])

    def marginals(self):
        return self.p_mat.sum(axis=1), self.p_mat.sum(axis=0)

    def mean(self):
        centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        m0, m1 = self.marginals()
        return float(m0 @ centers), float(m1 @ centers)

    def cate(self):
        m0, m1 = self.mean()
        return m1 - m0

    def rho(self):
        """Correlation of the joint — the quantity the 1D path assumes is 0."""
        centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        m0, m1 = self.marginals()
        E0, E1 = float(m0 @ centers), float(m1 @ centers)
        E01 = float((self.p_mat * centers[:, None] * centers[None, :]).sum())
        v0 = max(float((centers ** 2) @ m0) - E0 ** 2, 1e-12)
        v1 = max(float((centers ** 2) @ m1) - E1 ** 2, 1e-12)
        return float((E01 - E0 * E1) / np.sqrt(v0 * v1))

    def independent(self) -> tuple[CPFN1D, CPFN1D]:
        """The SAME prediction with the coupling thrown away.

        p_mat -> outer(marginals). Running `cpfn_tau_density` on this and on
        the joint isolates the cost of the Y0 _|_ Y1 assumption with everything
        else held fixed: identical model, identical query, identical bins.
        """
        m0, m1 = self.marginals()
        return (CPFN1D(p=m0 / m0.sum(), edges=self.edges),
                CPFN1D(p=m1 / m1.sum(), edges=self.edges))

    def to_joint2d(self) -> Joint2D:
        """Adapter: all mass in region 0 (inner x inner), regions 1-8 empty."""
        w = np.zeros(9, dtype=np.float64)
        w[0] = 1.0
        bw = self.bw
        return Joint2D(p_mat=self.p_mat, w=w, sL0=bw, sR0=bw, sL1=bw, sR1=bw,
                       rho=float(np.clip(self.rho(), -1 + 1e-6, 1 - 1e-6)),
                       edges=self.edges)


# ── tau densities (exact; no quadrature, because there are no tails) ─────────
def cpfn2d_tau_density(jt: CPFN2D, tau_points):
    """p(tau) for the cpfn2d joint. Exact: the interior diagonal term only."""
    return _interior_tau(_diag_sums(jt.p_mat), tau_points, jt.bw, 1.0)


def cpfn1d_tau_density(f0: CPFN1D, f1: CPFN1D, tau_points):
    """p(tau) for cpfn1d under Y0 _|_ Y1. Exact: interior diagonal only."""
    if f0.p.size != f1.p.size or not np.allclose(f0.edges, f1.edges):
        raise ValueError('cpfn1d arms must share a bin grid')
    return _interior_tau(_diag_sums_product(f0.p, f1.p), tau_points, f0.bw, 1.0)


# ── per-arm scaling: the two arms live on DIFFERENT raw grids ───────────────
def raw_edges(edges, shift, scale):
    """Map an arm's scaled bin edges to RAW outcome units."""
    return np.asarray(edges, dtype=np.float64) * float(scale) + float(shift)


def cpfn1d_tau_density_raw(p0, e0, p1, e1, tau_points):
    r"""p(tau) in RAW units for two histograms on DIFFERENT uniform grids.

    Needed for STD_MODE=per_arm, where arm0 and arm1 carry their own
    (shift, scale): the arms' raw bin grids then have different widths AND
    different offsets, so `_diag_sums_product` -- which assumes one shared
    uniform grid -- does not apply.

    Exact form. With tau = Y1 - Y0 and both densities piecewise constant,

        p(tau) = \int p0(y) p1(y + tau) dy
               = sum_i (p0[i] / w0) * [ P1(e0[i+1] + tau) - P1(e0[i] + tau) ]

    P1 is the arm-1 CDF, which is piecewise LINEAR, so np.interp evaluates it
    exactly at any point. Cost is J interpolations over the tau grid (J=32
    here), not J^2, and there is no rebinning and no quadrature.

    Under pooled scaling (e0 == e1) this returns the same answer as
    `cpfn1d_tau_density` rescaled by y_scale -- asserted in the tests.
    """
    p0 = np.asarray(p0, dtype=np.float64).reshape(-1)
    p1 = np.asarray(p1, dtype=np.float64).reshape(-1)
    e0 = np.asarray(e0, dtype=np.float64).reshape(-1)
    e1 = np.asarray(e1, dtype=np.float64).reshape(-1)
    if p0.size != e0.size - 1 or p1.size != e1.size - 1:
        raise ValueError('bin counts must match their edge arrays')
    if not (np.all(np.diff(e0) > 0) and np.all(np.diff(e1) > 0)):
        raise ValueError('edges must be strictly increasing')

    cdf1 = np.concatenate([[0.0], np.cumsum(p1)])
    cdf1 /= cdf1[-1]
    tau = np.atleast_1d(np.asarray(tau_points, dtype=np.float64))

    out = np.zeros(tau.shape, dtype=np.float64)
    w0 = np.diff(e0)
    p0n = p0 / p0.sum()
    for i in range(p0n.size):
        if p0n[i] <= 0.0:
            continue
        hi = np.interp(e0[i + 1] + tau, e1, cdf1, left=0.0, right=1.0)
        lo = np.interp(e0[i] + tau, e1, cdf1, left=0.0, right=1.0)
        out += (p0n[i] / w0[i]) * (hi - lo)
    return out


def tau_support_raw(e0, e1):
    """(lo, hi) of tau = Y1 - Y0 for arms on raw grids e0, e1."""
    e0 = np.asarray(e0, dtype=np.float64)
    e1 = np.asarray(e1, dtype=np.float64)
    return float(e1[0] - e0[-1]), float(e1[-1] - e0[0])


def tau_support(edges):
    """(lo, hi) of tau = Y1 - Y0 for a histogram on `edges`."""
    edges = np.asarray(edges, dtype=np.float64)
    span = float(edges[-1] - edges[0])
    return -span, span


def support_diagnostics(edges, true_tau_scaled):
    """How many truths the compact support can never cover.

    Report this beside coverage: a query whose truth is outside support is a
    guaranteed miss at EVERY level, so a model with a narrow y range is
    penalised on coverage for a reason that has nothing to do with calibration.
    """
    lo, hi = tau_support(edges)
    t = np.asarray(true_tau_scaled, dtype=np.float64).reshape(-1)
    out = (t < lo) | (t > hi)
    return {'tau_support_lo': lo, 'tau_support_hi': hi,
            'n_queries': int(t.size),
            'outside_support_frac': float(out.mean()) if t.size else 0.0}


# ── dump loading ─────────────────────────────────────────────────────────────
_REQ_1D = ('edges', 'p_y0_scaled', 'p_y1_scaled')
# per-arm dumps carry arm0_/arm1_ stats instead of a single y_scale
_PER_ARM = ('arm0_shift', 'arm0_scale', 'arm1_shift', 'arm1_scale')
_REQ_2D = _REQ_1D + ('p_joint_scaled',)


def load_dump(path):
    """Read a DENSITY_DUMP npz. Returns a dict with `kind` in {1d, 2d}."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    z = np.load(path, allow_pickle=True)
    keys = set(z.files)
    kind = '2d' if 'p_joint_scaled' in keys else '1d'
    need = _REQ_2D if kind == '2d' else _REQ_1D
    missing = [k for k in need if k not in keys]
    if missing:
        raise KeyError(f'{path}: dump is missing {missing}. Was the eval run '
                       f'with DENSITY_DUMP=1?')
    per_arm = all(k in keys for k in _PER_ARM)
    if not per_arm and 'y_scale' not in keys:
        raise KeyError(f'{path}: needs either y_scale (pooled) or '
                       f'{list(_PER_ARM)} (per-arm).')
    out = {'kind': kind, 'edges': np.asarray(z['edges'], dtype=np.float64),
           'p_y0_scaled': np.asarray(z['p_y0_scaled'], dtype=np.float64),
           'p_y1_scaled': np.asarray(z['p_y1_scaled'], dtype=np.float64),
           'y_scale': (float(np.asarray(z['y_scale']).reshape(-1)[0])
                       if 'y_scale' in keys else 1.0),
           'y_shift': float(np.asarray(z['y_shift']).reshape(-1)[0])
                      if 'y_shift' in keys else 0.0}
    out['per_arm'] = per_arm
    if per_arm:
        for k in _PER_ARM:
            out[k] = float(np.asarray(z[k]).reshape(-1)[0])
        # raw bin grids, one per arm
        out['e0_raw'] = raw_edges(out['edges'], out['arm0_shift'], out['arm0_scale'])
        out['e1_raw'] = raw_edges(out['edges'], out['arm1_shift'], out['arm1_scale'])
    if kind == '2d':
        out['p_joint_scaled'] = np.asarray(z['p_joint_scaled'], dtype=np.float64)
    if 'true_cate_per_query' in keys:
        out['true_cate_per_query'] = np.asarray(z['true_cate_per_query'],
                                                dtype=np.float64)
    return out


def iter_queries(dump):
    """Yield (index, tau_density_callable, extras) per query in a dump.

    The callable takes `tau_points` on the SCALED axis. For a 2d dump it also
    exposes the independence reconstruction of the SAME query under
    `extras['indep_density']`, so the joint-vs-independent comparison needs no
    second model run.
    """
    edges = dump['edges']
    n = dump['p_y0_scaled'].shape[0]
    for q in range(n):
        if dump['kind'] == '2d':
            jt = CPFN2D.from_dump(dump['p_joint_scaled'][q], edges)
            f0, f1 = jt.independent()
            extras = {'rho': jt.rho(), 'cate_scaled': jt.cate(),
                      'indep_density': (lambda t, a=f0, b=f1:
                                        cpfn1d_tau_density(a, b, t))}
            yield q, (lambda t, j=jt: cpfn2d_tau_density(j, t)), extras
        elif dump.get('per_arm'):
            # RAW units: the arms have different (shift, scale), so there is no
            # single scaled axis. Callers pass y_scale=1.0 and a RAW truth.
            p0 = dump['p_y0_scaled'][q]; p1 = dump['p_y1_scaled'][q]
            e0, e1 = dump['e0_raw'], dump['e1_raw']
            c0 = 0.5 * (e0[:-1] + e0[1:]); c1 = 0.5 * (e1[:-1] + e1[1:])
            extras = {'rho': 0.0, 'units': 'raw',
                      'cate_scaled': float((p1 / p1.sum()) @ c1
                                           - (p0 / p0.sum()) @ c0),
                      'tau_support': tau_support_raw(e0, e1)}
            yield q, (lambda t, a=p0, b=p1, x=e0, y=e1:
                      cpfn1d_tau_density_raw(a, x, b, y, t)), extras
        else:
            f0 = CPFN1D.from_dump(dump['p_y0_scaled'][q], edges)
            f1 = CPFN1D.from_dump(dump['p_y1_scaled'][q], edges)
            extras = {'rho': 0.0, 'units': 'scaled',
                      'cate_scaled': f1.mean() - f0.mean()}
            yield q, (lambda t, a=f0, b=f1: cpfn1d_tau_density(a, b, t)), extras
