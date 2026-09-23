"""Tier-C density eval, MALC VARIANT T: p(tau | x) smoothed in tau space.

The T counterpart of eval_density_tauC_malc.py (variant J). Same methods, same
truth, same metrics, same output schema -- summarize_density_tauC.py reads this
directory unchanged. What differs is WHERE the log-concave fit sits.

  Variant J   smooth the JOINT p(y0, y1) on the plane, then integrate the
              anti-diagonal to get tau. 1D methods build the independence
              joint f0 (x) f1 first, so the fit still happens in 2D.
              eval_density_tauC_malc.py.

  Variant T   form tau FIRST -- anti-diagonal projection for a 2D head,
              independence convolution for a 1D one -- then smooth that single
              1D density. This module.

T is not a cheaper approximation to J; the two smooth different objects. J
regularises the dependence structure as well as the shape, because a
log-concave fit on the plane constrains how the arms co-vary. T leaves the tau
pmf's construction untouched and only regularises its shape. Where they
disagree, the disagreement is attributable to the joint, which is the point of
running both. See UWYK_Fig3_4/tau_smoother.py for the same argument from the
smoother's side.

WHAT IS REUSED, AND WHY THAT MATTERS. The interior tau pmf this module feeds to
MALC is not new code: density_common._diag_sums (2D) and _diag_sums_product (1D)
already return S[k] = sum_i p(i, i+k) on a uniform lattice of spacing bw, which
is exactly the (atoms, pmf) pair tau_smoother.smooth_tau_pmf expects. So the raw
arm, the J arm and this one all build tau from the same two functions, and a
raw-vs-T difference cannot come from a difference in how tau was formed.

THE ONE STEP THAT FAILS SILENTLY. smooth_tau_pmf returns a pmf NORMALISED to
sum 1, which throws away the interior mass. The interior term must therefore be
multiplied back by w0 * S.sum():

    joint     S sums to 1        -> mass is w0 = jt.w[0]
    product   S sums to p0.p1    -> mass is that product, w0 = 1

Skip it and the density still integrates to something finite and still scores,
it just scores as a biased model. The `mass` column is the check: it must land
where the raw arm's does.

THE CATE CORRECTION IS EXACT, not a re-derivation. E[tau] splits by region as
w0 * m * E_S[tau] + (tail moment), and T changes only the first term, so

    cate_T = cate_raw + w0 * m * (E_smooth[tau] - E_S[tau])

with no need to touch the 8 tail regions at all. This is why variant T has no
analogue of J's malc_inner_mean / Joint2D.mean(inner=...) machinery: in tau
space the quantity being corrected is a scalar mean, not a pair of arm means.

MALC_N_Y0 IS GONE, deliberately. In J it sets the y0 quadrature that integrates
the 2D fit along the diagonal, with a measured 1.1e-4 interior error. Variant T
has no such integral -- the fit is already a function of tau -- so the knob is
not carried forward inert. That is a small accuracy gain for T over J and is
stated here rather than buried.

UNIFORM GRID. MALC_1D refuses a non-uniform grid rather than mis-calibrating
its Beta jitter against a varying bin width. DoPFN's borders are quantile
allocated (measured on IHDP r000 at 0.107 to 536, a 5032x ratio), so its arms
are rebinned onto a uniform grid first, exactly as in the J arm and with the
same DOPFN_MALC_BINS default. UWYK's bars are already uniform and CausalPFN1D
validates uniformity in from_pred, so both pass through untouched.

DOPFN'S RECONSTRUCTION IS THE REBINNED ONE. dopfn_tau_density is a closed form
for unequal bars that does not decompose into interior + tail, so it cannot be
used as the tail source here. Both MALC arms instead reconstruct DoPFN from its
rebinned uniform arms, and keep `raw=` on the native exact density so a
FALLBACK query is scored identically in the raw and MALC arms. Inherited from
variant J unchanged, so J and T stay comparable.

FALLBACK. The log-concave MLE is supported on the convex hull of its B
synthetic points -- in 1D, an interval strictly inside the atom range -- and is
exactly zero outside it. The TOTAL density stays positive there because the 8
tail regions have infinite support, which is the trap: the NLL comes back
finite but is pure tail, ~1e-3 of what it should be. So the trigger is the
INTERIOR term, not the total, exactly as in variant J. Those queries fall back
to the RAW density and are COUNTED in n_fallback_<method>, never dropped -- a
silent drop would quietly change which queries each method is scored on.

WATCH THE COARSE HEADS. Variant T gives MALC_1D only 2J-1 atoms. That is 1999
for UWYK's K=1000 arms and 63 for a J=32 joint, but only 19 for DoPFN's J=10
joint. The J arm already measured that head falling back on 11/75 queries
(14.7%) at B=100 with 100 bins to work from; with 19 atoms it has less. Read
n_fallback_dopfn_repro_joint2d before trusting that row, and prefer B=1000.

CausalPFN has no tail model at all (CausalPFN1D is a finite histogram with no
invented tail mass), so for its 1D rows the smoothed interior IS the whole
density. Its raw density is likewise zero off support, so an out-of-support
tau* gives nll=+inf in the raw arm too -- pre-existing, not introduced here.

Runs entirely off the prediction dumps eval_density_tauC.py writes with
SAVE_PREDICTIONS=1. CPU only -- no checkpoint, no GPU, no harness import, no
dataset loader. Parallel over QUERIES, one realization at a time.

Usage (CPU node):
    DUMPS=./results_density_tauC/5312884/IHDP/predictions \
    OUT=./results_density_tauC_malcT/IHDP \
    MALC_B=1000 N_WORKERS=32 \
    python -u benchmarks/eval_graph2d/eval_density_tauC_malcT.py
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
import zlib

# MALC's solver and numpy are both threaded; with one process per query they
# oversubscribe badly. BLAS reads these AT IMPORT, so this must stay ABOVE
# `import numpy` -- set below it they are silently already ignored.
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
# tau_smoother lives with the 1D/2D comparison figures, not in this package.
sys.path.insert(0, os.path.join(_REPO, 'UWYK_Fig3_4'))

from density_common import (                                        # noqa: E402
    Joint2D, UWYK1D, DoPFN1D, CausalPFN1D, independent_f2d,
    joint_tau_density, uwyk_tau_density, dopfn_tau_density,
    causalpfn_tau_density, truth_tau_density,
    l2_distance, kl, mass, point_metrics, TAU_CENTERS,
    # Private by name, shared by design: these three ARE the region
    # decomposition, and variant T has to split the density at exactly the
    # same seam the raw arm does or the tails would stop being comparable.
    _diag_sums, _diag_sums_product, _outside_only, _tail_term,
)
from tau_smoother import SmootherConfig, smooth_tau_pmf                # noqa: E402

DUMPS = os.environ['DUMPS']
OUT = os.environ.get('OUT', './results_density_tauC_malcT')
MALC_B = int(os.environ.get('MALC_B', '1000'))
MALC_K = int(os.environ.get('MALC_K', '1'))
# Base offset for the per-query seeds. 0 reproduces the J arm's recipe
# exactly, which is the default so UWYK rows draw from the same stream in
# both arms; raise it to repeat a run with an independent set of Beta-jitter
# draws and see how much of a result is the draw.
MALC_SEED = int(os.environ.get('MALC_SEED', '0'))
N_Y0 = int(os.environ.get('N_Y0', '4096'))          # tails, matches the raw run
N_WORKERS = int(os.environ.get(
    'N_WORKERS', os.environ.get('SLURM_CPUS_PER_TASK', '8')))
REAL_START = int(os.environ.get('REAL_START', '0'))
REAL_END = os.environ.get('REAL_END')

# Points the smoothed tau density is evaluated on before interpolation onto
# TAU_CENTERS. 12001 matches TAU_CENTERS' own resolution over a support that is
# narrower (the atom range, ~+-2), so the interpolation never coarsens the fit.
# dmalc_1d is closed form, so this costs nothing next to the fit itself.
MALC_N_TAU = int(os.environ.get('MALC_N_TAU', '12001'))

# uwyk | dopfn | causalpfn | all | auto. 'auto' takes whichever families the
# dump carries, so a single-family dump needs no flag.
MODEL_FAMILY = os.environ.get('MODEL_FAMILY', 'auto')
if MODEL_FAMILY not in ('uwyk', 'dopfn', 'causalpfn', 'all', 'auto'):
    raise ValueError(
        'MODEL_FAMILY must be uwyk, dopfn, causalpfn, all, or auto')

# Bins for the uniform grid DoPFN's 1D arms are rebinned onto -- see
# product_bundle. 1024 matches UWYK's 1000-bar product in both cost and
# fidelity: measured 0.004% relative L2 against the exact unequal-bar tau
# density, against 0.35% at 100 bins and 0.013% at 512.
DOPFN_MALC_BINS = int(os.environ.get('DOPFN_MALC_BINS', '1024'))

_G: dict = {}


# ---------------------------------------------------------------------------
def score(p_est, p_true, tau_star_density, tau_grid=TAU_CENTERS):
    """Byte-identical to eval_density_tauC.score -- same metrics, same order."""
    return dict(
        nll=float(-np.log(tau_star_density)),
        l2=l2_distance(p_true, p_est, tau_grid),
        kl_fwd=kl(p_true, p_est, tau_grid),
        kl_rev=kl(p_est, p_true, tau_grid),
        mass=mass(p_est, tau_grid),
    )


# ---------------------------------------------------------------------------
# Bundles: (interior tau pmf, tail term, raw density) per method.
#
# The ONLY place the methods differ. Everything downstream is shared, which is
# what makes the comparison a comparison. Each bundle carries
#
#   S       interior tau pmf on a uniform lattice, UNNORMALISED
#   bw      that lattice's spacing
#   w0      extra weight multiplier (jt.w[0] for a joint, 1 for a product)
#   tail    callable tau -> the 8 non-interior regions
#   raw     callable tau -> the full raw density, for fallback queries
#   raw_mean  callable -> E[tau] of the raw density
# ---------------------------------------------------------------------------
def joint_bundle(jt: Joint2D):
    """A trained joint head: its own diagonal sums are the MALC input."""
    lo, hi, pad = jt.lo, jt.hi, 8.0 * jt.max_scale
    align = jt.p_mat.shape[0]
    return dict(
        S=_diag_sums(jt.p_mat), bw=jt.bw, w0=float(jt.w[0]),
        tail=lambda t: _tail_term(_outside_only(jt.density, lo, hi),
                                  t, lo, hi, pad, N_Y0, align_bins=align),
        raw=lambda t: joint_tau_density(jt, t, n_y0=N_Y0),
        raw_mean=lambda: (lambda m: m[1] - m[0])(jt.mean()),
    )


def product_bundle(a: UWYK1D, b: UWYK1D, *, raw_density, uniform_bins=None):
    """Two 1D arms convolved under independence.

    `uniform_bins` rebins both arms onto a uniform grid first, and DoPFN needs
    it: MALC_1D reads ONE bin width off the grid to calibrate its Beta jitter
    and raises rather than guessing when the widths disagree. UWYK's bars are
    uniform so this is a no-op there. Both arms share a border set, so one grid
    over the common interior support serves both, and the half-normal tails --
    anchored at that support's ends, which the rebin grid preserves exactly --
    carry through untouched.

    `raw_density` stays a closure on the arms the CALLER passed, i.e. the
    native ones. A fallback query is then scored identically in the raw and
    MALC arms, and raw-vs-T stays attributable to the smoothing.
    """
    if uniform_bins is not None:
        if not np.allclose(a.edges, b.edges):
            raise ValueError('product_bundle: arms must share a border grid')
        grid = np.linspace(float(a.edges[0]), float(a.edges[-1]),
                           uniform_bins + 1)
        a, b = a.rebin(grid), b.rebin(grid)

    pa, pb = np.exp(a.log_pBars), np.exp(b.log_pBars)
    lo, hi = float(a.edges[0]), float(a.edges[-1])
    pad = 8.0 * max(a.max_scale, b.max_scale)
    align = len(a.widths)
    # The raw arm's own uniformity gate, applied to the grid MALC will see.
    bw = float(np.mean(a.widths))
    rel_dev = float(np.abs(a.widths - bw).max() / bw) if bw > 0 else np.inf
    if rel_dev > 1e-4:
        raise ValueError(
            f'variant T needs a uniform interior lattice; max relative width '
            f'deviation is {rel_dev:.2e} (> 1e-4). Pass uniform_bins to rebin.')
    return dict(
        S=_diag_sums_product(pa, pb), bw=bw, w0=1.0,
        tail=lambda t: _tail_term(_outside_only(independent_f2d(a, b), lo, hi),
                                  t, lo, hi, pad, N_Y0, align_bins=align),
        raw=lambda t: raw_density(t),
        # Rebinned, matching S. rebin preserves interior mass exactly, so the
        # base mean is coherent with the correction applied to it.
        raw_mean=lambda a=a, b=b: b.mean() - a.mean(),
    )


def causalpfn_bundle(f0: CausalPFN1D, f1: CausalPFN1D):
    """Two finite CausalPFN histograms: interior only, no tail model.

    CausalPFN1D invents no tail mass, so causalpfn_tau_density IS the interior
    term and the tail contributes exactly zero. That makes these the only rows
    where the smoothed object is the entire predictive.
    """
    if f0.edges.shape != f1.edges.shape or not np.allclose(f0.edges, f1.edges):
        raise ValueError('CausalPFN arms must share an outcome grid')
    return dict(
        S=_diag_sums_product(f0.p, f1.p), bw=f0.bw, w0=1.0,
        tail=lambda t: np.zeros(np.shape(np.atleast_1d(t)), dtype=np.float64),
        raw=lambda t: causalpfn_tau_density(f0, f1, t),
        raw_mean=lambda: f1.mean() - f0.mean(),
    )


# ---------------------------------------------------------------------------
def smooth_interior(S, bw, w0, seed):
    """MALC-1D fit of one interior tau pmf.

    Returns (density_at, hull_lo, hull_hi, e_smooth, e_raw, ok), where
    `density_at` maps tau -> the smoothed interior DENSITY, already carrying
    the interior mass w0 * S.sum(). `ok` is False when smooth_tau_pmf fell back
    to the raw pmf, which it does rather than raising when a component fit is
    not estimable.
    """
    S = np.asarray(S, dtype=np.float64).reshape(-1)
    n = S.size
    J = (n + 1) // 2
    atoms = np.arange(-(J - 1), J, dtype=np.float64) * bw
    m = float(S.sum())
    e_raw = float(atoms @ S) / m if m > 0 else float('nan')
    if not np.isfinite(m) or m <= 0 or not np.isfinite(S).all():
        return None, float('nan'), float('nan'), float('nan'), e_raw, False

    cfg = SmootherConfig(enabled=True, B=MALC_B, K=MALC_K, seed=int(seed),
                         n_tau=MALC_N_TAU, n_workers=1)
    grid, pmf = smooth_tau_pmf(atoms, S / m, cfg)

    # smooth_tau_pmf signals failure by returning its INPUT back, not by
    # raising -- so that a density MALC cannot improve on stays in the table
    # instead of being dropped. Detect it by identity with the atom grid.
    if grid.shape == atoms.shape and np.array_equal(grid, atoms):
        return None, float('nan'), float('nan'), float('nan'), e_raw, False

    nz = np.nonzero(pmf > 0.0)[0]
    if nz.size == 0:
        return None, float('nan'), float('nan'), float('nan'), e_raw, False

    dg = float(grid[1] - grid[0])
    dens = pmf / dg * (w0 * m)
    e_smooth = float(grid @ pmf)

    def density_at(t):
        return np.interp(np.atleast_1d(np.asarray(t, dtype=np.float64)),
                         grid, dens, left=0.0, right=0.0)

    return (density_at, float(grid[nz[0]]), float(grid[nz[-1]]),
            e_smooth, e_raw, True)


# ---------------------------------------------------------------------------
def build_query(q):
    """The bundle per method for one query."""
    d = _G['d']
    out = {}

    if _G['uwyk']:
        jt = Joint2D.from_pred(d['joint_logits'][q], _G['J'], _G['e2'])
        f0 = UWYK1D.from_pred(d['uwyk_pred0'][q], _G['be'], _G['bw'],
                              _G['sL'], _G['sR'])
        f1 = UWYK1D.from_pred(d['uwyk_pred1'][q], _G['be'], _G['bw'],
                              _G['sL'], _G['sR'])
        f0m, f1m = f0.rebin(_G['e2']), f1.rebin(_G['e2'])
        out['joint'] = joint_bundle(jt)
        out['uwyk_native'] = product_bundle(
            f0, f1, raw_density=lambda t: uwyk_tau_density(f0, f1, t, n_y0=N_Y0))
        out['uwyk_matched'] = product_bundle(
            f0m, f1m,
            raw_density=lambda t: uwyk_tau_density(f0m, f1m, t, n_y0=N_Y0))

    for name, spec in _G['dopfn'].items():
        if spec['kind'] == 'joint':
            out[name] = joint_bundle(
                Joint2D.from_pred(spec['logits'][q], spec['J'], spec['edges2d']))
        else:
            g0 = DoPFN1D.from_pred(spec['pred0'][q], spec['borders'],
                                   y_shift=spec['y_shift'],
                                   y_scale=spec['y_scale'],
                                   tail_scales=spec['tail_scales'])
            g1 = DoPFN1D.from_pred(spec['pred1'][q], spec['borders'],
                                   y_shift=spec['y_shift'],
                                   y_scale=spec['y_scale'],
                                   tail_scales=spec['tail_scales'])
            out[name] = product_bundle(
                g0, g1,
                raw_density=lambda t, g0=g0, g1=g1: dopfn_tau_density(g0, g1, t),
                uniform_bins=DOPFN_MALC_BINS)

    for name, spec in _G['causalpfn'].items():
        if spec['kind'] == 'joint':
            # from_pred on the NATIVE edges, then affine -- not from_pred on
            # the mapped edges. Joint2D.affine deliberately keeps rho fixed,
            # because recomputing it on a rescaled axis can hit from_pred's
            # variance floor and change the correlated corner tails. This is
            # the same construction density_causalpfn.predict used at dump time.
            out[name] = joint_bundle(
                Joint2D.from_pred(spec['logits'][q], spec['J'],
                                  spec['edges2d_native'])
                .affine(spec['factor'], spec['offset']))
        else:
            out[name] = causalpfn_bundle(
                CausalPFN1D.from_pred(spec['pred0'][q], spec['edges']),
                CausalPFN1D.from_pred(spec['pred1'][q], spec['edges']))
    return out


#: Seed slots for the three UWYK methods, pinned to the values their position
#: in the old fixed METHODS tuple gave them, so UWYK numbers stay aligned with
#: the J arm's.
_SEED_SLOT = {'uwyk_native': 0, 'uwyk_matched': 1, 'joint': 2}


def seed_slot(name: str) -> int:
    """Per-method seed offset. Must NOT be the position in the method list.

    An index shifts when MODEL_FAMILY selects a different set, which would
    silently give a method a different MALC fit depending on what it happened
    to be run alongside -- scoring dopfn alone and scoring it with uwyk would
    disagree for no reason. Hash the name instead; crc32 because Python's
    hash() is salted per process.
    """
    if name in _SEED_SLOT:
        return _SEED_SLOT[name]
    return 3 + (zlib.crc32(name.encode()) % 100_000)


def dopfn_specs(d):
    """DoPFN methods in a dump, as {method: spec}. Handles both dump schemas.

    Lifted unchanged from the J arm so the two arms discover exactly the same
    method set from the same dump.
    """
    def arms(prefix, borders_key, scales_key, y_shift, y_scale):
        return dict(kind='1d', pred0=d[f'{prefix}pred0'],
                    pred1=d[f'{prefix}pred1'], borders=d[borders_key],
                    tail_scales=d.get(scales_key), y_shift=y_shift,
                    y_scale=y_scale)

    y_shift, y_scale = float(d['y_shift']), float(d['y_scale'])
    out = {}

    if 'dopfn_methods' in d:                                  # DoPFNModelSet
        names = [str(x) for x in np.atleast_1d(d['dopfn_methods'])]
        kinds = [str(x) for x in np.atleast_1d(d['dopfn_kinds'])]
        for name, kind in zip(names, kinds):
            if kind.endswith('joint'):
                out[name] = dict(kind='joint', logits=d[f'{name}_logits'],
                                 J=int(d[f'{name}_J']),
                                 edges2d=d[f'{name}_edges2d'])
            elif kind == 'repro_1d':
                out[name] = arms(f'{name}_', f'{name}_borders_raw',
                                 f'{name}_tail_scales_raw', y_shift, y_scale)
            else:                                             # library model
                out[name] = arms('dopfn_', 'dopfn_borders0_raw',
                                 'dopfn_tail_scales0_raw', y_shift, y_scale)
        return out

    if 'dopfn_pred0' in d:                          # DoPFNDensityModels (old)
        out['dopfn_native'] = arms('dopfn_', 'dopfn_borders0_raw',
                                   'dopfn_tail_scales0_raw', y_shift, y_scale)
    if 'dopfn_joint_logits' in d:
        out['dopfn_joint'] = dict(kind='joint', logits=d['dopfn_joint_logits'],
                                  J=int(d['dopfn_J']),
                                  edges2d=d['dopfn_edges2d'])
    return out


def causalpfn_specs(d):
    """CausalPFN methods in a dump, as {method: spec}.

    New in this arm -- variant J never supported CausalPFN, so these five rows
    have no J counterpart to compare against unless the J arm is backfilled.

    `causalpfn_methods` already carries fully qualified names (causalpfn_j32_1d
    and so on), matching the keys density_causalpfn.predict wrote.
    """
    if 'causalpfn_methods' not in d:
        return {}
    y_shift, y_scale = float(d['y_shift']), float(d['y_scale'])
    # The affine map from the model's own pooled-context axis onto the harness
    # axis, reconstructed from the two scale pairs the dump records.
    factor = float(d['causalpfn_y_scale']) / y_scale
    offset = (float(d['causalpfn_y_shift']) - y_shift) / y_scale

    names = [str(x) for x in np.atleast_1d(d['causalpfn_methods'])]
    kinds = [str(x) for x in np.atleast_1d(d['causalpfn_kinds'])]
    out = {}
    for name, kind in zip(names, kinds):
        if kind == 'joint':
            out[name] = dict(kind='joint', logits=d[f'{name}_logits'],
                             J=int(d[f'{name}_J']),
                             edges2d_native=d[f'{name}_edges2d_native'],
                             factor=factor, offset=offset)
        else:
            # edges1d is already on the harness axis; the native copy is kept
            # in the dump for provenance only.
            out[name] = dict(kind='1d', pred0=d[f'{name}_pred0'],
                             pred1=d[f'{name}_pred1'],
                             edges=d[f'{name}_edges1d'])
    return out


# ---------------------------------------------------------------------------
def run_query(q):
    """One query, every discovered method. Returns plain dicts (picklable)."""
    d = _G['d']
    METHODS = _G['methods']
    tau_star = float(d['tau_star_scaled'][q])
    t_star = np.array([tau_star])
    p_true = truth_tau_density(float(d['mu0_scaled'][q]),
                               float(d['mu1_scaled'][q]),
                               float(d['sigma_scaled']), TAU_CENTERS)
    bundles = build_query(q)
    out = {}
    for name in METHODS:
        b = bundles[name]
        # Seed on (realization, query, method): reproducible, and independent
        # across methods so a bad draw cannot favour one of them. Same recipe
        # as the J arm, so UWYK rows draw the same jitter in both arms.
        seed = (MALC_SEED + _G['r'] * 1_000_003 + q * 1009
                + seed_slot(name) * 31) % (2 ** 31 - 1)

        rec = dict(fallback=0, reason='', tau_hull_lo=float('nan'),
                   tau_hull_hi=float('nan'), p_star=float('nan'),
                   interior_star=float('nan'))
        p_grid = d_star = cate = None

        density_at, hull_lo, hull_hi, e_smooth, e_raw, ok = smooth_interior(
            b['S'], b['bw'], b['w0'], seed)
        rec['tau_hull_lo'], rec['tau_hull_hi'] = hull_lo, hull_hi

        if not ok:
            rec['fallback'], rec['reason'] = 1, 'fit_failed'
        else:
            # THE INTERIOR TERM ALONE decides the fallback, not the total.
            # Adding the 8 tail regions makes the total strictly positive
            # everywhere, so a `d_star <= 0` test almost never fires -- what
            # actually happens when tau* misses the hull is that the interior
            # contributes NOTHING and the density comes back as pure tail:
            # finite, so nothing raises, and worth several nats. Trigger on
            # the interior instead, which is exactly the hull tau-range.
            interior_star = float(density_at(t_star)[0])
            rec['interior_star'] = interior_star
            d_star = interior_star + float(b['tail'](t_star)[0])
            rec['p_star'] = d_star
            if interior_star <= 0.0:
                rec['fallback'], rec['reason'] = 1, 'tau_outside_hull'
            elif not np.isfinite(d_star) or d_star <= 0.0:
                rec['fallback'], rec['reason'] = 1, 'zero_density'
            if rec['fallback']:
                d_star = None
            else:
                p_grid = density_at(TAU_CENTERS) + b['tail'](TAU_CENTERS)
                # E[tau] splits by region as w0*m*E_S + (tail moment), and T
                # changes only the first term -- so the correction is exact
                # and never touches the 8 tail regions.
                m = float(np.asarray(b['S']).sum())
                cate = float(b['raw_mean']()
                             + b['w0'] * m * (e_smooth - e_raw))

        if p_grid is None:                      # raw fallback for this query
            p_grid = b['raw'](TAU_CENTERS)
            d_star = float(np.atleast_1d(b['raw'](t_star))[0])
            cate = float(b['raw_mean']())

        out[name] = dict(score=score(p_grid, p_true, d_star), cate=cate, **rec)
    return q, out


def _init(payload):
    _G.update(payload)


# ---------------------------------------------------------------------------
def evaluate(path, pool_cls):
    with np.load(path, allow_pickle=True) as z:
        d = {k: z[k] for k in z.files}
    r = int(d['realization'])

    has = {'uwyk': 'uwyk_pred0' in d}
    dopfn, causalpfn = dopfn_specs(d), causalpfn_specs(d)
    has['dopfn'], has['causalpfn'] = bool(dopfn), bool(causalpfn)
    want = lambda fam: MODEL_FAMILY in (fam, 'all', 'auto')
    if MODEL_FAMILY in has and not has[MODEL_FAMILY]:
        raise SystemExit(f'[tauC-malcT] {path}: MODEL_FAMILY={MODEL_FAMILY} but '
                         f'the dump carries no {MODEL_FAMILY} predictions')
    use_uwyk = has['uwyk'] and want('uwyk')
    if not want('dopfn'):
        dopfn = {}
    if not want('causalpfn'):
        causalpfn = {}

    methods = ((['uwyk_native', 'uwyk_matched', 'joint'] if use_uwyk else [])
               + list(dopfn) + list(causalpfn))
    if not methods:
        raise SystemExit(f'[tauC-malcT] {path}: no scoreable methods')

    payload = dict(d=d, r=r, methods=methods, uwyk=use_uwyk, dopfn=dopfn,
                   causalpfn=causalpfn)
    if use_uwyk:
        payload.update(J=int(d['J']), e2=d['edges2d'],
                       be=d['bar_edges'], bw=d['bar_widths'],
                       sL=float(d['base_sL']), sR=float(d['base_sR']))
    n_q = len(d['tau_star_scaled'])

    if N_WORKERS > 1:
        with pool_cls(processes=min(N_WORKERS, n_q), initializer=_init,
                      initargs=(payload,)) as pool:
            results = pool.map(run_query, range(n_q), chunksize=1)
    else:
        _init(payload)
        results = [run_query(q) for q in range(n_q)]
    results.sort(key=lambda t: t[0])

    true_cate = np.asarray(d['true_cate'], dtype=np.float64).reshape(-1)
    y_scale = float(d['y_scale'])
    row = {'dataset': str(d['dataset']), 'realization': r, 'n_queries': n_q,
           'n_context': int(d['n_context']), 'anc_tag': str(d['anc_tag']),
           'truth_noise_source': str(d['truth_noise_source']),
           'sigma_raw': float(d['sigma_raw']),
           'sigma_scaled': float(d['sigma_scaled']),
           'sigma_residual_raw': float(d['sigma_residual_raw']),
           'sigma_residual_scaled': float(d['sigma_residual_scaled']),
           'y_scale': y_scale, 'y_shift': float(d['y_shift']),
           'y_scaling': str(d['y_scaling']), 'std_target': str(d['std_target']),
           'true_cate': true_cate,
           'malc_variant': 'T', 'malc_B': MALC_B, 'malc_K': MALC_K,
           'malc_seed': MALC_SEED, 'malc_n_tau': MALC_N_TAU,
           'n_y0': N_Y0, 'source_dump': str(path),
           'tau_star_scaled': np.asarray(d['tau_star_scaled']),
           'frac_tau_outside_grid': float(
               np.mean(np.abs(d['tau_star_scaled']) > 3.0))}

    for name in methods:
        recs = [res[name] for _, res in results]
        for k in recs[0]['score']:
            row[f'{k}_{name}'] = float(np.mean([x['score'][k] for x in recs]))
        means = [x['cate'] for x in recs]
        for metric, value in point_metrics(means, true_cate, y_scale).items():
            row[f'{metric}_{name}'] = value
        row[f'cate_pred_{name}'] = np.asarray(means) * y_scale
        # Per-query MALC diagnostics. The J arm also dumps the (B, 2) synthetic
        # points; variant T goes through tau_smoother, which does not expose
        # the 1D sample, so the fallback QUERIES are recorded instead of their
        # points. Nothing in summarize_density_tauC.py reads either.
        row[f'malc_tau_hull_lo_{name}'] = np.array(
            [x['tau_hull_lo'] for x in recs])
        row[f'malc_tau_hull_hi_{name}'] = np.array(
            [x['tau_hull_hi'] for x in recs])
        row[f'malc_p_star_{name}'] = np.array([x['p_star'] for x in recs])
        row[f'malc_interior_star_{name}'] = np.array(
            [x['interior_star'] for x in recs])
        row[f'malc_fallback_{name}'] = np.array(
            [x['fallback'] for x in recs], dtype=np.int8)
        row[f'malc_fallback_reason_{name}'] = np.array(
            [x['reason'] for x in recs])
        row[f'n_fallback_{name}'] = int(sum(x['fallback'] for x in recs))
        row[f'malc_fallback_query_{name}'] = np.array(
            [q for q, x in enumerate(recs) if x['fallback']], dtype=np.int32)
    row['methods'] = np.asarray(methods)
    return row, methods


def main():
    os.makedirs(OUT, exist_ok=True)
    paths = sorted(p for p in os.listdir(DUMPS) if p.endswith('.npz'))
    if not paths:
        raise SystemExit(f'[tauC-malcT] no dumps in {DUMPS}')
    dataset = paths[0].rsplit('_r', 1)[0]
    lo, hi = max(0, REAL_START), (len(paths) if REAL_END is None
                                  else min(len(paths), int(REAL_END)))
    print(f'[tauC-malcT] variant=T dataset={dataset} dumps={DUMPS} '
          f'family={MODEL_FAMILY} B={MALC_B} K={MALC_K} n_tau={MALC_N_TAU} '
          f'tail_n_y0={N_Y0} dopfn_bins={DOPFN_MALC_BINS} '
          f'workers={N_WORKERS}  realizations [{lo}, {hi}) of {len(paths)}',
          flush=True)

    # 'fork' shares the dump arrays copy-on-write; 'spawn' would repickle the
    # whole npz to every worker for every realization.
    pool_cls = mp.get_context('fork').Pool

    t0 = time.time()
    for path in paths[lo:hi]:
        row, methods = evaluate(os.path.join(DUMPS, path), pool_cls)
        r = int(row['realization'])
        np.savez(os.path.join(OUT, f'{dataset}_r{r:03d}.npz'),
                 **{k: np.array(v) for k, v in row.items()})
        print(f'r={r:03d}  ' + '  |  '.join(
            f'{m}: nll={row[f"nll_{m}"]:7.3f} l2={row[f"l2_{m}"]:6.3f} '
            f'klrev={row[f"kl_rev_{m}"]:7.4f} pehe={row[f"pehe_{m}"]:7.3f} '
            f'fb={row[f"n_fallback_{m}"]:d}' for m in methods)
            + f'   ({time.time()-t0:.0f}s)', flush=True)
    print(f'[tauC-malcT] done in {time.time()-t0:.0f}s -> {OUT}', flush=True)


if __name__ == '__main__':
    main()
