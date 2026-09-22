#!/usr/bin/env python
"""One figure per FOR_FURKAN.md table block: CATE density vs ATE density.

Each *category* here is one block of a live table in FOR_FURKAN.md -- the rows
between two rules, e.g.

    | new (raw)  | DoPFN (x)indep native        | 0.2115+-0.0343 | 1.3222+-0.1080 |
    | new (raw)  | DoPFN repro_1d_J10 (x)indep  | 0.6903+-0.0540 | 1.4401+-0.1024 |
    | new (raw)  | DoPFN repro_joint2d Joint-2D | 0.4802+-0.0449 | 1.4690+-0.1099 |

and every method in the block goes on one figure, against the generator truth.

Two panels, both densities over the same tau:

    left   p(tau | x_q*)   the CATE density of ONE individual -- within the
                           shared realization, the query the category's joint
                           model fits best (argmin of -log p_joint(tau*)).
                           A deliberate cherry-pick; the title says so.
    right  p_ATE(tau)      the ATE density, defined as the query-average of the
                           per-individual CATE densities within realization r*:
                               p_ATE(tau) = (1/n) sum_q p(tau | x_q)
                           truth the same way, from the truth densities. Same
                           object realcause_eval/eval_ate_density_metrics.py
                           calls the truth p_ATE, so it carries the per-query
                           width scale, NOT the much narrower N(ATE, sigma^2/n).

The realization is picked FIRST and is shared by every figure of a dataset, so
the panels all describe the same sample. "Average best performing" means lowest
mean nll across the drawn methods of every block of that dataset, each (block,
method) series standardized across realizations before averaging -- otherwise
the mean just tracks whichever method has the widest nll spread. IHDP and ACIC
necessarily get different realizations; --realization overrides.

--grouped builds a second kind of figure: several blocks' CATE panels on one
sheet (see --list), for reading the blocks side by side.

Beneath them, the block's own numbers -- nll, l2 (density tables) and sqrt PEHE,
CATE L1, ATE abs error (point tables), mean +- SE over realizations, recomputed
from the scored NPZs. They reproduce FOR_FURKAN.md to all printed digits;
--check-table prints them next to the plot so that stays verifiable.

Where a block lists more than one joint (2D) model, only the best-scoring one is
drawn (lowest mean nll) -- `--all-joints` keeps them all. This only bites on the
CausalPFN blocks, which carry two 2D heads.

Reads ONLY the prediction dumps under <run>/<dataset>/predictions, exactly as
plot_density_tauC.py does: no checkpoint and no GPU, because the dump carries
the raw head outputs plus every axis and truth parameter. Verified against the
scored NPZ -- rebuilding a dump reproduces nll_* to all printed digits.

MALC categories are NOT here. The MALC runs (results_density_tauC_malc/5344224,
/5564597) ship scored metrics only, no predictions/, so their densities cannot
be rebuilt from what is on disk. --list marks them.

Usage:
    python benchmarks/eval_graph2d/plot_category_densities.py --list
    python benchmarks/eval_graph2d/plot_category_densities.py          # all
    python benchmarks/eval_graph2d/plot_category_densities.py \
        --category dopfn-ihdp-raw --check-table
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from density_common import (                                       # noqa: E402
    CausalPFN1D, DoPFN1D, Joint2D, UWYK1D,
    causalpfn_tau_density, dopfn_tau_density, joint_tau_density,
    truth_tau_density, uwyk_tau_density, TAU_CENTERS,
)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# dataviz reference palette, categorical slots 1-4, assigned in fixed table
# order and never cycled. Truth is not a series -- it is the reference the
# series are read against -- so it wears text ink, not a hue. Slots 3 and 4 sit
# below 3:1 on the light surface, so the relief rule applies: the metrics table
# under the panels names every method, and the legend is always present.
SLOTS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4')
# Linestyle as secondary encoding, so the figure survives print and CVD.
DASHES = ('-', '-', (0, (5, 2)), (0, (1.5, 1.5)), (0, (7, 2, 1.5, 2)))
INK = '#0b0b0b'
INK_2 = '#52514e'
INK_MUTED = '#8a8985'
SURFACE = '#fcfcfb'
GRID = '#eceae5'
AXIS = '#d9d8d3'

METRIC_COLUMNS = (('nll', 'nll'), ('l2', 'l2'), ('pehe', 'sqrt PEHE'),
                  ('cate_l1', 'CATE L1'), ('ate_abs_err', 'ATE abs err'))


@dataclass(frozen=True)
class Category:
    """One block of a live FOR_FURKAN.md table."""
    key: str
    title: str                     # the block, as the document names it
    run: str                       # holds <dataset>/ and <dataset>/predictions/
    dataset: str
    methods: tuple                 # scored-NPZ names, in the table's row order
    labels: dict                   # method -> the label the table prints
    note: str = ''

    @property
    def scored_dir(self):
        return os.path.join(REPO, self.run, self.dataset)

    @property
    def pred_dir(self):
        return os.path.join(self.scored_dir, 'predictions')


# Run <-> table-block mapping, established by matching each run's aggregated
# nll/l2 against the document (every value agrees to all printed digits):
#   5312884 anc=noanc | 5312882 anc=v3a | 5312883 anc=v3b   -> UWYK raw blocks
#   60508900                                                -> DoPFN "new (raw)"
#   5571187                                                 -> CausalPFN
_UWYK_NOANC = {'uwyk_native': 'UWYK (x)indep K=1000', 'joint': 'UWYK Joint-2D'}
_UWYK_V3 = {'uwyk_native': 'UWYK (x)indep K=1000', 'joint': 'Joint-2D J=32'}
# uwyk_matched is scored but the live tables drop it -- it is the bridge
# contrast's control, not a model -- so it is not a row of any category here.
# The DoPFN tables carry THREE blocks, not one: new (raw), new (MALC), and a
# raw-vs-MALC pair for repro_1d_J100 pulled out on its own. A block is a
# category, so repro_1d_J100 is not a row of the new (raw) figure.
_DOPFN_RAW = {'dopfn_native': 'DoPFN (x)indep native',
              'dopfn_repro_1d_J10': 'DoPFN repro_1d_J10 (x)indep',
              'dopfn_repro_joint2d': 'DoPFN repro_joint2d Joint-2D'}
_DOPFN_J100 = {'dopfn_repro_1d_J100': 'DoPFN repro_1d_J100 (x)indep'}
_CPFN = {'causalpfn_j32_random_2d': 'CausalPFN j32_random_2d Joint-2D',
         'causalpfn_j32_eta0_y01_2d': 'CausalPFN j32_eta0_y01_2d Joint-2D',
         'causalpfn_j1024_headrand_1d': 'CausalPFN j1024_headrand_1d (x)indep',
         'causalpfn_j32_1d': 'CausalPFN j32_1d (x)indep',
         'causalpfn_botharms_1d': 'CausalPFN botharms_1d (x)indep'}

# The J100 block pairs one raw row against one MALC row. MALC has no dump, so
# the drawn block is the raw row alone -- against the truth, which is still the
# comparison the panels make.
_J100_NOTE = ('this block pairs raw against MALC; the MALC row has no '
              'predictions dump, so only the raw row is drawn')

CATEGORIES = tuple(
    [Category(f'uwyk-{ds.lower()}-{tag}', f'UWYK - {ds}, raw . {tag}',
              f'results_density_tauC/{run}', ds, ('uwyk_native', 'joint'), lab)
     for ds in ('IHDP', 'ACIC')
     for run, tag, lab in (('5312884', 'noanc', _UWYK_NOANC),
                           ('5312882', 'v3a', _UWYK_V3),
                           ('5312883', 'v3b', _UWYK_V3))]
    + [Category(f'dopfn-{ds.lower()}-raw', f'DoPFN - {ds}, new (raw)',
                'results_density_tauC/60508900', ds, tuple(_DOPFN_RAW), _DOPFN_RAW)
       for ds in ('IHDP', 'ACIC')]
    + [Category(f'dopfn-{ds.lower()}-j100', f'DoPFN - {ds}, repro_1d_J100',
                'results_density_tauC/60508900', ds, tuple(_DOPFN_J100),
                _DOPFN_J100, _J100_NOTE)
       for ds in ('IHDP', 'ACIC')]
    + [Category(f'causalpfn-{ds.lower()}', f'CausalPFN - {ds}',
                'results_density_tauC/5571187', ds, tuple(_CPFN), _CPFN)
       for ds in ('IHDP', 'ACIC')]
)

# ---- grouped figures: several blocks' CATE panels on one sheet -------------
# A panel is (category key, methods to draw or None for the category's own).
# A subset is kept in the block's table order, and '@joint' stands for whichever
# 2D head that block's best-joint rule picks -- so one recipe covers both
# datasets even though they disagree on which CausalPFN j32 2D head wins.
@dataclass(frozen=True)
class Grouped:
    key: str
    title: str
    dataset: str
    panels: tuple
    ncol: int = 3
    xlim: tuple = None             # plotted units; None -> fit to the curves
    query: int = None              # pin the individual; None -> choose_query


GROUPED = (
    Grouped('ihdp_grouped_cate', 'IHDP - CATE densities, one panel per block',
            'IHDP',
            (('uwyk-ihdp-noanc', None),
             ('uwyk-ihdp-v3a', None),
             ('uwyk-ihdp-v3b', None),
             ('causalpfn-ihdp', ('@joint',
                                 'causalpfn_j1024_headrand_1d',
                                 'causalpfn_j32_1d')),
             ('dopfn-ihdp-raw', None)), xlim=(-10.0, 10.0), query=46),
    Grouped('acic_grouped_cate', 'ACIC - CATE densities, one panel per block',
            'ACIC',
            (('uwyk-acic-noanc', None),
             ('uwyk-acic-v3a', None),
             ('uwyk-acic-v3b', None),
             ('causalpfn-acic', ('@joint',
                                 'causalpfn_j1024_headrand_1d',
                                 'causalpfn_j32_1d')),
             ('dopfn-acic-raw', None)), xlim=(-10.0, 10.0)),
)


def resolve_subset(cat, subset, all_joints=False):
    """Subset -> the block's methods to draw, in its table order."""
    own, kinds, _ = drawn_methods(cat, all_joints)
    if not subset:
        return own
    want = set(subset)
    if '@joint' in want:
        want.discard('@joint')
        want |= {m for m in own if kinds[m] == 'joint'}
    return [m for m in cat.methods if m in want]

# Blocks that exist in the document but cannot be drawn, and why. Printed by
# --list so a missing figure is never mistaken for an oversight.
UNAVAILABLE = (
    ('UWYK - IHDP/ACIC, MALC . noanc', 'results_density_tauC_malc/5344224',
     'no predictions/ dump'),
    ('DoPFN - IHDP/ACIC, new (MALC)', 'results_density_tauC_malc/5564597',
     'no predictions/ dump'),
    ('DoPFN - IHDP/ACIC, repro_1d_J100 MALC row',
     'results_density_tauC_malc/5564597',
     'no predictions/ dump -- its raw partner IS drawn, alone'),
)


# ---------------------------------------------------------------- rebuilding

def realizations(cat):
    """Realization indices that have BOTH a scored NPZ and a dump."""
    def idx(pattern):
        return {int(os.path.basename(f)[len(cat.dataset) + 2:-4])
                for f in glob.glob(pattern)}
    have = idx(os.path.join(cat.scored_dir, f'{cat.dataset}_r*.npz')) & \
        idx(os.path.join(cat.pred_dir, f'{cat.dataset}_r*.npz'))
    return sorted(have)


def load_dump(cat, r):
    return np.load(os.path.join(cat.pred_dir, f'{cat.dataset}_r{r:03d}.npz'),
                   allow_pickle=True)


def density_builder(dump, method, n_y0):
    """-> (kind, fn(q, tau) -> density on the scaled tau axis).

    The kind and every axis parameter are read from the dump's own keys, never
    from the method name, so a renamed checkpoint cannot silently pick up the
    wrong reconstruction. Each branch mirrors the producer exactly:
    density_dopfn.DoPFNModelSet.predict / density_causalpfn.CausalPFNModelSet
    .predict for the PFN rows, eval_density_tauC's UWYK block for the rest.
    """
    keys = set(dump.files)
    ys, yc = float(dump['y_shift']), float(dump['y_scale'])

    if method == 'joint':                       # UWYK's 2D head
        jt = [Joint2D.from_pred(p, int(dump['J']), dump['edges2d'])
              for p in dump['joint_logits']]
        return 'joint', lambda q, tau: joint_tau_density(jt[q], tau, n_y0=n_y0)

    if method in ('uwyk_native', 'uwyk_matched'):
        be, bw = dump['bar_edges'], dump['bar_widths']
        sL, sR = float(dump['base_sL']), float(dump['base_sR'])
        arms = [[UWYK1D.from_pred(p, be, bw, sL, sR) for p in dump[f'uwyk_pred{a}']]
                for a in (0, 1)]
        if method == 'uwyk_matched':            # re-binned to the 2D head's J
            arms = [[f.rebin(dump['edges2d']) for f in arm] for arm in arms]
        return '1d', lambda q, tau: uwyk_tau_density(arms[0][q], arms[1][q], tau,
                                                     n_y0=n_y0)

    if method == 'dopfn_native':                # the library DoPFNRegressor
        arms = [[DoPFN1D.from_pred(p, dump[f'dopfn_borders{a}_raw'],
                                   y_shift=ys, y_scale=yc,
                                   tail_scales=dump[f'dopfn_tail_scales{a}_raw'])
                 for p in dump[f'dopfn_pred{a}']] for a in (0, 1)]
        return '1d', lambda q, tau: dopfn_tau_density(arms[0][q], arms[1][q], tau)

    if method == 'dopfn_joint':                 # training_dopfn_base bb_joint
        jt = [Joint2D.from_pred(p, int(dump['dopfn_J']), dump['dopfn_edges2d'])
              for p in dump['dopfn_joint_logits']]
        return 'joint', lambda q, tau: joint_tau_density(jt[q], tau, n_y0=n_y0)

    if f'{method}_logits' in keys:              # any named J x J head
        jt = [Joint2D.from_pred(p, int(dump[f'{method}_J']), dump[f'{method}_edges2d'])
              for p in dump[f'{method}_logits']]
        return 'joint', lambda q, tau: joint_tau_density(jt[q], tau, n_y0=n_y0)

    if f'{method}_pred0' in keys:               # any named 1D pair of arms
        if f'{method}_borders_raw' in keys:     # training_dopfn_repro dopfn_1d
            borders = dump[f'{method}_borders_raw']
            arms = [[DoPFN1D.from_pred(p, borders, y_shift=ys, y_scale=yc)
                     for p in dump[f'{method}_pred{a}']] for a in (0, 1)]
            return '1d', lambda q, tau: dopfn_tau_density(arms[0][q], arms[1][q], tau)
        if f'{method}_edges1d' in keys:         # a CausalPFN 1D head
            edges = dump[f'{method}_edges1d']
            arms = [[CausalPFN1D.from_pred(p, edges)
                     for p in dump[f'{method}_pred{a}']] for a in (0, 1)]
            return '1d', lambda q, tau: causalpfn_tau_density(arms[0][q], arms[1][q],
                                                              tau)
    raise KeyError(f'{method}: no reconstruction in this dump '
                   f'(keys: {sorted(k for k in keys if method in k)})')


def truth_builder(dump):
    mu0, mu1 = dump['mu0_scaled'], dump['mu1_scaled']
    sigma = float(dump['sigma_scaled'])
    return lambda q, tau: truth_tau_density(mu0[q], mu1[q], sigma, tau)


# ------------------------------------------------------------------ metrics

_PER_REALIZATION = {}


def per_realization(cat):
    """{method: {metric: {r: value}}} straight off the scored NPZs.

    These are the realization-level numbers the document averages; reading them
    here costs nothing and needs no density rebuild.
    """
    if cat.key not in _PER_REALIZATION:
        out = {m: {key: {} for key, _ in METRIC_COLUMNS} for m in cat.methods}
        for r in realizations(cat):
            z = np.load(os.path.join(cat.scored_dir, f'{cat.dataset}_r{r:03d}.npz'))
            for m in cat.methods:
                for key, _ in METRIC_COLUMNS:
                    if f'{key}_{m}' in z.files:
                        out[m][key][r] = float(z[f'{key}_{m}'])
        _PER_REALIZATION[cat.key] = out
    return _PER_REALIZATION[cat.key]


def table_metrics(cat):
    """{method: {metric: (mean, se)}}, aggregated exactly as the document does."""
    book = per_realization(cat)
    out = {}
    for m in cat.methods:
        cells = {}
        for key, _ in METRIC_COLUMNS:
            v = np.asarray(list(book[m][key].values()))
            if v.size:
                cells[key] = (v.mean(), v.std(ddof=1) / np.sqrt(v.size))
        out[m] = cells
    return out


def keep_best_joint(cat, metrics, kinds):
    """Drop every joint but the lowest-nll one, as the answer to `--all-joints`."""
    joints = [m for m in cat.methods if kinds.get(m) == 'joint']
    if len(joints) < 2:
        return list(cat.methods), None
    best = min(joints, key=lambda m: metrics[m]['nll'][0])
    return [m for m in cat.methods if kinds.get(m) != 'joint' or m == best], best


# --------------------------------------------------------------- selection

def drawn_methods(cat, all_joints=False):
    """The methods a figure of this category actually draws."""
    probe = load_dump(cat, realizations(cat)[0])
    kinds = {m: density_builder(probe, m, int(probe['n_y0']))[0] for m in cat.methods}
    if all_joints:
        return list(cat.methods), kinds, None
    keep, dropped = keep_best_joint(cat, table_metrics(cat), kinds)
    return keep, kinds, dropped


def choose_realization(cats, scope_cats, verbose=True):
    """The realization every figure of this dataset should use.

    "Average best performing" = lowest MEAN nll across the drawn methods, where
    each (category, method) series is standardized across realizations before
    averaging. The standardizing matters: raw nll runs from about -0.6
    (CausalPFN) to +0.7 (repro_1d_J10), so an unstandardized mean would just
    track whichever method has the widest spread instead of asking, of each
    model, "was this realization an easy one for you?".

    Ranked over the realizations every category in `scope_cats` has, so the
    answer does not depend on which subset of figures is being rendered.
    """
    common = sorted(set.intersection(*(set(realizations(c)) for c in scope_cats)))
    if not common:
        raise SystemExit('no realization is shared by every category of this '
                         'dataset; pass --realization <n> explicitly')
    z_total = np.zeros(len(common))
    n_series = 0
    for c in scope_cats:
        book = per_realization(c)
        for m in drawn_methods(c)[0]:
            v = np.asarray([book[m]['nll'][r] for r in common])
            sd = v.std(ddof=1)
            if sd <= 0:
                continue
            z_total += (v - v.mean()) / sd
            n_series += 1
    z = z_total / max(n_series, 1)
    order = np.argsort(z)
    best = common[order[0]]
    if verbose:
        runners = ', '.join(f'r{common[i]:03d} {z[i]:+.3f}' for i in order[1:4])
        print(f'  [realization] {scope_cats[0].dataset}: r{best:03d} '
              f'(mean z of nll {z[order[0]]:+.3f} over {n_series} model series, '
              f'{len(common)} shared realizations)   next: {runners}')
    return best, float(z[order[0]]), n_series, len(common)


def pick_individual(cat, method, r, how='best', verbose=True):
    """(q, nll) for the individual this method fits best / worst / median.

    Scored on -log p(tau* | x_q) -- the per-query form of the table's own nll
    column -- within the ONE realization every figure shares.
    """
    t0 = time.time()
    dump = load_dump(cat, r)
    _, fn = density_builder(dump, method, int(dump['n_y0']))
    tau_star = dump['tau_star_scaled']
    scores = []
    for q in range(len(tau_star)):
        d = fn(q, np.array([tau_star[q]]))[0]
        scores.append((float(-np.log(d)) if d > 0 else np.inf, r, q))
    scores.sort()
    scores = [s for s in scores if np.isfinite(s[0])] or scores
    pick = {'best': scores[0], 'worst': scores[-1],
            'median': scores[len(scores) // 2]}[how]
    if verbose:
        print(f'    [select] {how} individual for {method}: r{pick[1]:03d} '
              f'q={pick[2]}  -log p(tau*)={pick[0]:+.4f}   '
              f'({len(scores)} individuals scanned, {time.time() - t0:.1f}s)')
    return pick[1], pick[2], pick[0]


# ------------------------------------------------------------------ figure

_TRAPZ = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz


def ate_density(fn, n_q, tau, stride=1):
    """p_ATE(tau) = mean over the realization's queries of p(tau | x_q)."""
    qs = list(range(0, n_q, stride))
    acc = np.zeros_like(tau)
    for q in qs:
        acc += fn(q, tau)
    return acc / len(qs)


def _style_axes(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(labelsize=8, colors=INK_2)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)


def _window(xs, curves, q=0.04, pad_frac=0.08, truth=None, cap=6.0):
    """x-limits holding every curve's central 1-2q mass.

    A fraction-of-peak threshold is no good here: DoPFN's half-normal tails and
    the joint heads' 8 tail regions stay above any sane floor out to the edge of
    the tau grid, which squashes every bulk into the middle pixel. Quantiles of
    each curve's own CDF track the bulk instead, and the union keeps the widest
    method fully in frame.
    """
    lo, hi = np.inf, -np.inf
    for c in curves:
        cdf = np.cumsum(c)
        if cdf[-1] <= 0:
            continue
        cdf /= cdf[-1]
        lo = min(lo, xs[np.searchsorted(cdf, q)])
        hi = max(hi, xs[min(np.searchsorted(cdf, 1.0 - q), xs.size - 1)])
    # One heavy-tailed method can otherwise set the frame for everybody and
    # squeeze every bulk into the middle. Clamp to a multiple of the truth's
    # own width: a curve that runs past the frame still reads as running past.
    if truth is not None and cap:
        cdf = np.cumsum(truth)
        if cdf[-1] > 0:
            cdf /= cdf[-1]
            t_lo = xs[np.searchsorted(cdf, 0.005)]
            t_hi = xs[min(np.searchsorted(cdf, 0.995), xs.size - 1)]
            mid, half = 0.5 * (t_lo + t_hi), 0.5 * (t_hi - t_lo)
            lo, hi = max(lo, mid - cap * half), min(hi, mid + cap * half)
    m = pad_frac * (hi - lo)
    return lo - m, hi + m


def build_figure(cat, args, r_chosen, r_note=''):
    import matplotlib.pyplot as plt

    rs = realizations(cat)
    if not rs:
        raise SystemExit(f'{cat.key}: no realization has both a scored NPZ and '
                         f'a dump under {cat.pred_dir}')
    probe = load_dump(cat, rs[0])
    n_y0 = args.n_y0 or int(probe['n_y0'])
    kinds = {m: density_builder(probe, m, n_y0)[0] for m in cat.methods}
    metrics = table_metrics(cat)
    methods, dropped_for = ((list(cat.methods), None) if args.all_joints
                            else keep_best_joint(cat, metrics, kinds))
    joints = [m for m in methods if kinds[m] == 'joint']
    anchor = args.anchor or (joints[0] if joints else methods[0])

    if r_chosen not in rs:
        raise SystemExit(f'{cat.key}: realization r{r_chosen:03d} has no dump '
                         f'here (available: {rs[0]}..{rs[-1]})')
    r_star, q_star, nll_star = pick_individual(cat, anchor, r_chosen, args.select)
    dump = load_dump(cat, r_star)
    tau = np.asarray(dump['tau_grid']) if 'tau_grid' in dump.files else TAU_CENTERS
    y_scale = float(dump['y_scale'])
    n_q = len(dump['tau_star_scaled'])

    # raw <- scaled: tau_raw = tau * y_scale, and p_raw = p_scaled / y_scale so
    # the curve stays a density (area 1) after the change of variable.
    xs, dscale = (tau, 1.0) if args.scaled else (tau * y_scale, 1.0 / y_scale)
    unit = 'scaled' if args.scaled else 'raw'

    truth_fn = truth_builder(dump)
    cate = {'truth': truth_fn(q_star, tau)}
    n_avg = len(range(0, n_q, args.ate_stride))
    ate = {'truth': ate_density(truth_fn, n_q, tau, args.ate_stride)}
    per_individual = {}
    for m in methods:
        t0 = time.time()
        _, fn = density_builder(dump, m, n_y0)
        cate[m] = fn(q_star, tau)
        ate[m] = ate_density(fn, n_q, tau, args.ate_stride)
        d = fn(q_star, np.array([dump['tau_star_scaled'][q_star]]))[0]
        per_individual[m] = float(-np.log(d)) if d > 0 else np.inf
        print(f'    [build] {m:<28s} {time.time() - t0:5.1f}s')

    # ATE error of the plotted curve, in raw units: the density's own mean
    # against this realization's true ATE. Companion to the table's aggregate.
    true_ate = float(np.mean(dump['true_cate']))
    ate_mean = {m: float(_TRAPZ(tau * ate[m], tau)) * y_scale for m in methods}

    fig = plt.figure(figsize=(13.4, 7.6))
    fig.patch.set_facecolor(SURFACE)
    # Explicit margins, not tight_layout: the metrics axes is a text block with
    # no data extent, which tight_layout cannot measure.
    gs = fig.add_gridspec(2, 2, height_ratios=(1.0, 0.36), hspace=0.30,
                          wspace=0.16, left=0.055, right=0.985,
                          top=0.815, bottom=0.03)
    ax_c, ax_a = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    for ax, book, title in (
            (ax_c, cate, f'CATE   $p(\\tau \\mid x_q)$   individual r{r_star:03d} q={q_star}'),
            (ax_a, ate, f'ATE   $p_{{ATE}}(\\tau)$ = mean over {n_avg} individuals, r{r_star:03d}')):
        ax.plot(xs, book['truth'] * dscale, color=INK, lw=2.0, ls=(0, (5, 2)),
                label='truth', zorder=6)
        for i, m in enumerate(methods):
            ax.plot(xs, book[m] * dscale, color=SLOTS[i % len(SLOTS)], lw=2.0,
                    ls=DASHES[i % len(DASHES)], label=cat.labels[m],
                    zorder=5 - i * 0.1, solid_capstyle='round')
        ax.set_xlim(*_window(xs, list(book.values()), args.window_quantile,
                             truth=book['truth'], cap=args.window_cap))
        ax.set_title(title, fontsize=10, color=INK, pad=8)
        ax.set_xlabel(rf'$\tau$ ({unit} units)', fontsize=8.5, color=INK_2)
        _style_axes(ax)
    ax_c.set_ylabel(r'$p(\tau \mid x)$', fontsize=8.5, color=INK_2)
    ax_a.set_ylabel(r'$p_{ATE}(\tau)$', fontsize=8.5, color=INK_2)

    # tau* on the CATE panel, the realization's true ATE on the ATE panel
    for ax, v, lab in ((ax_c, float(dump['tau_star_scaled'][q_star]), r'$\tau^*$'),
                       (ax_a, true_ate / y_scale, 'true ATE')):
        x = v * (1.0 if args.scaled else y_scale)
        ax.axvline(x, color=INK_MUTED, lw=1.0, zorder=1)
        ax.annotate(lab, (x, ax.get_ylim()[1]), xytext=(4, -11),
                    textcoords='offset points', color=INK_MUTED, fontsize=8)

    # The table view -- also the relief the low-contrast hues oblige. Every row
    # the block lists appears, drawn or not, so the figure stands alone against
    # FOR_FURKAN.md. The last two columns are this figure's own numbers, not the
    # document's: the selected individual's nll and the plotted ATE curve's
    # mean error in realization r*.
    ax_t = fig.add_subplot(gs[1, :])
    ax_t.axis('off')
    w = max(len(cat.labels[m]) for m in cat.methods) + 2
    head = (f'{"method":<{w}s}' + ''.join(f'{t:>18s}' for _, t in METRIC_COLUMNS)
            + f'{"-log p(tau*)":>15s}{f"ATE err r{r_star:03d}":>15s}')
    lines = [head, '-' * len(head)]
    for m in cat.methods:
        cells = ''
        for key, _ in METRIC_COLUMNS:
            c = metrics[m].get(key)
            cells += f'{c[0]:>10.4f}+-{c[1]:<6.4f}' if c else f'{"--":>18s}'
        extra = (f'{per_individual[m]:>15.4f}{abs(ate_mean[m] - true_ate):>15.4f}'
                 if m in per_individual else f'{"not drawn":>15s}{"":>15s}')
        lines.append(f'{cat.labels[m]:<{w}s}{cells}{extra}')
    ax_t.text(0.0, 1.0, '\n'.join(lines), transform=ax_t.transAxes, va='top',
              ha='left', fontsize=7.2, family='monospace', color=INK_2)

    foot = [f'nll / l2 on the scaled tau axis, sqrt PEHE / CATE L1 / ATE abs '
            f'error in raw outcome units, mean +- SE over {len(rs)} realizations'
            f' -- the FOR_FURKAN.md values.  Axes hold each curve\'s '
            f'central {100 * (1 - 2 * args.window_quantile):.0f}%'
            + (f', clamped to {args.window_cap:g}x the truth\'s own width.'
               if args.window_cap else '.')]
    if dropped_for:
        foot.append(f'Joint rows other than {cat.labels[dropped_for]} are listed '
                    f'but not drawn (best joint by nll); --all-joints draws them.')
    if cat.note:
        foot.append(cat.note + '.')
    ax_t.text(0.0, 0.0, '\n'.join(foot), transform=ax_t.transAxes, va='bottom',
              ha='left', fontsize=7.4, color=INK_MUTED)

    handles, labels = ax_c.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=min(5, len(labels)),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.905))
    note = {'best': 'best-fit', 'worst': 'worst-fit', 'median': 'median'}[args.select]
    fig.suptitle(f'{cat.title}   --   CATE vs ATE densities', fontsize=13,
                 color=INK, y=0.978)
    fig.text(0.5, 0.932, f'realization r{r_star:03d}{r_note};  {note} individual '
             f'within it for {cat.labels[anchor]} -- a selected query, not a '
             f'typical one', ha='center', fontsize=9, color=INK_MUTED)
    return fig, dict(r=r_star, q=q_star, nll_star=nll_star, anchor=anchor,
                     metrics=metrics, methods=methods)


# --------------------------------------------------------------------- main

def query_scores(cat, args, r, methods):
    """{method: -log p(tau*|x_q) for every q} in one realization."""
    dump = load_dump(cat, r)
    n_y0 = args.n_y0 or int(dump['n_y0'])
    tau_star = dump['tau_star_scaled']
    out = {}
    for m in methods:
        _, fn = density_builder(dump, m, n_y0)
        v = np.empty(len(tau_star))
        for q in range(len(tau_star)):
            d = fn(q, np.array([tau_star[q]]))[0]
            v[q] = -np.log(d) if d > 0 else np.inf
        out[m] = v
    return out, np.asarray(dump['true_cate'], dtype=np.float64)


def choose_query(specs, args, r, verbose=True):
    """The one individual every panel of a grouped figure shows.

    Same standardize-then-average rule as the realization choice, but over
    queries: each (block, method) series of -log p(tau*) is standardized across
    the realization's queries, then averaged. Without the standardizing the
    mean would track whichever model has the widest per-query spread.

    Only meaningful because the blocks share a test split -- checked here
    against true_cate, not assumed, since the blocks come from different runs.
    """
    t0 = time.time()
    ref = ref_key = None
    n_series = 0
    acc = None
    for cat, methods in specs:
        book, true_cate = query_scores(cat, args, r, methods)
        if ref is None:
            ref, ref_key = true_cate, cat.key
            acc = np.zeros(true_cate.size)
        elif true_cate.shape != ref.shape or not np.allclose(true_cate, ref):
            raise SystemExit(
                f'{cat.key} and {ref_key} disagree on realization r{r:03d}\'s '
                f'test split, so a shared query index would not be the same '
                f'individual in both -- use --query per-block')
        for m, v in book.items():
            finite = np.isfinite(v)
            if not finite.any():
                continue
            w = np.where(finite, v, v[finite].max())   # penalise, do not break
            sd = w.std(ddof=1)
            if sd <= 0:
                continue
            acc += (w - w.mean()) / sd
            n_series += 1
    z = acc / max(n_series, 1)
    order = np.argsort(z)
    q = int(order[0] if args.select == 'best' else
            order[-1] if args.select == 'worst' else order[len(order) // 2])
    if verbose:
        runners = ', '.join(f'q={int(i)} {z[i]:+.3f}' for i in order[1:4])
        print(f'  [query] {specs[0][0].dataset} r{r:03d}: q={q} '
              f'(mean z of -log p(tau*) {z[q]:+.3f} over {n_series} model '
              f'series, {z.size} individuals, {time.time() - t0:.1f}s)'
              f'   next: {runners}')
    return q


def panel_curves(cat, args, r, methods, q):
    """One block's CATE panel: truth + each method at the shared individual."""
    dump = load_dump(cat, r)
    n_y0 = args.n_y0 or int(dump['n_y0'])
    tau = np.asarray(dump['tau_grid']) if 'tau_grid' in dump.files else TAU_CENTERS
    t_star = np.array([dump['tau_star_scaled'][q]])

    curves = {'truth': truth_builder(dump)(q, tau)}
    star = {}
    for m in methods:
        _, fn = density_builder(dump, m, n_y0)
        curves[m] = fn(q, tau)
        d = fn(q, t_star)[0]
        star[m] = float(-np.log(d)) if d > 0 else np.inf
    return dict(cat=cat, methods=methods, q=q, tau=tau,
                curves=curves, star=star, y_scale=float(dump['y_scale']),
                tau_star=float(dump['tau_star_scaled'][q]),
                true_cate=float(dump['true_cate'][q]))


def build_grouped(spec, args, r_chosen, r_note=''):
    """Several blocks' CATE panels on one sheet, plus one shared metrics table.

    Hues stay assigned by each block's own table order, so every panel matches
    the standalone figure it came from. That means a hue identifies a model
    only WITHIN a panel -- each panel therefore carries its own legend, and the
    table below names every curve again.
    """
    import matplotlib.pyplot as plt

    specs = [(next(c for c in CATEGORIES if c.key == key),
              resolve_subset(next(c for c in CATEGORIES if c.key == key),
                             subset, args.all_joints))
             for key, subset in spec.panels]

    if args.query not in ('shared', 'per-block'):
        q_shared = int(args.query)               # explicit CLI wins
    elif args.query == 'per-block':
        q_shared = None
    elif spec.query is not None:
        q_shared = spec.query
        print(f'  [query] {spec.dataset} r{r_chosen:03d}: q={q_shared} (pinned '
              f'by the figure; --query shared re-ranks)')
    else:
        q_shared = choose_query(specs, args, r_chosen)

    panels = []
    for cat, methods in specs:
        q = q_shared
        if q is None:                      # per-block: each block's own pick
            joints = [m for m in methods if drawn_methods(cat)[1][m] == 'joint']
            anchor = (args.anchor if args.anchor in methods
                      else (joints[0] if joints else methods[0]))
            _, q, _ = pick_individual(cat, anchor, r_chosen, args.select,
                                      verbose=False)
        panels.append(panel_curves(cat, args, r_chosen, methods, q))
        print(f'    [panel] {cat.key:<20s} q={panels[-1]["q"]:<4d} '
              f'{len(methods)} methods')

    # The metrics strip is sized from its actual line count -- one row per
    # drawn method, a blank line between blocks -- so it never runs off the
    # sheet or collides with the note beneath it.
    n_lines = 2 + sum(len(p['methods']) + 1 for p in panels)
    nrow = int(np.ceil(len(panels) / spec.ncol))
    table_in = 0.135 * n_lines + 0.5
    # Row height and hspace carry the per-panel legends, which sit BELOW each
    # axes rather than inside it -- nothing may overlap a curve.
    row_in = 4.6
    fig = plt.figure(figsize=(5.3 * spec.ncol, row_in * nrow + table_in + 1.1))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(nrow + 1, spec.ncol,
                          height_ratios=[1.0] * nrow + [table_in / row_in],
                          hspace=0.62, wspace=0.20, left=0.05, right=0.985,
                          top=0.885, bottom=0.055)

    # Every block of a dataset carries the same y_scale / y_shift / tau grid
    # (they are the harness's, not the model's), so the panels already live on
    # one axis and shared limits are a presentation choice, not a rescaling.
    fixed = tuple(args.xlim) if args.xlim else spec.xlim
    scales = {float(p['y_scale']) for p in panels}
    shared = (args.shared_axes or fixed) and len(scales) == 1
    if args.shared_axes and not shared:
        print(f'    [axes] blocks disagree on y_scale {sorted(scales)}; '
              f'falling back to per-panel limits')
    xlim = ylim = None
    if shared:
        y0 = panels[0]['y_scale']
        xs0, ds0 = ((panels[0]['tau'], 1.0) if args.scaled
                    else (panels[0]['tau'] * y0, 1.0 / y0))
        every = [c * ds0 for p in panels for c in p['curves'].values()]
        xlim = fixed or _window(xs0, every, args.window_quantile,
                                truth=panels[0]['curves']['truth'],
                                cap=args.window_cap)
        # Height from what is actually in frame -- a peak outside the window
        # must not flatten every panel.
        vis = (xs0 >= xlim[0]) & (xs0 <= xlim[1])
        top = max(c[vis].max() for c in every)
        ylim = (-0.02 * top, 1.08 * top)

    for i, p in enumerate(panels):
        ax = fig.add_subplot(gs[i // spec.ncol, i % spec.ncol])
        cat, y_scale = p['cat'], p['y_scale']
        xs, dscale = ((p['tau'], 1.0) if args.scaled
                      else (p['tau'] * y_scale, 1.0 / y_scale))
        ax.plot(xs, p['curves']['truth'] * dscale, color=INK, lw=1.9,
                ls=(0, (5, 2)), label='truth', zorder=6)
        # Slot by the block's own table order, not by draw order, so dropping a
        # row (the CausalPFN subset) does not repaint the survivors.
        for m in p['methods']:
            i_slot = list(cat.methods).index(m)
            ax.plot(xs, p['curves'][m] * dscale, color=SLOTS[i_slot % len(SLOTS)],
                    lw=1.9, ls=DASHES[i_slot % len(DASHES)],
                    label=cat.labels[m], zorder=5, solid_capstyle='round')
        if shared:
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
        else:
            ax.set_xlim(*_window(xs, list(p['curves'].values()),
                                 args.window_quantile,
                                 truth=p['curves']['truth'],
                                 cap=args.window_cap))
        x = p['tau_star'] * (1.0 if args.scaled else y_scale)
        ax.axvline(x, color=INK_MUTED, lw=1.0, zorder=1)
        ax.annotate(r'$\tau^*$', (x, ax.get_ylim()[1]), xytext=(4, -11),
                    textcoords='offset points', color=INK_MUTED, fontsize=8)
        title = (cat.title if q_shared is not None else
                 f'{cat.title}\nq={p["q"]}   true CATE {p["true_cate"]:+.2f}')
        ax.set_title(title, fontsize=9.5, color=INK, pad=6)
        ax.set_xlabel(rf'$\tau$ ({"scaled" if args.scaled else "raw"} units)',
                      fontsize=8.5, color=INK_2)
        ax.set_ylabel(r'$p(\tau \mid x_q)$', fontsize=8.5, color=INK_2)
        # Under the axes, never in it: any in-plot corner sat on a curve in
        # some panel, and 'best' moved the legend from panel to panel.
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.155),
                  frameon=False, fontsize=7.4, labelcolor=INK_2,
                  handlelength=2.4, borderaxespad=0.0,
                  ncol=1, alignment='left')
        _style_axes(ax)

    ax_t = fig.add_subplot(gs[nrow, :])
    ax_t.axis('off')
    w = max(len(p['cat'].labels[m]) for p in panels for m in p['methods']) + 2
    bw = max(len(p['cat'].title) for p in panels) + 3
    if q_shared is None:
        bw = max(len(f'{p["cat"].title}  q={p["q"]}') for p in panels) + 3
    head = (f'{"block":<{bw}s}{"method":<{w}s}{"nll":>18s}{"l2":>18s}'
            f'{"-log p(tau*)":>15s}')
    lines = [head, '-' * len(head)]
    for p in panels:
        cat = p['cat']
        met = table_metrics(cat)
        for j, m in enumerate(p['methods']):
            cells = ''
            for key in ('nll', 'l2'):
                c = met[m].get(key)
                cells += f'{c[0]:>10.4f}+-{c[1]:<6.4f}' if c else f'{"--":>18s}'
            block = ('' if j else
                     cat.title if q_shared is not None
                     else f'{cat.title}  q={p["q"]}')
            lines.append(f'{block:<{bw}s}{cat.labels[m]:<{w}s}{cells}'
                         f'{p["star"][m]:>15.4f}')
        lines.append('')
    ax_t.text(0.0, 1.0, '\n'.join(lines), transform=ax_t.transAxes, va='top',
              ha='left', fontsize=7.2, family='monospace', color=INK_2)
    # Figure coordinates, not axes: the table's own height varies with the
    # number of blocks, so anchoring the note to the axes let it ride up into
    # the last rows.
    axes_note = ('All panels share one x and y range'
                 + (f', x fixed to [{xlim[0]:g}, {xlim[1]:g}].' if fixed
                    else '.') if shared else
                 'Each panel scales its own axes.')
    fig.text(0.05, 0.020,
             'nll / l2 on the scaled tau axis, mean +- SE over each block\'s '
             'realizations -- the FOR_FURKAN.md values; -log p(tau*) is this '
             'panel\'s individual.',
             ha='left', fontsize=7.4, color=INK_MUTED)
    fig.text(0.05, 0.007,
             'Hues follow each block\'s own table order, so a hue identifies a '
             'model within a panel, not across panels.  ' + axes_note,
             ha='left', fontsize=7.4, color=INK_MUTED)

    note = {'best': 'best-fit', 'worst': 'worst-fit', 'median': 'median'}[args.select]
    fig.suptitle(spec.title, fontsize=14, color=INK, y=0.977)
    if q_shared is not None:
        sub = (f'realization r{r_chosen:03d}{r_note};  every panel shows the '
               f'SAME individual q={q_shared} (true CATE '
               f'{panels[0]["true_cate"]:+.2f}), the {note} one across all '
               f'blocks -- a selected individual, not a typical one')
    else:
        sub = (f'realization r{r_chosen:03d}{r_note};  {note} individual within '
               f'it for each block\'s joint model -- selected queries, not '
               f'typical ones')
    fig.text(0.5, 0.945, sub, ha='center', fontsize=9, color=INK_MUTED)
    return fig


def print_table_check(cat, info):
    print(f'\n    [check-table] {cat.title}  --  compare against FOR_FURKAN.md')
    w = max(len(cat.labels[m]) for m in cat.methods) + 2
    print(f'    {"method":<{w}s}' + ''.join(f'{t:>19s}' for _, t in METRIC_COLUMNS))
    for m in cat.methods:
        row = ''
        for key, _ in METRIC_COLUMNS:
            c = info['metrics'][m].get(key)
            row += f'{c[0]:>11.4f}+-{c[1]:<6.4f}' if c else f'{"--":>19s}'
        print(f'    {cat.labels[m]:<{w}s}{row}')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--category', action='append', default=None,
                    help='category key (repeatable); default: every category')
    ap.add_argument('--grouped', action='append', default=None,
                    help='grouped-figure key (repeatable): several blocks\' '
                         'CATE panels on one sheet. Given alone, only the '
                         'grouped figures are built.')
    ap.add_argument('--list', action='store_true',
                    help='list the categories and the blocks that have no dump')
    ap.add_argument('--realization', default='shared',
                    help="which realization every panel uses: 'shared' "
                         "(default) picks one per dataset by lowest mean "
                         "standardized nll across every block of that dataset, "
                         "so all its figures agree; 'category' ranks the same "
                         "way but only over the block being drawn; or an "
                         "explicit index, e.g. 47")
    ap.add_argument('--xlim', nargs=2, type=float, metavar=('LO', 'HI'),
                    default=None,
                    help='grouped figures: fix the x range, in the plotted '
                         'units (raw unless --scaled). Overrides the figure\'s '
                         'own xlim and the fit-to-curves window; y is then '
                         'scaled to what falls inside it.')
    ap.add_argument('--free-axes', dest='shared_axes', action='store_false',
                    help='grouped figures: let each panel scale its own axes. '
                         'By default all panels share one x and y range, so '
                         'widths and heights are comparable by eye.')
    ap.add_argument('--query', default='shared',
                    help="grouped figures: 'shared' (default) shows ONE "
                         "individual in every panel, so the truth curve is "
                         "identical throughout -- ranked by mean standardized "
                         "-log p(tau*) over every drawn model; 'per-block' "
                         "lets each block pick its own; or an explicit index")
    ap.add_argument('--select', default='best',
                    choices=('best', 'worst', 'median'),
                    help="which individual the CATE panel shows, ranked by the "
                         "anchor model's -log p(tau*) over every realization "
                         "(default: best -- a cherry-pick, and labelled as one)")
    ap.add_argument('--anchor', default=None,
                    help='method the individual is selected for (default: the '
                         'category\'s joint model)')
    ap.add_argument('--all-joints', action='store_true',
                    help='draw every joint in the block, not just the best-nll one')
    ap.add_argument('--scaled', action='store_true',
                    help='plot on the model tau axis instead of raw outcome units')
    ap.add_argument('--window-quantile', type=float, default=0.04,
                    help='axes hold each curve\'s central 1-2q mass (default '
                         '0.04 -> central 92%%); raise it to zoom in further')
    ap.add_argument('--window-cap', type=float, default=6.0,
                    help='clamp the axes to this multiple of the truth density '
                         "'s own central-99%% width, so one heavy-tailed method "
                         'cannot set the frame for everybody (0 disables)')
    ap.add_argument('--ate-stride', type=int, default=1,
                    help='average every k-th individual into p_ATE (default 1: all)')
    ap.add_argument('--n-y0', type=int, default=None,
                    help='tail-quadrature resolution (default: as run)')
    ap.add_argument('--check-table', action='store_true',
                    help='print the aggregated metrics for eyeballing against the doc')
    ap.add_argument('--out-dir', default=os.path.join(REPO, 'figures',
                                                      'category_densities'))
    ap.add_argument('--dpi', type=int, default=160)
    a = ap.parse_args()

    if a.list:
        print('categories with prediction dumps:\n')
        for c in CATEGORIES:
            rs = realizations(c)
            print(f'  {c.key:<22s} {c.title:<34s} {len(rs):>3d} realizations  '
                  f'{c.run}/{c.dataset}')
            print(f'  {"":<22s}   rows: ' + ' | '.join(c.labels[m] for m in c.methods))
        print('\ngrouped figures:\n')
        for g in GROUPED:
            print(f'  {g.key:<22s} {g.title}')
            for key, subset in g.panels:
                c = next(x for x in CATEGORIES if x.key == key)
                extra = ('  [subset: ' + ', '.join(
                    c.labels[m] for m in resolve_subset(c, subset)) + ']'
                    if subset else '')
                print(f'  {"":<22s}   panel: {c.title}{extra}')
        print('\nblocks in FOR_FURKAN.md that cannot be drawn:\n')
        for title, run, why in UNAVAILABLE:
            print(f'  {title:<36s} {run:<40s} {why}')
        return

    import matplotlib
    matplotlib.use('Agg')

    group_index = {g.key: g for g in GROUPED}
    group_keys = a.grouped or ([] if a.category else [g.key for g in GROUPED])
    unknown = [k for k in group_keys if k not in group_index]
    if unknown:
        raise SystemExit(f'unknown grouped figure {unknown}; --list shows the keys')
    keys = a.category or ([] if a.grouped else [c.key for c in CATEGORIES])
    index = {c.key: c for c in CATEGORIES}
    unknown = [k for k in keys if k not in index]
    if unknown:
        raise SystemExit(f'unknown category {unknown}; --list shows the keys')

    os.makedirs(a.out_dir, exist_ok=True)
    # Chosen over EVERY block of the dataset, not just the ones being drawn, so
    # rendering one figure and rendering all ten land on the same realization.
    chosen = {}
    if a.realization not in ('shared', 'category'):
        fixed = int(a.realization)
        chosen = {c.key: (fixed, ' (given)') for c in CATEGORIES}
    elif a.realization == 'shared':
        wanted = ({index[k].dataset for k in keys}
                  | {group_index[g].dataset for g in group_keys})
        for ds in sorted(wanted):
            scope = [c for c in CATEGORIES if c.dataset == ds]
            r, z, n_series, n_common = choose_realization(scope, scope)
            note = (f' -- best average nll over all {ds} blocks '
                    f'(mean z {z:+.2f} of {n_series} series, {n_common} shared)')
            for c in scope:
                chosen[c.key] = (r, note)
    else:
        need = list(keys) + [pk for g in group_keys
                             for pk, _ in group_index[g].panels]
        for k in dict.fromkeys(need):
            c = index[k]
            r, z, n_series, n_common = choose_realization([c], [c])
            chosen[c.key] = (r, f' -- best average nll for this block '
                                f'(mean z {z:+.2f})')

    for k in keys:
        cat = index[k]
        print(f'[{cat.key}] {cat.title}')
        fig, info = build_figure(cat, a, *chosen[cat.key])
        out = os.path.join(a.out_dir, f'{cat.key}.png')
        fig.savefig(out, dpi=a.dpi, facecolor=fig.get_facecolor())
        print(f'    [plot] {out}')
        if a.check_table:
            print_table_check(cat, info)
        import matplotlib.pyplot as plt
        plt.close(fig)

    import matplotlib.pyplot as plt
    for gk in group_keys:
        spec = group_index[gk]
        print(f'[{spec.key}] {spec.title}')
        fig = build_grouped(spec, a, *chosen[spec.panels[0][0]])
        out = os.path.join(a.out_dir, f'{spec.key}.png')
        fig.savefig(out, dpi=a.dpi, facecolor=fig.get_facecolor())
        print(f'    [plot] {out}')
        plt.close(fig)


if __name__ == '__main__':
    main()
