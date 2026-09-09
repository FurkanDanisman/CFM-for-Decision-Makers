"""Plot p(tau | x): generator truth vs UWYK-1D (x) indep vs Joint-2D.

Reads ONLY the prediction dumps eval_density_tauC.py writes under
OUT/predictions (SAVE_PREDICTIONS=1, the default). Neither checkpoint nor a
GPU is needed: the dump carries the raw head outputs plus every axis and
truth parameter, so density_common rebuilds the same curves the eval scored.
Verified against the scored NPZ: recomputing NLL from a dump reproduces
nll_joint / nll_uwyk_native to all printed digits.

Rows plotted per query (same three the eval scores, plus truth):
    truth         N(mu1 - mu0, 2 sigma^2), generator sigma
    joint         2D head's own joint, diagonal-integrated
    uwyk_native   UWYK's K=1000 bars convolved under independence
    uwyk_matched  UWYK re-binned to the 2D head's J bins, then convolved
`uwyk_*` are 'UWYK (x) indep' -- the independence assumption is ours.

Usage:
    python benchmarks/eval_graph2d/plot_density_tauC.py \
        results_density_tauC/5312882/IHDP/predictions/IHDP_r000.npz \
        --n-queries 6 --out fig_tauC_IHDP_r000.png

    # arm marginals f0(y), f1(y) against the truth normals as well
    ... --marginals
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from density_common import (                                       # noqa: E402
    Joint2D, UWYK1D, joint_tau_density, uwyk_tau_density,
    truth_tau_density, TAU_CENTERS,
)

# dataviz reference palette, categorical slots 1-3 (blue / orange / aqua):
# the documented all-pairs-validated subset. Truth is not a series -- it is the
# reference, so it wears text ink, not a hue.
INK = '#0b0b0b'
INK_MUTED = '#8a8985'
# The two UWYK rows sit almost exactly on top of each other (re-binning barely
# moves the convolved tau density), so `matched` is drawn thinner and dashed
# ON TOP of `native` -- otherwise one hue is simply invisible under the other.
SERIES = {
    'joint':        ('#2a78d6', 'Joint-2D',                    2.0, '-',        4),
    'uwyk_native':  ('#eb6834', 'UWYK (x) indep, native K=1000', 2.6, '-',       3),
    'uwyk_matched': ('#1baf7a', 'UWYK (x) indep, matched to J', 1.5, (0, (4, 2)), 4),
}
SHORT = {'joint': 'joint', 'uwyk_native': 'uwyk native',
         'uwyk_matched': 'uwyk matched'}


def joint_marginals(jt, y, n_y=1024, n_pad_sigma=8.0):
    """The joint's TRUE marginals f(y0), f(y1), tail regions included.

    p_mat.sum(axis) alone is the INTERIOR-conditional marginal: it stops dead
    at the head's bin edges and hides regions 1-8. Integrating jt.density over
    the partner axis instead gives the marginal the eval actually scores.
    Midpoint cells aligned to the bin grid, so no cell straddles a region
    boundary -- same treatment as tau_density_quadrature.
    """
    J = jt.p_mat.shape[0]
    per_bin = max(1, int(round(n_y / J)))
    n_inner = J * per_bin
    step = (jt.hi - jt.lo) / n_inner
    n_pad = int(np.ceil(n_pad_sigma * jt.max_scale / step))
    ce = jt.lo + step * np.arange(-n_pad, n_inner + n_pad + 1)
    g = 0.5 * (ce[:-1] + ce[1:])

    y = np.asarray(y, dtype=np.float64)
    m0 = np.empty(y.shape, dtype=np.float64)
    m1 = np.empty(y.shape, dtype=np.float64)
    chunk = max(1, 4_000_000 // g.size)
    for i in range(0, y.size, chunk):
        yy = y[i:i + chunk][:, None]
        G = np.broadcast_to(g, (yy.shape[0], g.size))
        Y = np.broadcast_to(yy, G.shape)
        m0[i:i + chunk] = jt.density(Y, G).sum(axis=1) * step
        m1[i:i + chunk] = jt.density(G, Y).sum(axis=1) * step
    return m0, m1


def rebuild(d, q, n_y0, tau):
    """The four densities for query q, on the scaled tau axis."""
    J, e2 = int(d['J']), d['edges2d']
    be, bwd = d['bar_edges'], d['bar_widths']
    sL, sR = float(d['base_sL']), float(d['base_sR'])

    jt = Joint2D.from_pred(d['joint_logits'][q], J, e2)
    f0 = UWYK1D.from_pred(d['uwyk_pred0'][q], be, bwd, sL, sR)
    f1 = UWYK1D.from_pred(d['uwyk_pred1'][q], be, bwd, sL, sR)
    f0m, f1m = f0.rebin(e2), f1.rebin(e2)
    curves = {
        'joint': joint_tau_density(jt, tau, n_y0=n_y0),
        'uwyk_native': uwyk_tau_density(f0, f1, tau, n_y0=n_y0),
        'uwyk_matched': uwyk_tau_density(f0m, f1m, tau, n_y0=n_y0),
    }
    truth = truth_tau_density(d['mu0_scaled'][q], d['mu1_scaled'][q],
                              float(d['sigma_scaled']), tau)
    return truth, curves, (f0, f1, jt)


def nll_at_star(d, q, n_y0):
    """-log p(tau*) per method, evaluated at the point, not read off the grid."""
    t = np.array([d['tau_star_scaled'][q]])
    _, curves, _ = rebuild(d, q, n_y0, t)
    return {k: float(-np.log(v[0])) for k, v in curves.items()}


def pick_queries(d, n, how, n_y0):
    n_q = d['joint_logits'].shape[0]
    n = min(n, n_q)
    if how == 'first':
        return list(range(n))
    if how == 'spread':                      # even quantiles of the true CATE
        order = np.argsort(np.asarray(d['true_cate']).reshape(-1))
        return sorted(order[np.linspace(0, n_q - 1, n).round().astype(int)])
    if how in ('gap', 'worst'):              # where the methods disagree most
        gaps = []
        for q in range(n_q):
            s = nll_at_star(d, q, n_y0)
            gaps.append(min(s['uwyk_native'], s['uwyk_matched']) - s['joint'])
        order = np.argsort(gaps)[::-1] if how == 'gap' else np.argsort(gaps)
        return sorted(order[:n])
    raise ValueError(how)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('pred', help='OUT/predictions/<DATASET>_r<NNN>.npz')
    ap.add_argument('--queries', help='comma-separated test-row indices')
    ap.add_argument('--n-queries', type=int, default=6)
    ap.add_argument('--select', default='spread',
                    choices=('spread', 'gap', 'worst', 'first'),
                    help="spread: even quantiles of true CATE (default -- the "
                         "unbiased view); gap/worst: the queries where the joint "
                         "beats / loses to UWYK by the most NLL, for diagnosis "
                         "only -- both are cherry-picks and must be labelled as "
                         "such; first: 0..n-1")
    ap.add_argument('--scaled', action='store_true',
                    help='plot on the model tau axis instead of raw outcome units')
    ap.add_argument('--n-y0', type=int, default=None,
                    help='tail-quadrature resolution (default: as run)')
    ap.add_argument('--marginals', action='store_true',
                    help='second figure: arm densities f0(y), f1(y) vs truth')
    ap.add_argument('--out', default=None, help='output PNG (default: alongside pred)')
    ap.add_argument('--dpi', type=int, default=160)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    d = np.load(a.pred, allow_pickle=True)
    n_y0 = a.n_y0 or int(d['n_y0'])
    tau = np.asarray(d['tau_grid']) if 'tau_grid' in d.files else TAU_CENTERS
    y_scale, y_shift = float(d['y_scale']), float(d['y_shift'])
    ds, r = str(d['dataset']), int(d['realization'])

    qs = ([int(x) for x in a.queries.split(',')] if a.queries
          else pick_queries(d, a.n_queries, a.select, n_y0))

    # raw <- scaled: x_raw = tau * y_scale, and p_raw = p_scaled / y_scale so
    # the curve stays a density (area 1) after the change of variable.
    xs, dens_scale = (tau, 1.0) if a.scaled else (tau * y_scale, 1.0 / y_scale)
    unit = 'scaled' if a.scaled else 'raw'

    ncol = min(3, len(qs))
    nrow = int(np.ceil(len(qs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.3 * nrow),
                             squeeze=False)
    fig.patch.set_facecolor('#fcfcfb')

    for ax, q in zip(axes.ravel(), qs):
        truth, curves, _ = rebuild(d, q, n_y0, tau)
        scores = nll_at_star(d, q, n_y0)
        t_star = float(d['tau_star_scaled'][q]) * (1.0 if a.scaled else y_scale)

        ax.plot(xs, truth * dens_scale, color=INK, lw=2.0, ls=(0, (5, 2)),
                label='truth', zorder=5)
        for key, (color, label, lw, ls, z) in SERIES.items():
            ax.plot(xs, curves[key] * dens_scale, color=color, lw=lw, ls=ls,
                    label=label, zorder=z, solid_capstyle='round')
        ax.axvline(t_star, color=INK_MUTED, lw=1.0, zorder=1)
        ax.annotate(r'$\tau^*$', (t_star, ax.get_ylim()[1]), xytext=(3, -10),
                    textcoords='offset points', color=INK_MUTED, fontsize=8)

        # window on the truth, widened to hold every model's bulk
        s_tau = np.sqrt(2.0) * float(d['sigma_scaled']) * (1.0 if a.scaled else y_scale)
        c = (float(d['mu1_scaled'][q]) - float(d['mu0_scaled'][q])) * \
            (1.0 if a.scaled else y_scale)
        peaks = [xs[np.argmax(v)] for v in curves.values()]
        lo = min(c - 4 * s_tau, min(peaks) - 2 * s_tau)
        hi = max(c + 4 * s_tau, max(peaks) + 2 * s_tau)
        ax.set_xlim(lo, hi)

        # relief rule: the curves are also identified by their NLL, in ink
        txt = '\n'.join(f'{SHORT[k]:<13s}{scores[k]:7.3f}'
                        for k in ('joint', 'uwyk_native', 'uwyk_matched'))
        ax.text(0.02, 0.97, f'$-\\log p(\\tau^*)$\n{txt}', transform=ax.transAxes,
                va='top', ha='left', fontsize=7.5, family='monospace',
                color='#52514e', zorder=6,
                bbox=dict(facecolor='#fcfcfb', edgecolor='none', alpha=0.85,
                          boxstyle='round,pad=0.3'))
        ax.set_title(f'q={q}   true CATE {float(d["true_cate"][q]):+.2f}',
                     fontsize=9, color=INK)
        ax.set_xlabel(rf'$\tau$ ({unit} units)', fontsize=8.5, color='#52514e')
        ax.set_ylabel(r'$p(\tau \mid x)$', fontsize=8.5, color='#52514e')
        ax.tick_params(labelsize=8, colors='#52514e')
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color('#d9d8d3')
        ax.grid(True, color='#eceae5', lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.set_facecolor('#fcfcfb')
    for ax in axes.ravel()[len(qs):]:
        ax.set_visible(False)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.005))
    note = {'gap': '  [SELECTED: joint\'s largest NLL wins -- not representative]',
            'worst': '  [SELECTED: joint\'s largest NLL losses -- not representative]',
            'first': '', 'spread': ''}[a.select if not a.queries else 'first']
    fig.suptitle(f'{ds} r{r:03d}  --  p(tau | x): truth vs Joint-2D vs '
                 f'UWYK (x) indep   [anc={str(d["anc_tag"])}, n_y0={n_y0}]{note}',
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.045, 1, 0.97))

    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.pred)),
                                f'{ds}_r{r:03d}_tau_density.png')
    fig.savefig(out, dpi=a.dpi, facecolor=fig.get_facecolor())
    print(f'[plot] {out}   queries={list(map(int, qs))}')

    if a.marginals:
        # Window must cover BOTH heads' tails, not just the shared interior --
        # clipping at the bin edges is what made the joint look guillotined.
        jts = [rebuild(d, q, n_y0, np.array([0.0]))[2][2] for q in qs]
        f0s = [rebuild(d, q, n_y0, np.array([0.0]))[2][0] for q in qs]
        # Evaluate wide (so nothing is cut), then set the VIEW from where the
        # mass is -- the tail scales alone would zoom out to a useless window.
        pad = 6.0 * max(max(j.max_scale for j in jts),
                        max(f.max_scale for f in f0s))
        lo_e, hi_e = float(d['edges2d'][0]), float(d['edges2d'][-1])
        be_ = d['bar_edges']
        y = np.linspace(min(lo_e, float(be_[0])) - pad,
                        max(hi_e, float(be_[-1])) + pad, 4001)
        yx = y if a.scaled else y * y_scale + y_shift
        bx = (lambda v: v if a.scaled else v * y_scale + y_shift)
        dsc = 1.0 if a.scaled else 1.0 / y_scale
        sig = float(d['sigma_scaled'])
        fig2, axes2 = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.3 * nrow),
                                   squeeze=False)
        fig2.patch.set_facecolor('#fcfcfb')
        for ax, q in zip(axes2.ravel(), qs):
            _, _, (f0, f1, jt) = rebuild(d, q, n_y0, np.array([0.0]))
            jm0, jm1 = joint_marginals(jt, y)
            for arm, (f, jm, mu) in enumerate(
                    ((f0, jm0, d['mu0_scaled'][q]), (f1, jm1, d['mu1_scaled'][q]))):
                ls = '-' if arm else (0, (4, 2))
                ax.plot(yx, f.density(y) * dsc, color=SERIES['uwyk_native'][0],
                        lw=1.8, ls=ls, label=f'UWYK arm {arm}', zorder=3)
                ax.plot(yx, jm * dsc, color=SERIES['joint'][0], lw=1.8, ls=ls,
                        label=f'Joint-2D arm {arm}', zorder=4)
                z = (y - float(mu)) / sig
                ax.plot(yx, np.exp(-0.5 * z * z) / (sig * np.sqrt(2 * np.pi)) * dsc,
                        color=INK, lw=1.3, ls=ls, alpha=0.75,
                        label=f'truth arm {arm}', zorder=5)
            for e in (lo_e, hi_e):        # the head's hard support boundary
                ax.axvline(bx(e), color=INK_MUTED, lw=1.0, ls=':', zorder=1)
            m0b, m1b = jt.p_mat.sum(axis=1), jt.p_mat.sum(axis=0)
            edge = max(m0b[-3:].sum(), m1b[-3:].sum(),
                       m0b[:3].sum(), m1b[:3].sum())
            ax.text(0.02, 0.97,
                    f'outer-3-bin mass {edge:.3f}\ntail regions   {1-jt.w[0]:.1e}',
                    transform=ax.transAxes, va='top', ha='left', fontsize=7.5,
                    family='monospace', color='#52514e', zorder=6,
                    bbox=dict(facecolor='#fcfcfb', edgecolor='none', alpha=0.85,
                              boxstyle='round,pad=0.3'))
            drawn = [f0.density(y), f1.density(y), jm0, jm1]
            keep = np.zeros(y.shape, dtype=bool)
            for c in drawn:                      # 1e-3 of each curve's peak
                keep |= c > 1e-3 * c.max()
            xl, xh = yx[keep].min(), yx[keep].max()
            m = 0.06 * (xh - xl)
            ax.set_xlim(min(xl - m, bx(lo_e) - m), max(xh + m, bx(hi_e) + m))
            ax.set_title(f'q={q}', fontsize=9, color=INK)
            ax.set_xlabel(f'y ({unit} units)', fontsize=8.5, color='#52514e')
            ax.set_ylabel('density', fontsize=8.5, color='#52514e')
            ax.tick_params(labelsize=8, colors='#52514e')
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            ax.grid(True, color='#eceae5', lw=0.8)
            ax.set_axisbelow(True)
            ax.set_facecolor('#fcfcfb')
        for ax in axes2.ravel()[len(qs):]:
            ax.set_visible(False)
        h, l = axes2.ravel()[0].get_legend_handles_labels()
        fig2.legend(h, l, loc='lower center', ncol=3, frameon=False, fontsize=8.5,
                    bbox_to_anchor=(0.5, -0.005))
        fig2.suptitle(f'{ds} r{r:03d}  --  arm densities f0(y), f1(y)   '
                      f'(dotted verticals = 2D head bin support)',
                      fontsize=11, color=INK)
        fig2.tight_layout(rect=(0, 0.10, 1, 0.97))
        out2 = out.replace('.png', '_marginals.png')
        fig2.savefig(out2, dpi=a.dpi, facecolor=fig2.get_facecolor())
        print(f'[plot] {out2}')


if __name__ == '__main__':
    main()
