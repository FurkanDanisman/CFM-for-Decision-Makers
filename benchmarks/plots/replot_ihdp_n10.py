"""Rebuild every IHDP N=10 PNG locally from the cache.npz files that the
sibling compute scripts drop into benchmarks/plots/ihdp_n10/<FOLDER>/.

Killarney runs the heavy scripts (inference, MALC, UWYK sampling) once and
writes cache.npz + PNGs per folder. This script needs only numpy +
matplotlib — you can iterate on styling here without re-running any
model. Regenerates:

  UWYK-2DMALC/           joint.png, marginals.png, te.png, ot.png
  UWYK-2DMALC-NAIVE/     marginals.png, te.png, ot.png
  UWYK-NOANC/            marginals.png, te.png, ot.png
  UWYK-NOANC-COMONOTONIC marginals.png, te.png, ot.png
  MARGINALS-COMPARE/     marginals_compare.png     (+ true overlay if IHDP-TRUE-TE cache exists)
  IHDP-TRUE-TE/          marginals.png, te.png, ot.png

If a folder's cache.npz is missing, that folder is skipped with a note.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

_HERE   = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.join(_HERE, 'ihdp_n10')

# ── Styling knobs (edit here) ────────────────────────────────────────────
COLOR_DO0        = '#2E7DAF'
COLOR_DO1        = '#7B3E9E'
COLOR_JOINT      = '#2E4A6F'
COLOR_NAIVE      = '#C1420F'
COLOR_COMONO     = '#0F8A3C'
COLOR_TRUE       = 'red'
COLOR_MEAN_MEANS = '#F5A623'   # orange (was #C1420F)
COLOR_OT         = '#0F8A3C'
COLOR_TRUE_ATE   = 'red'

# ── 1-D W2 barycenter via Agueh–Carlier quantile averaging ───────────────
# Mirrors MALC/Optimal_Transport/ot_barycenter.py so local re-plots match
# the killarney-produced curves exactly.
def _density_to_quantile(f, x_grid, taus):
    dx = float(x_grid[1] - x_grid[0])
    F = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * dx)])
    if F[-1] <= 0:
        return np.full_like(taus, float(np.mean(x_grid)))
    F = F / F[-1]
    return np.interp(taus, F, x_grid)


def w2_barycenter_1d(densities: np.ndarray, x_grid: np.ndarray, n_tau: int = 4001) -> np.ndarray:
    densities = np.atleast_2d(np.asarray(densities, dtype=float))
    N, M = densities.shape
    weights = np.full(N, 1.0 / N)
    taus = (np.arange(n_tau) + 0.5) / n_tau
    Q = np.stack([_density_to_quantile(densities[i], x_grid, taus) for i in range(N)])
    Q_bary = (weights[:, None] * Q).sum(axis=0)
    F_bary = np.interp(x_grid, Q_bary, taus, left=0.0, right=1.0)
    dx = float(x_grid[1] - x_grid[0])
    f_bary = np.gradient(F_bary, dx)
    f_bary = np.clip(f_bary, 0.0, None)
    s = float(f_bary.sum() * dx)
    if s > 0:
        f_bary = f_bary / s
    return f_bary


def _grid_layout(n):
    n_cols = 5 if n == 10 else 3
    n_rows = (n + n_cols - 1) // n_cols
    return n_rows, n_cols


def _load(folder):
    p = os.path.join(_ROOT, folder, 'cache.npz')
    if not os.path.isfile(p):
        print(f'[skip] {folder}: no cache.npz')
        return None
    print(f'[read] {p}')
    return np.load(p, allow_pickle=False)


def _save(fig, folder, name):
    out = os.path.join(_ROOT, folder, name)
    fig.savefig(out, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'[save] {out}')


def _ot_plot(densities, tau_centers, true_ate, title, out_folder):
    tau_step = float(tau_centers[1] - tau_centers[0])
    bary = w2_barycenter_1d(densities, tau_centers)
    bary /= max(bary.sum() * tau_step, 1e-12)
    bary_mode = float(tau_centers[int(np.argmax(bary))])
    per_q_means = (tau_centers[None, :] * densities).sum(axis=1) * tau_step
    mean_of_means = float(per_q_means.mean())

    n_q = densities.shape[0]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    palette_Q = plt.cm.tab10(np.linspace(0, 0.9, n_q))
    for k in range(n_q):
        ax.plot(tau_centers, densities[k], color=palette_Q[k], lw=1.1, alpha=0.35)
    ax.fill_between(tau_centers, bary, alpha=0.20, color=COLOR_OT)
    ax.plot(tau_centers, bary, color=COLOR_OT, lw=2.6, label='W₂ barycenter (OT)')
    ax.axvline(true_ate, color=COLOR_TRUE_ATE, ls='--', lw=1.6,
                label=f'true population ATE = {true_ate:+.2f}')
    ax.axvline(bary_mode, color=COLOR_OT, ls='--', lw=1.6,
                label=f'OT-mode = {bary_mode:+.2f}')
    ax.axvline(mean_of_means, color=COLOR_MEAN_MEANS, ls='--', lw=1.6,
                label=f'mean-of-means = {mean_of_means:+.2f}')
    ax.set_xlim(float(tau_centers[0]), float(tau_centers[-1]))
    ax.set_xlabel(r'$\tau = Y_{do1} - Y_{do0}$  (scaled)')
    ax.set_ylabel('density')
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, out_folder, 'ihdp_n10_ot.png')


def _dot_at(ax, x, ys, color, filled=True):
    y_here = float(np.interp(x, np.linspace(-1.5, 1.5, len(ys)), ys))
    if filled:
        ax.plot(x, y_here, 'o', color=color, markersize=9,
                 markeredgecolor='white', markeredgewidth=1.0, zorder=5)
    else:
        ax.plot(x, y_here, 'o', markerfacecolor='none', markeredgecolor=color,
                 markersize=10, markeredgewidth=1.8, zorder=5)


def _marginals_panel(centers, p_y0, p_y1, query_idxs, true_cate_scaled,
                       title, out_folder, out_name='ihdp_n10_marginals.png',
                       true_overlay=None):
    """true_overlay: optional dict {'p_y0_true':..., 'p_y1_true':..., 'mu0_scaled':..., 'mu1_scaled':..., 'sigma_scaled':...}"""
    n_q = len(query_idxs)
    n_rows, n_cols = _grid_layout(n_q)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 3.6 * n_rows),
                              squeeze=False)
    bin_width = float(centers[1] - centers[0])
    for k, q in enumerate(query_idxs):
        ax = axes[k // n_cols][k % n_cols]
        ax.plot(centers, p_y0[k], color=COLOR_DO0, lw=1.9, label=r'$p(Y_{do0})$')
        ax.plot(centers, p_y1[k], color=COLOR_DO1, lw=1.9, label=r'$p(Y_{do1})$')
        E_y0 = float((centers * p_y0[k]).sum() * bin_width)
        E_y1 = float((centers * p_y1[k]).sum() * bin_width)
        ax.plot(E_y0, float(np.interp(E_y0, centers, p_y0[k])), 'o',
                 color=COLOR_DO0, markersize=9, markeredgecolor='white',
                 markeredgewidth=1.0, zorder=5)
        ax.plot(E_y1, float(np.interp(E_y1, centers, p_y1[k])), 'o',
                 color=COLOR_DO1, markersize=9, markeredgecolor='white',
                 markeredgewidth=1.0, zorder=5)
        if true_overlay is not None:
            p_y0_true = true_overlay['p_y0_true'][k]
            p_y1_true = true_overlay['p_y1_true'][k]
            mu0 = float(true_overlay['mu0_scaled'][q])
            mu1 = float(true_overlay['mu1_scaled'][q])
            ax.plot(centers, p_y0_true, color=COLOR_TRUE, lw=0.9, ls=':', alpha=0.9)
            ax.plot(centers, p_y1_true, color=COLOR_TRUE, lw=0.9, ls=':', alpha=0.9)
            ax.plot(mu0, float(np.interp(mu0, centers, p_y0_true)),
                     'o', color=COLOR_TRUE, markersize=6, zorder=6)
            ax.plot(mu1, float(np.interp(mu1, centers, p_y1_true)),
                     'o', color=COLOR_TRUE, markersize=6, zorder=6)
        ax.set_title(f'query {int(q)}   $\\tau_{{true}}$={float(true_cate_scaled[q]):+.2f}',
                      fontsize=10)
        if k // n_cols == n_rows - 1: ax.set_xlabel(r'$Y$  (scaled)')
        if k %  n_cols == 0:          ax.set_ylabel('density')
        ax.grid(alpha=0.25)
        if k == 0: ax.legend(fontsize=9, loc='upper right')
    for k in range(n_q, n_rows * n_cols):
        axes[k // n_cols][k % n_cols].set_visible(False)
    fig.suptitle(title, fontsize=12, y=0.999)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    _save(fig, out_folder, out_name)


# ── UWYK-2DMALC (Ours) + UWYK-2DMALC-NAIVE ──────────────────────────────
ours = _load('UWYK-2DMALC')
if ours is not None:
    p_mats           = ours['p_mats']
    centers          = ours['centers']
    bin_width        = float(ours['bin_width'])
    edges            = ours['edges']
    tau_centers      = ours['tau_centers']
    p_taus           = ours['p_taus']
    p_taus_naive     = ours['p_taus_naive']
    true_cate_scaled = ours['true_cate_scaled']
    QUERY_IDXS       = list(map(int, ours['QUERY_IDXS']))
    REALIZATION      = int(ours['realization'])
    N_CONTEXT        = int(ours['n_context'])
    N_QUERIES        = len(QUERY_IDXS)
    true_ate = float(true_cate_scaled.mean())

    # marginals from p_mats
    p_y0_ours = np.zeros((N_QUERIES, len(centers)))
    p_y1_ours = np.zeros((N_QUERIES, len(centers)))
    for k in range(N_QUERIES):
        m0 = p_mats[k].sum(axis=1); m1 = p_mats[k].sum(axis=0)
        p_y0_ours[k] = m0 / max(m0.sum() * bin_width, 1e-12)
        p_y1_ours[k] = m1 / max(m1.sum() * bin_width, 1e-12)

    # try to fetch true overlay for the marginals-compare figure later
    true_cache = _load('IHDP-TRUE-TE')

    # -- joint p(Y_do0, Y_do1) ---------------------------------------------
    n_rows, n_cols = _grid_layout(N_QUERIES)
    extent = [edges[0], edges[-1], edges[0], edges[-1]]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 4.0 * n_rows),
                              squeeze=False)
    for k, q in enumerate(QUERY_IDXS):
        ax = axes[k // n_cols][k % n_cols]
        im = ax.imshow(p_mats[k].T, origin='lower', extent=extent,
                        cmap='viridis', aspect='auto')
        ax.plot([edges[0], edges[-1]], [edges[0], edges[-1]],
                 'r--', lw=0.7, alpha=0.55)
        ax.set_title(f'query {q}   $\\tau_{{true}}$={float(true_cate_scaled[q]):+.2f}',
                      fontsize=10)
        if k // n_cols == n_rows - 1: ax.set_xlabel(r'$Y_{do0}$  (scaled)')
        if k %  n_cols == 0:          ax.set_ylabel(r'$Y_{do1}$  (scaled)')
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    for k in range(N_QUERIES, n_rows * n_cols):
        axes[k // n_cols][k % n_cols].set_visible(False)
    fig.suptitle(f'IHDP r={REALIZATION}   joint $p(Y_{{do0}}, Y_{{do1}})$   at N={N_CONTEXT}',
                  fontsize=12, y=0.999)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    _save(fig, 'UWYK-2DMALC', 'ihdp_n10_joint.png')

    # -- marginals (same for both folders) ---------------------------------
    _marginals_panel(centers, p_y0_ours, p_y1_ours, QUERY_IDXS, true_cate_scaled,
                      f'IHDP r={REALIZATION}   marginal potential-outcome densities at N={N_CONTEXT}',
                      out_folder='UWYK-2DMALC')
    _marginals_panel(centers, p_y0_ours, p_y1_ours, QUERY_IDXS, true_cate_scaled,
                      f'IHDP r={REALIZATION}   marginal potential-outcome densities at N={N_CONTEXT}',
                      out_folder='UWYK-2DMALC-NAIVE')

    # -- TE per-query (joint vs naive overlay) -----------------------------
    def _te_figure(include_joint, out_folder):
        n_rows, n_cols = _grid_layout(N_QUERIES)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 3.6 * n_rows),
                                  squeeze=False)
        tau_step = float(tau_centers[1] - tau_centers[0])
        for k, q in enumerate(QUERY_IDXS):
            ax = axes[k // n_cols][k % n_cols]
            p_tau = p_taus[k]; p_tau_n = p_taus_naive[k]
            E_tau   = float((tau_centers * p_tau).sum() * tau_step)
            E_tau_n = float((tau_centers * p_tau_n).sum() * tau_step)
            if include_joint:
                ax.fill_between(tau_centers, p_tau, alpha=0.25, color=COLOR_JOINT)
                ax.plot(tau_centers, p_tau, color=COLOR_JOINT, lw=2.0,
                         label=f'joint $p(\\tau)$  E={E_tau:+.2f}')
                ax.axvline(E_tau, color=COLOR_JOINT, ls='--', lw=1.4, alpha=0.85)
                ax.plot(tau_centers, p_tau_n, color=COLOR_NAIVE, lw=1.8, ls='--',
                         label=f'naive $p(\\tau)$  E={E_tau_n:+.2f}')
                ax.axvline(E_tau_n, color=COLOR_NAIVE, ls=':', lw=1.4, alpha=0.85)
            else:
                ax.fill_between(tau_centers, p_tau_n, alpha=0.25, color=COLOR_NAIVE)
                ax.plot(tau_centers, p_tau_n, color=COLOR_NAIVE, lw=2.0,
                         label=f'naive $p(\\tau)$  E={E_tau_n:+.2f}')
                ax.axvline(E_tau_n, color=COLOR_NAIVE, ls='--', lw=1.4, alpha=0.85)
            ax.axvline(float(true_cate_scaled[q]), color=COLOR_TRUE, ls='--', lw=1.4,
                        label=f'true $\\tau$={float(true_cate_scaled[q]):+.2f}')
            ax.plot(float(true_cate_scaled[q]), 0, 'o', color=COLOR_TRUE,
                     markersize=9, clip_on=False, zorder=6)
            ax.set_xlim(-1.0, 1.0)
            ax.set_title(f'query {int(q)}', fontsize=10)
            if k // n_cols == n_rows - 1: ax.set_xlabel(r'$\tau = Y_{do1} - Y_{do0}$  (scaled)')
            if k %  n_cols == 0:          ax.set_ylabel(r'$p(\tau)$')
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8, loc='upper right')
        for k in range(N_QUERIES, n_rows * n_cols):
            axes[k // n_cols][k % n_cols].set_visible(False)
        if include_joint:
            title = (f'IHDP r={REALIZATION}   per-query TE distributions at N={N_CONTEXT}   '
                      '(joint = Ours, naive = marginal-only independence)')
        else:
            title = (f'IHDP r={REALIZATION}   naive per-query TE at N={N_CONTEXT}   '
                      '(from marginals under independence)')
        fig.suptitle(title, fontsize=12, y=0.999)
        fig.tight_layout(rect=[0, 0, 1, 0.985])
        _save(fig, out_folder, 'ihdp_n10_te.png')

    _te_figure(include_joint=True,  out_folder='UWYK-2DMALC')
    _te_figure(include_joint=False, out_folder='UWYK-2DMALC-NAIVE')

    # -- OT plots (Ours joint / Ours naive) --------------------------------
    _ot_plot(p_taus, tau_centers, true_ate,
              f'IHDP r={REALIZATION}   OT aggregation (joint p(τ)) at N={N_CONTEXT}',
              'UWYK-2DMALC')
    _ot_plot(p_taus_naive, tau_centers, true_ate,
              f'IHDP r={REALIZATION}   OT aggregation (naive p(τ)) at N={N_CONTEXT}',
              'UWYK-2DMALC-NAIVE')

# ── UWYK-NOANC (histogrammed marginals + naive p(τ)) ────────────────────
noanc = _load('UWYK-NOANC')
if noanc is not None:
    centers = noanc['centers']; edges = noanc['edges']
    p_y0 = noanc['p_y0']; p_y1 = noanc['p_y1']
    tau_centers = noanc['tau_centers']
    p_taus_naive = noanc['p_taus_naive']
    true_cate_scaled = noanc['true_cate_scaled']
    QUERY_IDXS = list(map(int, noanc['QUERY_IDXS']))
    REALIZATION = int(noanc['realization']); N_CONTEXT = int(noanc['n_context'])
    N_QUERIES = len(QUERY_IDXS); true_ate = float(true_cate_scaled.mean())

    _marginals_panel(centers, p_y0, p_y1, QUERY_IDXS, true_cate_scaled,
                      f'IHDP r={REALIZATION}   UWYK No-Ancestral marginal densities at N={N_CONTEXT}',
                      out_folder='UWYK-NOANC')

    # TE panel (naive only, matches sibling script styling)
    n_rows, n_cols = _grid_layout(N_QUERIES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 3.6 * n_rows),
                              squeeze=False)
    tau_step = float(tau_centers[1] - tau_centers[0])
    for k, q in enumerate(QUERY_IDXS):
        ax = axes[k // n_cols][k % n_cols]
        p_tau_n = p_taus_naive[k]
        E_tau_n = float((tau_centers * p_tau_n).sum() * tau_step)
        ax.fill_between(tau_centers, p_tau_n, alpha=0.25, color=COLOR_NAIVE)
        ax.plot(tau_centers, p_tau_n, color=COLOR_NAIVE, lw=2.0,
                 label=f'naive $p(\\tau)$  E={E_tau_n:+.2f}')
        ax.axvline(E_tau_n, color=COLOR_NAIVE, ls='--', lw=1.4, alpha=0.85)
        ax.axvline(float(true_cate_scaled[q]), color=COLOR_TRUE, ls='--', lw=1.4,
                    label=f'true $\\tau$={float(true_cate_scaled[q]):+.2f}')
        ax.plot(float(true_cate_scaled[q]), 0, 'o', color=COLOR_TRUE, markersize=9,
                 clip_on=False, zorder=6)
        ax.set_xlim(-1.5, 1.5)
        ax.set_title(f'query {int(q)}', fontsize=10)
        if k // n_cols == n_rows - 1: ax.set_xlabel(r'$\tau = Y_{do1} - Y_{do0}$  (scaled)')
        if k %  n_cols == 0:          ax.set_ylabel(r'$p(\tau)$')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc='upper right')
    for k in range(N_QUERIES, n_rows * n_cols):
        axes[k // n_cols][k % n_cols].set_visible(False)
    fig.suptitle(f'IHDP r={REALIZATION}   UWYK No-Ancestral naive TE (independence) at N={N_CONTEXT}',
                  fontsize=12, y=0.999)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    _save(fig, 'UWYK-NOANC', 'ihdp_n10_te.png')

    _ot_plot(p_taus_naive, tau_centers, true_ate,
              f'IHDP r={REALIZATION}   UWYK No-Ancestral OT aggregation (naive TE) at N={N_CONTEXT}',
              'UWYK-NOANC')

# ── UWYK-NOANC-COMONOTONIC ─────────────────────────────────────────────
comono = _load('UWYK-NOANC-COMONOTONIC')
if comono is not None:
    centers = comono['centers']; edges = comono['edges']
    p_y0 = comono['p_y0']; p_y1 = comono['p_y1']
    tau_centers = comono['tau_centers']
    p_taus_c = comono['p_taus_comono']
    true_cate_scaled = comono['true_cate_scaled']
    QUERY_IDXS = list(map(int, comono['QUERY_IDXS']))
    REALIZATION = int(comono['realization']); N_CONTEXT = int(comono['n_context'])
    N_QUERIES = len(QUERY_IDXS); true_ate = float(true_cate_scaled.mean())

    _marginals_panel(centers, p_y0, p_y1, QUERY_IDXS, true_cate_scaled,
                      f'IHDP r={REALIZATION}   UWYK No-Ancestral marginals at N={N_CONTEXT}',
                      out_folder='UWYK-NOANC-COMONOTONIC')

    n_rows, n_cols = _grid_layout(N_QUERIES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 3.6 * n_rows),
                              squeeze=False)
    tau_step = float(tau_centers[1] - tau_centers[0])
    for k, q in enumerate(QUERY_IDXS):
        ax = axes[k // n_cols][k % n_cols]
        p = p_taus_c[k]
        E = float((tau_centers * p).sum() * tau_step)
        ax.fill_between(tau_centers, p, alpha=0.25, color=COLOR_COMONO)
        ax.plot(tau_centers, p, color=COLOR_COMONO, lw=2.0,
                 label=f'comonotonic $p(\\tau)$  E={E:+.2f}')
        ax.axvline(E, color=COLOR_COMONO, ls='--', lw=1.4, alpha=0.85)
        ax.axvline(float(true_cate_scaled[q]), color=COLOR_TRUE, ls='--', lw=1.4,
                    label=f'true $\\tau$={float(true_cate_scaled[q]):+.2f}')
        ax.plot(float(true_cate_scaled[q]), 0, 'o', color=COLOR_TRUE, markersize=9,
                 clip_on=False, zorder=6)
        ax.set_xlim(-1.5, 1.5)
        ax.set_title(f'query {int(q)}', fontsize=10)
        if k // n_cols == n_rows - 1: ax.set_xlabel(r'$\tau = Y_{do1} - Y_{do0}$  (scaled)')
        if k %  n_cols == 0:          ax.set_ylabel(r'$p(\tau)$')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc='upper right')
    for k in range(N_QUERIES, n_rows * n_cols):
        axes[k // n_cols][k % n_cols].set_visible(False)
    fig.suptitle(f'IHDP r={REALIZATION}   UWYK No-Ancestral comonotonic TE at N={N_CONTEXT}',
                  fontsize=12, y=0.999)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    _save(fig, 'UWYK-NOANC-COMONOTONIC', 'ihdp_n10_te.png')

    _ot_plot(p_taus_c, tau_centers, true_ate,
              f'IHDP r={REALIZATION}   UWYK No-Ancestral OT aggregation (comonotonic TE) at N={N_CONTEXT}',
              'UWYK-NOANC-COMONOTONIC')

# ── IHDP-TRUE-TE (true marginals + true TE + true OT) ───────────────────
true_ = _load('IHDP-TRUE-TE')
if true_ is not None:
    centers = true_['centers']; edges = true_['edges']
    tau_centers = true_['tau_centers']
    p_y0_t = true_['p_y0_true']; p_y1_t = true_['p_y1_true']
    p_taus_t = true_['p_taus_true']
    mu0_scaled = true_['mu0_scaled']; mu1_scaled = true_['mu1_scaled']
    sigma_scaled = float(true_['sigma_scaled'])
    true_cate_scaled = true_['true_cate_scaled']
    QUERY_IDXS = list(map(int, true_['QUERY_IDXS']))
    REALIZATION = int(true_['realization']); N_CONTEXT = int(true_['n_context'])
    N_QUERIES = len(QUERY_IDXS); true_ate = float(true_cate_scaled.mean())

    # marginals figure — thin red curves, red dots
    n_rows, n_cols = _grid_layout(N_QUERIES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 3.6 * n_rows),
                              squeeze=False)
    for k, q in enumerate(QUERY_IDXS):
        ax = axes[k // n_cols][k % n_cols]
        ax.plot(centers, p_y0_t[k], color=COLOR_TRUE, lw=1.0, ls=':',
                 alpha=0.9, label=r'true $p(Y_{do0})$' if k == 0 else None)
        ax.plot(centers, p_y1_t[k], color=COLOR_TRUE, lw=1.0, ls='-',
                 alpha=0.9, label=r'true $p(Y_{do1})$' if k == 0 else None)
        ax.plot(float(mu0_scaled[q]), float(np.interp(mu0_scaled[q], centers, p_y0_t[k])),
                 'o', color=COLOR_TRUE, markersize=7, zorder=6)
        ax.plot(float(mu1_scaled[q]), float(np.interp(mu1_scaled[q], centers, p_y1_t[k])),
                 'o', color=COLOR_TRUE, markersize=7, zorder=6)
        ax.set_title(f'query {int(q)}   $\\tau_{{true}}$={float(true_cate_scaled[q]):+.2f}',
                      fontsize=10)
        if k // n_cols == n_rows - 1: ax.set_xlabel(r'$Y$  (scaled)')
        if k %  n_cols == 0:          ax.set_ylabel('density')
        ax.grid(alpha=0.25)
        if k == 0: ax.legend(fontsize=9, loc='upper right')
    for k in range(N_QUERIES, n_rows * n_cols):
        axes[k // n_cols][k % n_cols].set_visible(False)
    fig.suptitle(f'IHDP r={REALIZATION}   TRUE per-query marginals at N={N_CONTEXT}   '
                  f'(σ={sigma_scaled:.3f} scaled)',
                  fontsize=12, y=0.999)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    _save(fig, 'IHDP-TRUE-TE', 'ihdp_n10_marginals.png')

    # TE per query
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 3.6 * n_rows),
                              squeeze=False)
    for k, q in enumerate(QUERY_IDXS):
        ax = axes[k // n_cols][k % n_cols]
        E = float(mu1_scaled[q] - mu0_scaled[q])
        ax.fill_between(tau_centers, p_taus_t[k], alpha=0.20, color=COLOR_TRUE)
        ax.plot(tau_centers, p_taus_t[k], color=COLOR_TRUE, lw=2.0,
                 label=f'true $p(\\tau)$  E={E:+.2f}')
        ax.axvline(E, color=COLOR_TRUE, ls='--', lw=1.4, alpha=0.85)
        ax.plot(float(true_cate_scaled[q]), 0, 'o', color=COLOR_TRUE, markersize=9,
                 clip_on=False, zorder=6)
        ax.set_xlim(-1.5, 1.5)
        ax.set_title(f'query {int(q)}', fontsize=10)
        if k // n_cols == n_rows - 1: ax.set_xlabel(r'$\tau = Y_{do1} - Y_{do0}$  (scaled)')
        if k %  n_cols == 0:          ax.set_ylabel(r'$p(\tau)$')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc='upper right')
    for k in range(N_QUERIES, n_rows * n_cols):
        axes[k // n_cols][k % n_cols].set_visible(False)
    fig.suptitle(f'IHDP r={REALIZATION}   TRUE per-query TE distribution at N={N_CONTEXT}   '
                  f'(τ|x ~ N(μ₁−μ₀, 2σ²), σ={sigma_scaled:.3f} scaled)',
                  fontsize=12, y=0.999)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    _save(fig, 'IHDP-TRUE-TE', 'ihdp_n10_te.png')

    _ot_plot(p_taus_t, tau_centers, true_ate,
              f'IHDP r={REALIZATION}   TRUE OT aggregation at N={N_CONTEXT}',
              'IHDP-TRUE-TE')

# ── MARGINALS-COMPARE (Ours vs UWYK-NoAnc, + true overlay if avail) ─────
compare = _load('MARGINALS-COMPARE')
if compare is not None:
    centers = compare['centers']; bin_width = float(compare['bin_width'])
    p_y0_ours = compare['p_y0_ours']; p_y1_ours = compare['p_y1_ours']
    p_y0_uwyk = compare['p_y0_uwyk']; p_y1_uwyk = compare['p_y1_uwyk']
    true_cate_scaled = compare['true_cate_scaled']
    QUERY_IDXS = list(map(int, compare['QUERY_IDXS']))
    REALIZATION = int(compare['realization']); N_CONTEXT = int(compare['n_context'])
    N_QUERIES = len(QUERY_IDXS)

    # Optional true overlay from IHDP-TRUE-TE cache
    true_overlay = None
    if true_ is not None:
        # need p_y0_true / p_y1_true on the compare grid
        _t_centers = true_['centers']
        # if grid mismatches, interpolate
        if len(_t_centers) != len(centers) or float(_t_centers[0]) != float(centers[0]):
            p_y0_true_grid = np.stack([np.interp(centers, _t_centers, true_['p_y0_true'][k])
                                        for k in range(N_QUERIES)])
            p_y1_true_grid = np.stack([np.interp(centers, _t_centers, true_['p_y1_true'][k])
                                        for k in range(N_QUERIES)])
        else:
            p_y0_true_grid = true_['p_y0_true']
            p_y1_true_grid = true_['p_y1_true']
        true_overlay = dict(
            p_y0_true=p_y0_true_grid, p_y1_true=p_y1_true_grid,
            mu0_scaled=true_['mu0_scaled'], mu1_scaled=true_['mu1_scaled'],
            sigma_scaled=float(true_['sigma_scaled']),
        )

    n_rows, n_cols = _grid_layout(N_QUERIES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.6 * n_cols, 3.7 * n_rows),
                              squeeze=False)
    for k, q in enumerate(QUERY_IDXS):
        ax = axes[k // n_cols][k % n_cols]
        # Ours — solid + filled dot
        ax.plot(centers, p_y0_ours[k], color=COLOR_DO0, lw=1.9, ls='-',
                 label=r'$p(Y_{do0})$  Ours' if k == 0 else None)
        ax.plot(centers, p_y1_ours[k], color=COLOR_DO1, lw=1.9, ls='-',
                 label=r'$p(Y_{do1})$  Ours' if k == 0 else None)
        E_y0_o = float((centers * p_y0_ours[k]).sum() * bin_width)
        E_y1_o = float((centers * p_y1_ours[k]).sum() * bin_width)
        ax.plot(E_y0_o, float(np.interp(E_y0_o, centers, p_y0_ours[k])),
                 'o', color=COLOR_DO0, markersize=9, markeredgecolor='white',
                 markeredgewidth=1.0, zorder=6)
        ax.plot(E_y1_o, float(np.interp(E_y1_o, centers, p_y1_ours[k])),
                 'o', color=COLOR_DO1, markersize=9, markeredgecolor='white',
                 markeredgewidth=1.0, zorder=6)
        # UWYK-NoAnc — dashed + open dot
        ax.plot(centers, p_y0_uwyk[k], color=COLOR_DO0, lw=1.9, ls='--',
                 label=r'$p(Y_{do0})$  UWYK-NoAnc' if k == 0 else None)
        ax.plot(centers, p_y1_uwyk[k], color=COLOR_DO1, lw=1.9, ls='--',
                 label=r'$p(Y_{do1})$  UWYK-NoAnc' if k == 0 else None)
        E_y0_u = float((centers * p_y0_uwyk[k]).sum() * bin_width)
        E_y1_u = float((centers * p_y1_uwyk[k]).sum() * bin_width)
        ax.plot(E_y0_u, float(np.interp(E_y0_u, centers, p_y0_uwyk[k])),
                 'o', markerfacecolor='none', markeredgecolor=COLOR_DO0,
                 markersize=10, markeredgewidth=1.8, zorder=6)
        ax.plot(E_y1_u, float(np.interp(E_y1_u, centers, p_y1_uwyk[k])),
                 'o', markerfacecolor='none', markeredgecolor=COLOR_DO1,
                 markersize=10, markeredgewidth=1.8, zorder=6)
        # True — thin red + red dot
        if true_overlay is not None:
            p0t = true_overlay['p_y0_true'][k]
            p1t = true_overlay['p_y1_true'][k]
            ax.plot(centers, p0t, color=COLOR_TRUE, lw=0.9, ls=':', alpha=0.9,
                     label=r'true $p(Y_{do})$' if k == 0 else None)
            ax.plot(centers, p1t, color=COLOR_TRUE, lw=0.9, ls=':', alpha=0.9)
            mu0 = float(true_overlay['mu0_scaled'][q])
            mu1 = float(true_overlay['mu1_scaled'][q])
            ax.plot(mu0, float(np.interp(mu0, centers, p0t)),
                     'o', color=COLOR_TRUE, markersize=6, zorder=7)
            ax.plot(mu1, float(np.interp(mu1, centers, p1t)),
                     'o', color=COLOR_TRUE, markersize=6, zorder=7)
        ax.set_title(f'query {int(q)}   $\\tau_{{true}}$={float(true_cate_scaled[q]):+.2f}',
                      fontsize=10)
        if k // n_cols == n_rows - 1: ax.set_xlabel(r'$Y$  (scaled)')
        if k %  n_cols == 0:          ax.set_ylabel('density')
        ax.grid(alpha=0.25)
        if k == 0: ax.legend(fontsize=8, loc='upper right')
    for k in range(N_QUERIES, n_rows * n_cols):
        axes[k // n_cols][k % n_cols].set_visible(False)
    fig.suptitle(f'IHDP r={REALIZATION}   marginals comparison '
                  f'(Ours vs UWYK No-Ancestral{" vs true" if true_overlay is not None else ""}) '
                  f'at N={N_CONTEXT}',
                  fontsize=12, y=0.999)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    _save(fig, 'MARGINALS-COMPARE', 'ihdp_n10_marginals_compare.png')

# ── UWYK-2DMALC-VS-TRUE (Ours plots with the TRUE overlay) ──────────────
# Uses arrays already loaded from the UWYK-2DMALC and IHDP-TRUE-TE caches.
if ours is not None and true_ is not None:
    os.makedirs(os.path.join(_ROOT, 'UWYK-2DMALC-VS-TRUE'), exist_ok=True)

    p_mats           = ours['p_mats']
    centers_o        = ours['centers']
    bin_width_o      = float(ours['bin_width'])
    tau_centers_o    = ours['tau_centers']
    p_taus_o         = ours['p_taus']
    true_cate_scaled = ours['true_cate_scaled']
    QUERY_IDXS       = list(map(int, ours['QUERY_IDXS']))
    REALIZATION      = int(ours['realization'])
    N_CONTEXT        = int(ours['n_context'])
    N_QUERIES        = len(QUERY_IDXS)
    true_ate         = float(true_cate_scaled.mean())

    # marginals derived from p_mats — same as UWYK-2DMALC
    p_y0_ours = np.zeros((N_QUERIES, len(centers_o)))
    p_y1_ours = np.zeros((N_QUERIES, len(centers_o)))
    for k in range(N_QUERIES):
        m0 = p_mats[k].sum(axis=1); m1 = p_mats[k].sum(axis=0)
        p_y0_ours[k] = m0 / max(m0.sum() * bin_width_o, 1e-12)
        p_y1_ours[k] = m1 / max(m1.sum() * bin_width_o, 1e-12)

    # true arrays on the Ours grid
    _tc = true_['centers']
    if len(_tc) != len(centers_o) or float(_tc[0]) != float(centers_o[0]):
        p_y0_true_grid = np.stack([np.interp(centers_o, _tc, true_['p_y0_true'][k])
                                    for k in range(N_QUERIES)])
        p_y1_true_grid = np.stack([np.interp(centers_o, _tc, true_['p_y1_true'][k])
                                    for k in range(N_QUERIES)])
    else:
        p_y0_true_grid = true_['p_y0_true']
        p_y1_true_grid = true_['p_y1_true']
    mu0_scaled = true_['mu0_scaled']; mu1_scaled = true_['mu1_scaled']
    _ttc = true_['tau_centers']
    if len(_ttc) != len(tau_centers_o) or float(_ttc[0]) != float(tau_centers_o[0]):
        p_taus_true_grid = np.stack([np.interp(tau_centers_o, _ttc, true_['p_taus_true'][k])
                                      for k in range(N_QUERIES)])
    else:
        p_taus_true_grid = true_['p_taus_true']

    # -- marginals: Ours solid + filled dots, true thin dotted red + red dots
    _marginals_panel(
        centers_o, p_y0_ours, p_y1_ours, QUERY_IDXS, true_cate_scaled,
        f'IHDP r={REALIZATION}   Ours marginals vs TRUE at N={N_CONTEXT}',
        out_folder='UWYK-2DMALC-VS-TRUE',
        true_overlay=dict(
            p_y0_true=p_y0_true_grid, p_y1_true=p_y1_true_grid,
            mu0_scaled=mu0_scaled, mu1_scaled=mu1_scaled,
            sigma_scaled=float(true_['sigma_scaled']),
        ),
    )

    # -- per-query TE: Ours joint p(τ) filled + true p(τ) as thin red line
    n_rows, n_cols = _grid_layout(N_QUERIES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 3.6 * n_rows),
                              squeeze=False)
    tau_step = float(tau_centers_o[1] - tau_centers_o[0])
    for k, q in enumerate(QUERY_IDXS):
        ax = axes[k // n_cols][k % n_cols]
        p_o = p_taus_o[k]
        p_t = p_taus_true_grid[k]
        E_o = float((tau_centers_o * p_o).sum() * tau_step)
        E_t = float(mu1_scaled[q] - mu0_scaled[q])
        ax.fill_between(tau_centers_o, p_o, alpha=0.25, color=COLOR_JOINT)
        ax.plot(tau_centers_o, p_o, color=COLOR_JOINT, lw=2.0,
                 label=f'Ours joint $p(\\tau)$  E={E_o:+.2f}')
        ax.axvline(E_o, color=COLOR_JOINT, ls='--', lw=1.4, alpha=0.85)
        ax.plot(tau_centers_o, p_t, color=COLOR_TRUE, lw=1.0, ls=':', alpha=0.9,
                 label=f'true $p(\\tau)$  E={E_t:+.2f}')
        ax.plot(float(true_cate_scaled[q]), 0, 'o', color=COLOR_TRUE, markersize=9,
                 clip_on=False, zorder=6)
        ax.set_xlim(-1.0, 1.0)
        ax.set_title(f'query {int(q)}', fontsize=10)
        if k // n_cols == n_rows - 1: ax.set_xlabel(r'$\tau = Y_{do1} - Y_{do0}$  (scaled)')
        if k %  n_cols == 0:          ax.set_ylabel(r'$p(\tau)$')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc='upper right')
    for k in range(N_QUERIES, n_rows * n_cols):
        axes[k // n_cols][k % n_cols].set_visible(False)
    fig.suptitle(f'IHDP r={REALIZATION}   Ours joint p(τ) vs TRUE at N={N_CONTEXT}',
                  fontsize=12, y=0.999)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    _save(fig, 'UWYK-2DMALC-VS-TRUE', 'ihdp_n10_te.png')

    # -- OT: Ours' W2 barycenter, true's W2 barycenter overlay
    bary_ours = w2_barycenter_1d(p_taus_o, tau_centers_o)
    bary_ours /= max(bary_ours.sum() * tau_step, 1e-12)
    bary_true = w2_barycenter_1d(p_taus_true_grid, tau_centers_o)
    bary_true /= max(bary_true.sum() * tau_step, 1e-12)
    mode_o = float(tau_centers_o[int(np.argmax(bary_ours))])
    mode_t = float(tau_centers_o[int(np.argmax(bary_true))])
    per_q_o = (tau_centers_o[None, :] * p_taus_o).sum(axis=1) * tau_step
    mom_o = float(per_q_o.mean())

    fig, ax = plt.subplots(figsize=(9, 4.6))
    palette_Q = plt.cm.tab10(np.linspace(0, 0.9, N_QUERIES))
    for k in range(N_QUERIES):
        ax.plot(tau_centers_o, p_taus_o[k], color=palette_Q[k], lw=1.0, alpha=0.30)
    ax.fill_between(tau_centers_o, bary_ours, alpha=0.20, color=COLOR_OT)
    ax.plot(tau_centers_o, bary_ours, color=COLOR_OT, lw=2.6,
             label=f'Ours OT (mode={mode_o:+.2f})')
    ax.plot(tau_centers_o, bary_true, color=COLOR_TRUE, lw=1.6, ls=':',
             label=f'true OT (mode={mode_t:+.2f})')
    ax.axvline(true_ate, color=COLOR_TRUE_ATE, ls='--', lw=1.6,
                label=f'true population ATE = {true_ate:+.2f}')
    ax.axvline(mode_o, color=COLOR_OT, ls='--', lw=1.6)
    ax.axvline(mom_o, color=COLOR_MEAN_MEANS, ls='--', lw=1.6,
                label=f'Ours mean-of-means = {mom_o:+.2f}')
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel(r'$\tau = Y_{do1} - Y_{do0}$  (scaled)')
    ax.set_ylabel('density')
    ax.set_title(f'IHDP r={REALIZATION}   Ours OT vs TRUE at N={N_CONTEXT}',
                  fontsize=12)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, 'UWYK-2DMALC-VS-TRUE', 'ihdp_n10_ot.png')

print('\n[done]')
