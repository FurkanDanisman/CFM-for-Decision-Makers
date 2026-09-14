"""Plot the OBSERVATIONAL distributions of one case-study dataset.

What a model actually sees: each covariate X_j, the treatment T, and the
outcome Y split by arm. `--with-truth` adds the counterfactual quantities
(mu_0, mu_1, CATE) that only the generator knows.

Y is drawn per ARM rather than pooled, because that split is the whole point:
under confounding the two arms differ in BOTH the outcome and the covariates
that selected into them, and a single pooled histogram hides it.

    python case_study/plot_dataset.py --root case_study/d_variation/shift+2/d3 \
        --case Observed_Confounder --n 1000 --r 0 --out dataset_d3.png
    python case_study/plot_dataset.py <npz> --with-truth --out x.png
"""
from __future__ import annotations

import argparse
import glob
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Same palette as case_study/cate_distributions/plot_distributions.py.
C0 = "#E8820C"        # arm 0 / primary fill (orange)
C1 = "#1B5E20"        # arm 1 / outlined series (dark green)
INK = "#222222"
MUTED = "#8A8A8A"


def _panel(ax, title):
    ax.set_title(title, color=INK, fontsize=10, pad=6)
    ax.set_yticks([])
    ax.tick_params(colors=INK, labelsize=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)


def plot(path, out, bins=40, max_cols=12, with_truth=False):
    z = np.load(path, allow_pickle=True)
    names = [str(x) for x in z['feature_names']]
    X, T, Y = z['X'], z['T'].reshape(-1), z['Y'].reshape(-1)
    t0, t1 = T < 0.5, T >= 0.5

    shown = names[:max_cols]
    trunc = len(names) - len(shown)
    panels = [("cov", j, n) for j, n in enumerate(shown)]
    panels += [("T", None, "T"), ("Y", None, "Y  (by arm)")]
    if with_truth:
        panels += [("mu", None, "mu_0 / mu_1"), ("cate", None, "CATE")]

    ncol = min(4, len(panels))
    nrow = math.ceil(len(panels) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 2.4 * nrow),
                             squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)

    for k, (kind, j, label) in enumerate(panels):
        ax = axes[k // ncol][k % ncol]
        ax.set_visible(True)
        if kind == "cov":
            x = X[:, j]
            edges = np.linspace(x.min(), x.max(), bins + 1)
            # split by arm: under confounding the treated and control
            # covariate distributions differ, and that IS the confounding.
            ax.hist(x[t0], bins=edges, color=C0, alpha=0.55, edgecolor="none",
                    label="T=0")
            ax.hist(x[t1], bins=edges, histtype="step", color=C1, lw=1.8,
                    label="T=1")
            _panel(ax, f"{label}   (sd={x.std():.2f})")
        elif kind == "T":
            ax.bar([0, 1], [int(t0.sum()), int(t1.sum())], width=0.6,
                   color=[C0, C1])
            ax.set_xticks([0, 1])
            _panel(ax, f"{label}   (treated {T.mean():.1%})")
        elif kind == "Y":
            edges = np.linspace(Y.min(), Y.max(), bins + 1)
            ax.hist(Y[t0], bins=edges, color=C0, alpha=0.55, edgecolor="none",
                    label="Y | T=0")
            ax.hist(Y[t1], bins=edges, histtype="step", color=C1, lw=1.8,
                    label="Y | T=1")
            _panel(ax, f"{label}   (mean {Y[t0].mean():+.2f} / {Y[t1].mean():+.2f})")
        elif kind == "mu":
            m0, m1 = z['mu_0'].reshape(-1), z['mu_1'].reshape(-1)
            edges = np.linspace(min(m0.min(), m1.min()),
                                max(m0.max(), m1.max()), bins + 1)
            ax.hist(m0, bins=edges, color=C0, alpha=0.55, edgecolor="none")
            ax.hist(m1, bins=edges, histtype="step", color=C1, lw=1.8)
            _panel(ax, label)
        else:
            c = z['cate'].reshape(-1)
            ax.hist(c, bins=bins, color=C0, alpha=0.55, edgecolor="none")
            ax.axvline(float(c.mean()), color=C1, lw=2.0)
            _panel(ax, f"{label}   (ATE {c.mean():+.3f})")

    h, l = axes[0][0].get_legend_handles_labels()
    if h:
        fig.legend(h, ["T = 0", "T = 1"], loc="upper center", ncol=2,
                   frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.01),
                   labelcolor=INK)
    sub = (f"{str(z['case_study'])}   d={X.shape[1]}   N={int(z['n_context'])}"
           f"   beta={float(z['cate_shift']):+g}   r={int(z['seed'])%10000}")
    if trunc:
        sub += f"   (first {len(shown)} of {len(names)} covariates)"
    fig.suptitle(sub, y=1.03, fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"[plot] {out}")
    if trunc:
        print(f"[plot] {trunc} covariates not shown (--max-cols to raise)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz', nargs='?')
    ap.add_argument('--root')
    ap.add_argument('--case', default='Observed_Confounder')
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--r', type=int, default=0)
    ap.add_argument('--bins', type=int, default=40)
    ap.add_argument('--max-cols', type=int, default=12)
    ap.add_argument('--with-truth', action='store_true')
    ap.add_argument('--out', default='dataset.png')
    a = ap.parse_args()
    path = a.npz
    if not path:
        if not a.root:
            ap.error('give an npz path or --root')
        path = os.path.join(a.root, a.case, f'N{a.n}', f'{a.case}_{a.r}.npz')
        if not os.path.isfile(path):
            hits = glob.glob(os.path.join(a.root, a.case, f'N{a.n}', '*.npz'))
            ap.error(f'{path} not found ({len(hits)} realizations there)')
    plot(path, a.out, a.bins, a.max_cols, a.with_truth)


if __name__ == '__main__':
    main()
