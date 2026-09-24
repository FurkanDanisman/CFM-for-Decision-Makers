#!/usr/bin/env python
"""ATE density illustration: 2 x 4, one column per 1D/2D architecture pair.

Each panel shows, for one model on ONE realization:
    faint      p(tau | x_q) for every query q          -- the per-query CATE densities
    bold       p(ATE), their 1D 2-Wasserstein barycenter
    dashed     the true ATE

Top row is the 1D member of each pair, bottom row its 2D partner, so a column is
the comparison and a row is the head type.

The per-query curves are the point of the figure: the ATE density is narrow not
because any single query is known precisely but because averaging over queries
concentrates it, and showing only the barycenter hides that.

Densities and the barycenter come from compute_ate_density_w2_cell, not from a
reimplementation here: that module already converts each model's own bin grid to
RAW Y units (tau_raw = tau_scaled * y_scale, density / (bin_width * y_scale)), and
getting that conversion subtly wrong would misplace whole curves while still
looking plausible.

    python benchmarks/plots/plot_ate_density_2x4.py \
        --root $SCRATCH/ihdp_dens_8models --dataset IHDP --realization 0 \
        --repo $PWD/R-PFN --out $SCRATCH/ihdp_r0_ate_2x4.png
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, "realcause_eval"))
from compute_ate_density_w2_cell import (                      # noqa: E402
    _import_barycenter, _load_ptau_raw,
)

C_1D, C_2D = "#D97706", "#8C2F39"

# (column label, 1D model dirs, 1D tag, 2D model dirs, 2D tag)
#
# The anc mode is a KEY SUFFIX inside the npz, not a directory: one uwyk1d dump carries
# p_y0_scaled_noanc beside p_y0_scaled_v3a, and graph2d is the same.
#
# Several directory candidates per slot because the dump driver names a directory after
# its harness, not after the checkpoint: Do-PFN 2D is dopfn_repro_joint2d, but running
# it through submit_rc_density puts it under whichever slot carried the checkpoint. The
# first directory present wins, so the figure does not depend on that accident.
PAIRS = [
    ("Do-PFN",      ("dopfn_native",), None,
     ("dopfn_repro_joint2d", "dopfn_joint2d", "dopfn_bb"), None),
    ("UWYK No-Anc", ("uwyk1d",), "noanc", ("graph2d",), "noanc"),
    ("UWYK Anc",    ("uwyk1d",), "v3a",   ("graph2d",), "v3a"),
    ("CausalPFN-C", ("cpfn1d_j1024", "cpfn1d"), None,
     ("cpfn2d_eta0", "cpfn2d"), None),
]


def _find(root, models, dataset, r):
    """The realization-r density npz under <root>/<model>/<dataset>/.

    `models` is a tuple of candidate directory names; the first that exists wins.
    """
    d = next((os.path.join(root, m, dataset) for m in models
              if os.path.isdir(os.path.join(root, m, dataset))), None)
    if d is None:
        return None
    for pat in (f"*r{r:03d}*.npz", f"*r{r}.npz", "*.npz"):
        hits = sorted(f for f in glob.glob(os.path.join(d, pat))
                      if os.path.basename(f) != "summary.npz")
        if hits:
            return hits[min(r, len(hits) - 1)] if pat == "*.npz" else hits[0]
    return None


def _panel(ax, path, source, tag, colour, bary, n_bg, label):
    if path is None:
        ax.text(0.5, 0.5, "no dump", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="0.5")
        ax.set_xticks([]); ax.set_yticks([])
        return None
    try:
        p_tau, tau, true_cate, _ys, _yh = _load_ptau_raw(path, source, tag=tag)
    except TypeError:
        # older signature without tag: the anc mode then cannot be selected, and
        # silently plotting the un-suffixed key would mislabel the panel.
        if tag:
            ax.text(0.5, 0.5, f"need tag={tag}\n(loader too old)", ha="center",
                    va="center", transform=ax.transAxes, fontsize=8, color="0.5")
            ax.set_xticks([]); ax.set_yticks([])
            return None
        p_tau, tau, true_cate, _ys, _yh = _load_ptau_raw(path, source)

    dt = float(tau[1] - tau[0])
    p_tau = p_tau / (p_tau.sum(axis=-1, keepdims=True) * dt).clip(min=1e-12)

    # Background: a sample of the per-query densities. All of them at a few hundred
    # queries is an ink blot; a fixed stride keeps the spread visible.
    idx = np.linspace(0, p_tau.shape[0] - 1, min(n_bg, p_tau.shape[0])).astype(int)
    for i in idx:
        ax.plot(tau, p_tau[i], color=colour, alpha=0.12, lw=0.6, zorder=1)

    p_ate = bary(p_tau, tau, n_tau=4001)
    s = p_ate.sum() * dt
    if s > 0:
        p_ate = p_ate / s
    ax.plot(tau, p_ate, color=colour, lw=2.0, zorder=3)

    true_ate = float(np.asarray(true_cate).mean())
    ax.axvline(true_ate, color="black", ls="--", lw=1.0, zorder=4)

    ax.set_title(label, fontsize=9)
    ax.tick_params(labelsize=7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return true_ate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="OUT_ROOT of submit_rc_density")
    ap.add_argument("--dataset", default="IHDP")
    ap.add_argument("--realization", type=int, default=0)
    ap.add_argument("--repo", default=_REPO)
    ap.add_argument("--n-background", type=int, default=60,
                    help="how many per-query curves to draw behind the ATE")
    ap.add_argument("--xlim", nargs=2, type=float, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    bary = _import_barycenter(a.repo)
    fig, axes = plt.subplots(2, len(PAIRS), figsize=(3.0 * len(PAIRS), 4.6),
                             squeeze=False)
    for ci, (lab, m1, t1, m2, t2) in enumerate(PAIRS):
        for ri, (m, tg, src, colour, suffix) in enumerate((
                (m1, t1, "marginals", C_1D, ""),
                (m2, t2, "joint", C_2D, " 2D"))):
            ax = axes[ri][ci]
            path = _find(a.root, m, a.dataset, a.realization)
            _panel(ax, path, src, tg, colour, bary, a.n_background, lab + suffix)
            if a.xlim:
                ax.set_xlim(*a.xlim)
            if ci == 0:
                ax.set_ylabel("density", fontsize=8)
            if ri == 1:
                ax.set_xlabel("ATE / CATE (raw)", fontsize=8)
    fig.suptitle(f"{a.dataset}, realization {a.realization} — per-query CATE "
                 f"(faint) and ATE (bold); dashed = true ATE", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
