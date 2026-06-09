"""Runnable example for MALC_2D.

Builds three illustrative scenarios — a unimodal bivariate normal, a 2D
bimodal mixture, and a 2D trimodal mixture — and fits each with the default
two-stage pipeline (B_select=B_fit=500, max_K=5).

Each scenario produces a contour plot saved as PNG.

Run:
    python example.py

This assumes `log_concave_2d_fast.py` and `malc_2d.py` are on the Python path
(either in the same directory or installed as a package). The simplest setup
is to drop them next to this script.
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np

from malc_2d import MALC_2D, dmalc_2d


def _phi(z):
    return np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)


def _dnorm(x, mu=0.0, sd=1.0):
    return _phi((x - mu) / sd) / sd


def make_p_mat(true_pdf, grid_x, grid_y, n_sub: int = 10) -> np.ndarray:
    """Build a binned probability matrix by integrating `true_pdf` over each bin."""
    n_x = len(grid_x) - 1
    n_y = len(grid_y) - 1
    pm = np.zeros((n_y, n_x))
    for i in range(n_y):
        for j in range(n_x):
            xs = np.linspace(grid_x[j], grid_x[j + 1], n_sub)
            ys = np.linspace(grid_y[i], grid_y[i + 1], n_sub)
            XX, YY = np.meshgrid(xs, ys, indexing="xy")
            pts = np.column_stack([XX.ravel(), YY.ravel()])
            pm[i, j] = true_pdf(pts).mean() * (grid_x[j + 1] - grid_x[j]) * (grid_y[i + 1] - grid_y[i])
    return pm / pm.sum()


def plot_fit_vs_true(fit, true_pdf, name, savepath, n_eval=80):
    """Side-by-side contour: estimate vs true density."""
    xs = np.linspace(fit.grid_x.min(), fit.grid_x.max(), n_eval)
    ys = np.linspace(fit.grid_y.min(), fit.grid_y.max(), n_eval)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    Z_est = dmalc_2d(fit, pts).reshape(n_eval, n_eval)
    Z_true = true_pdf(pts).reshape(n_eval, n_eval)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    l2 = float(np.sqrt(((Z_est - Z_true) ** 2 * dx * dy).sum()))
    levels = np.linspace(0, max(Z_true.max(), Z_est.max()), 11)[1:]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].contour(XX, YY, Z_est, levels=levels, colors="navy", linewidths=1.5)
    axes[0].set_title(f"{name}\nMALC_2D estimate (K={fit.K}, L2={l2:.4f})")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y"); axes[0].set_aspect("equal")
    axes[1].contour(XX, YY, Z_true, levels=levels, colors="firebrick", linewidths=1.5)
    axes[1].set_title("True density")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y"); axes[1].set_aspect("equal")
    fig.tight_layout()
    fig.savefig(savepath, dpi=120)
    plt.close(fig)
    return l2


# -------- Three example scenarios --------


def example_1_unimodal():
    """Bivariate standard normal N((0,0), I). True K=1."""
    grid_x = np.arange(-4, 4 + 1e-9, 0.5)
    grid_y = np.arange(-4, 4 + 1e-9, 0.5)
    pdf = lambda pts: _dnorm(pts[:, 0]) * _dnorm(pts[:, 1])
    p_mat = make_p_mat(pdf, grid_x, grid_y)

    print("\n=== Example 1: Bivariate Normal N((0,0), I), true K=1 ===")
    t0 = time.time()
    fit = MALC_2D(p_mat, grid_x, grid_y, seed=20180621, verbose=False)
    print(f"  Selected K={fit.K}  pi={np.round(fit.pi, 3)}  time={time.time()-t0:.1f}s")
    print(f"  BIC table: {fit.bic_table}")
    l2 = plot_fit_vs_true(fit, pdf, "Bivariate Normal", "example_1_normal.png")
    print(f"  L2 = {l2:.4f}  → saved example_1_normal.png")


def example_2_bimodal():
    """Bimodal: 0.5 N((-2,-2), 0.6²·I) + 0.5 N((2,2), 0.6²·I). True K=2."""
    grid_x = np.arange(-5, 5 + 1e-9, 0.5)
    grid_y = np.arange(-5, 5 + 1e-9, 0.5)
    pdf = lambda pts: (
        0.5 * _dnorm(pts[:, 0], -2, 0.6) * _dnorm(pts[:, 1], -2, 0.6)
        + 0.5 * _dnorm(pts[:, 0], 2, 0.6) * _dnorm(pts[:, 1], 2, 0.6)
    )
    p_mat = make_p_mat(pdf, grid_x, grid_y)

    print("\n=== Example 2: 2D Bimodal Mixture, true K=2 ===")
    t0 = time.time()
    fit = MALC_2D(p_mat, grid_x, grid_y, seed=20180621, verbose=False)
    print(f"  Selected K={fit.K}  pi={np.round(fit.pi, 3)}  time={time.time()-t0:.1f}s")
    print(f"  BIC table: {fit.bic_table}")
    l2 = plot_fit_vs_true(fit, pdf, "2D Bimodal", "example_2_bimodal.png")
    print(f"  L2 = {l2:.4f}  → saved example_2_bimodal.png")


def example_3_trimodal():
    """Trimodal: three normals at (0,3), (-2.6,-1.5), (2.6,-1.5), all sd=0.6. True K=3."""
    grid_x = np.arange(-5, 5 + 1e-9, 0.5)
    grid_y = np.arange(-5, 5 + 1e-9, 0.5)
    pdf = lambda pts: (
        _dnorm(pts[:, 0], 0, 0.6) * _dnorm(pts[:, 1], 3, 0.6)
        + _dnorm(pts[:, 0], -2.6, 0.6) * _dnorm(pts[:, 1], -1.5, 0.6)
        + _dnorm(pts[:, 0], 2.6, 0.6) * _dnorm(pts[:, 1], -1.5, 0.6)
    ) / 3.0
    p_mat = make_p_mat(pdf, grid_x, grid_y)

    print("\n=== Example 3: 2D Trimodal Mixture, true K=3 ===")
    t0 = time.time()
    fit = MALC_2D(p_mat, grid_x, grid_y, seed=20180621, verbose=False)
    print(f"  Selected K={fit.K}  pi={np.round(fit.pi, 3)}  time={time.time()-t0:.1f}s")
    print(f"  BIC table: {fit.bic_table}")
    l2 = plot_fit_vs_true(fit, pdf, "2D Trimodal", "example_3_trimodal.png")
    print(f"  L2 = {l2:.4f}  → saved example_3_trimodal.png")


if __name__ == "__main__":
    example_1_unimodal()
    example_2_bimodal()
    example_3_trimodal()
    print("\nDone. Three PNG files saved in the current directory.")
