"""Plot the 4 analytical truth densities for each of DoPFN's 6 case studies.

For each realization r and query i we have (from the regenerated pkls):
    mu_0[i], mu_1[i]  : structural means with noise held at 0
    sigma_eps         : Y-noise std
    rho_y_noise       : correlation between eps_obs and eps_int

Analytical truth densities (all Gaussian mixtures over queries × realizations):

  p(Y | do(0)) = (1/M) Σ_{r,i} N(mu_0^{(r)}[i], sigma_eps^2)
  p(Y | do(1)) = (1/M) Σ_{r,i} N(mu_1^{(r)}[i], sigma_eps^2)
  p(tau)       = (1/M) Σ_{r,i} N(mu_1^{(r)}[i] - mu_0^{(r)}[i], 2·sigma_eps^2·(1-rho))
  p(ATE)       = KDE over the R per-realization sample means (1/N) Σ_i (mu_1 - mu_0)

Output: 6 rows (case studies) × 4 cols (Ydo0, Ydo1, CATE, ATE) figure.

Usage:
    python plot_true_densities.py \
        --pkl-root   /scratch/.../external/dopfn/data/prior_sampling_rho02 \
        --dopfn-root /scratch/.../external/dopfn \
        --out        /scratch/.../results_density_rho02/true_densities.pdf \
        [--n-max-per-case 100]

`--dopfn-root` (or the DOPFN_ROOT env var) must point at the dopfn_upstream
repo root so the unpickler can import DoPFN's InterventionalDataset class.
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CASE_STUDIES = [
    "Observed_Confounder",
    "Observed_Mediator",
    "Observed_Mediator_and_Confounder",
    "Unobserved_Confounder",
    "Frontdoor_Criterion",
    "Backdoor_Criterion",
]


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pkl-root", required=True,
                   help="Directory containing <CASE>/*.pkl regenerated pkls.")
    p.add_argument("--dopfn-root", default=os.environ.get("DOPFN_ROOT", ""),
                   help="Path to dopfn_upstream repo root (has priors/, datasets/). "
                        "Falls back to $DOPFN_ROOT env var. Required so pickle can "
                        "import InterventionalDataset when loading the pkls.")
    p.add_argument("--out", required=True,
                   help="Output PDF path.")
    p.add_argument("--n-max-per-case", type=int, default=100,
                   help="Cap on # realizations loaded per case (for speed).")
    p.add_argument("--grid-n", type=int, default=1000,
                   help="Grid resolution for smooth density curves.")
    p.add_argument("--grid-pad-sigmas", type=float, default=4.0,
                   help="Grid extends this many sigmas past the data range.")
    return p.parse_args()


def _install_dopfn_paths(dopfn_root: str) -> None:
    if not dopfn_root:
        print("[warn] no --dopfn-root and no $DOPFN_ROOT — pkl unpickling will "
              "fail with `No module named 'datasets'`.", file=sys.stderr)
        return
    if not os.path.isdir(dopfn_root):
        print(f"[warn] --dopfn-root does not exist: {dopfn_root}", file=sys.stderr)
        return
    if dopfn_root not in sys.path:
        sys.path.insert(0, dopfn_root)
    # The `datasets` package is at $DOPFN_ROOT/datasets/. Sanity import.
    try:
        import datasets as _dopfn_datasets  # noqa: F401
    except Exception as e:
        print(f"[warn] failed to import DoPFN's `datasets` from {dopfn_root}: {e}",
              file=sys.stderr)


def _gaussian_mixture_pdf(centers: np.ndarray, sigma: float,
                          x_grid: np.ndarray) -> np.ndarray:
    """Evaluate (1/N) Σ_i N(center_i, sigma^2) on x_grid, vectorised."""
    # x_grid: (G,) ; centers: (N,) → pairwise (N, G).
    diff = (x_grid[None, :] - centers[:, None]) / sigma
    logp = -0.5 * diff * diff - 0.5 * np.log(2 * np.pi * sigma * sigma)
    return np.exp(logp).mean(axis=0)


def _kde_pdf(samples: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Silverman-bandwidth Gaussian KDE, self-contained (no scipy dep)."""
    n = len(samples)
    if n < 2:
        return np.zeros_like(x_grid)
    std = float(np.std(samples, ddof=1))
    if std <= 0:
        # Degenerate — plot a narrow spike so it is still visible.
        std = max(1e-6, 0.01 * float(np.abs(samples).mean() or 1.0))
    h = 1.06 * std * n ** (-1 / 5)
    diff = (x_grid[None, :] - samples[:, None]) / h
    kern = np.exp(-0.5 * diff * diff) / np.sqrt(2 * np.pi)
    return kern.mean(axis=0) / h


def _load_case(pkl_dir: str, n_max: int):
    """Return (mu0_all, mu1_all, ate_r, sigma_eps, rho) from all realizations in pkl_dir."""
    paths = sorted(glob.glob(os.path.join(pkl_dir, "*.pkl")))[:n_max]
    if not paths:
        return None
    mu0_all, mu1_all, ate_r = [], [], []
    sigma_eps_r, rho_r = [], []
    for p in paths:
        try:
            with open(p, "rb") as f:
                ds = pickle.load(f)
        except Exception as e:
            print(f"[warn] failed to load {p}: {e}", file=sys.stderr)
            continue
        # Skip pkls without the extended fields (i.e. the ORIGINAL DoPFN pkls).
        if not hasattr(ds, "mu_0_per_query"):
            continue
        m0 = np.asarray(ds.mu_0_per_query, dtype=np.float64)
        m1 = np.asarray(ds.mu_1_per_query, dtype=np.float64)
        mu0_all.append(m0)
        mu1_all.append(m1)
        ate_r.append((m1 - m0).mean())
        sigma_eps_r.append(float(getattr(ds, "sigma_eps", np.nan)))
        rho_r.append(float(getattr(ds, "rho_y_noise", 0.0)))
    if not mu0_all:
        return None
    return {
        "mu0": np.concatenate(mu0_all),
        "mu1": np.concatenate(mu1_all),
        "ate_r": np.array(ate_r),
        "sigma_eps": float(np.nanmean(sigma_eps_r)),
        "rho": float(np.nanmean(rho_r)),
        "n_reals": len(mu0_all),
        "n_query_per_real": len(mu0_all[0]),
    }


def _panel(ax, x, pdf, title: str, xlabel: str, color: str = None):
    if color is None:
        ax.plot(x, pdf, linewidth=1.2)
        ax.fill_between(x, 0, pdf, alpha=0.15)
    else:
        ax.plot(x, pdf, linewidth=1.2, color=color)
        ax.fill_between(x, 0, pdf, alpha=0.15, color=color)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("density", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, linewidth=0.5)


def main():
    args = _parse_args()
    _install_dopfn_paths(args.dopfn_root)

    fig, axes = plt.subplots(len(CASE_STUDIES), 4,
                             figsize=(4 * 3.2, len(CASE_STUDIES) * 2.3),
                             squeeze=False)

    for row, case in enumerate(CASE_STUDIES):
        pkl_dir = os.path.join(args.pkl_root, case)
        data = _load_case(pkl_dir, args.n_max_per_case)
        if data is None:
            for c in range(4):
                axes[row, c].text(0.5, 0.5, f"no pkls in\n{case}",
                                  ha="center", va="center",
                                  transform=axes[row, c].transAxes,
                                  fontsize=9)
                axes[row, c].axis("off")
            continue

        mu0, mu1 = data["mu0"], data["mu1"]
        sigma_eps = data["sigma_eps"]
        rho = data["rho"]
        ate_r = data["ate_r"]

        tau_center = mu1 - mu0
        sigma_tau = float(np.sqrt(2.0 * sigma_eps * sigma_eps * (1.0 - rho)))

        # Y grid: pool mu0/mu1 range + sigma_eps padding.
        y_lo = min(mu0.min(), mu1.min()) - args.grid_pad_sigmas * sigma_eps
        y_hi = max(mu0.max(), mu1.max()) + args.grid_pad_sigmas * sigma_eps
        y_grid = np.linspace(y_lo, y_hi, args.grid_n)

        # τ grid.
        tau_lo = tau_center.min() - args.grid_pad_sigmas * max(sigma_tau, 1e-6)
        tau_hi = tau_center.max() + args.grid_pad_sigmas * max(sigma_tau, 1e-6)
        tau_grid = np.linspace(tau_lo, tau_hi, args.grid_n)

        # ATE grid.
        ate_span = float(ate_r.max() - ate_r.min()) or 1e-3
        ate_lo = ate_r.min() - 0.5 * ate_span
        ate_hi = ate_r.max() + 0.5 * ate_span
        ate_grid = np.linspace(ate_lo, ate_hi, args.grid_n)

        # PDFs.
        p_ydo0 = _gaussian_mixture_pdf(mu0, sigma_eps, y_grid)
        p_ydo1 = _gaussian_mixture_pdf(mu1, sigma_eps, y_grid)
        p_cate = _gaussian_mixture_pdf(tau_center,
                                       max(sigma_tau, 1e-6), tau_grid)
        p_ate  = _kde_pdf(ate_r, ate_grid)

        subtitle = (f"{case}   "
                    f"σε={sigma_eps:.3f}  ρ={rho:.2f}  "
                    f"R={data['n_reals']}  N/real={data['n_query_per_real']}")

        _panel(axes[row, 0], y_grid,   p_ydo0, f"p(Y | do(0))", "Y", "C0")
        _panel(axes[row, 1], y_grid,   p_ydo1, f"p(Y | do(1))", "Y", "C1")
        _panel(axes[row, 2], tau_grid, p_cate, f"p(τ)  (all queries)", "τ", "C2")
        _panel(axes[row, 3], ate_grid, p_ate,  f"p(ATE) (across {data['n_reals']} reals)", "ATE", "C3")

        # Attach case label + summary to the leftmost axis.
        axes[row, 0].text(-0.32, 0.5, subtitle, transform=axes[row, 0].transAxes,
                           fontsize=8, rotation=90, va="center", ha="right")

    fig.suptitle("Analytical truth densities (regenerated pkls, ρ Y-noise correlation)",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=[0.03, 0, 1, 0.98])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"[ok] wrote {args.out}")


if __name__ == "__main__":
    main()
