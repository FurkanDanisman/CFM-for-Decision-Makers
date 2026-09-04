"""Plot the 4 analytical truth densities for each of DoPFN's 6 case studies.

Per realization we compute a per-realization Y-scale
    y_shift_r = (y_int.max() + y_int.min()) / 2
    y_scale_r = (y_int.max() - y_int.min()) / 2   (matches the eval convention)
and STANDARDIZE mu_0, mu_1, sigma_eps into the [-1, +1]-normalised space that
all downstream models actually see at eval time. Pooling standardized values
across realizations gives comparable-scale mixtures and prevents a handful of
SCM-blowup realizations from dominating the axis.

Densities (Gaussian mixtures over queries × realizations, all on standardized Y):

  p(Ỹ | do(0)) = (1/M) Σ_{r,i} N(mu_0_scaled^{(r)}[i], sigma_eps_scaled^{(r)}²)
  p(Ỹ | do(1)) = (1/M) Σ_{r,i} N(mu_1_scaled^{(r)}[i], sigma_eps_scaled^{(r)}²)
  p(τ̃)        = (1/M) Σ_{r,i} N(mu_1_scaled-mu_0_scaled, 2·σ²·(1-ρ))
  p(ATẼ)      = KDE over the R per-realization sample means (standardized)

Axis limits use robust 1st–99th percentiles to trim SCM-blowup tails.

Usage:
    python plot_true_densities.py \
        --pkl-root   /scratch/.../external/dopfn/data/prior_sampling_rho02 \
        --dopfn-root /scratch/.../external/dopfn \
        --out        /scratch/.../results_density_rho02/true_densities.pdf

`--dopfn-root` (or $DOPFN_ROOT) must point at dopfn_upstream so pickle can
import InterventionalDataset when loading the pkls.
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
                   help="Path to dopfn_upstream root (needs `datasets` package). "
                        "Falls back to $DOPFN_ROOT env var.")
    p.add_argument("--out", required=True,
                   help="Output PDF path.")
    p.add_argument("--n-max-per-case", type=int, default=100,
                   help="Cap on # realizations loaded per case.")
    p.add_argument("--grid-n", type=int, default=1000,
                   help="Grid resolution for density curves.")
    p.add_argument("--axis-lo-pct", type=float, default=1.0,
                   help="Percentile for x-axis lower bound (default 1).")
    p.add_argument("--axis-hi-pct", type=float, default=99.0,
                   help="Percentile for x-axis upper bound (default 99).")
    p.add_argument("--outlier-clip-sigmas", type=float, default=0.0,
                   help="If >0, drop realizations whose |mu_scaled|.max() > this "
                        "many standard-deviations. Default 0 = keep all.")
    return p.parse_args()


def _install_dopfn_paths(dopfn_root: str) -> None:
    if not dopfn_root:
        print("[warn] no --dopfn-root and no $DOPFN_ROOT — pkl unpickling will fail.",
              file=sys.stderr)
        return
    if not os.path.isdir(dopfn_root):
        print(f"[warn] --dopfn-root does not exist: {dopfn_root}", file=sys.stderr)
        return
    if dopfn_root not in sys.path:
        sys.path.insert(0, dopfn_root)
    try:
        import datasets as _dopfn_datasets  # noqa: F401
    except Exception as e:
        print(f"[warn] failed to import DoPFN's `datasets` from {dopfn_root}: {e}",
              file=sys.stderr)


def _gaussian_mixture_pdf(centers: np.ndarray, sigmas: np.ndarray,
                          x_grid: np.ndarray) -> np.ndarray:
    """Evaluate (1/N) Σ_i N(center_i, sigma_i^2) on x_grid.

    centers, sigmas both shape (N,); returns pdf on x_grid shape (G,).
    """
    # Broadcast-friendly (N, G) computation, chunked to stay under ~200 MB.
    N, G = len(centers), len(x_grid)
    if N == 0:
        return np.zeros_like(x_grid)
    chunk = max(1, min(N, int(2e7 / G)))
    out = np.zeros(G, dtype=np.float64)
    for start in range(0, N, chunk):
        c = centers[start:start + chunk][:, None]      # (n, 1)
        s = sigmas [start:start + chunk][:, None]      # (n, 1)
        diff = (x_grid[None, :] - c) / s               # (n, G)
        logp = -0.5 * diff * diff - 0.5 * np.log(2 * np.pi * s * s)
        out += np.exp(logp).sum(axis=0)
    return out / N


def _kde_pdf(samples: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    n = len(samples)
    if n < 2:
        return np.zeros_like(x_grid)
    std = float(np.std(samples, ddof=1))
    if std <= 0:
        std = max(1e-6, 0.01 * float(np.abs(samples).mean() or 1.0))
    h = 1.06 * std * n ** (-1 / 5)
    diff = (x_grid[None, :] - samples[:, None]) / h
    kern = np.exp(-0.5 * diff * diff) / np.sqrt(2 * np.pi)
    return kern.mean(axis=0) / h


def _load_case(pkl_dir: str, n_max: int):
    """Return per-realization standardized mu/sigma + pooled data for one case."""
    paths = sorted(glob.glob(os.path.join(pkl_dir, "*.pkl")))[:n_max]
    if not paths:
        return None

    reals = []
    for p in paths:
        try:
            with open(p, "rb") as f:
                ds = pickle.load(f)
        except Exception as e:
            print(f"[warn] failed to load {p}: {e}", file=sys.stderr)
            continue
        if not hasattr(ds, "mu_0_per_query"):
            continue

        y_int = np.asarray(ds.y_int if hasattr(ds, "y_int") else ds.target_y,
                            dtype=np.float64).reshape(-1)
        # Per-realization Y-scale (matches eval convention: fit on y_int range).
        y_min, y_max = float(y_int.min()), float(y_int.max())
        y_shift = 0.5 * (y_min + y_max)
        y_scale = max(0.5 * (y_max - y_min), 1e-9)

        m0 = (np.asarray(ds.mu_0_per_query, dtype=np.float64) - y_shift) / y_scale
        m1 = (np.asarray(ds.mu_1_per_query, dtype=np.float64) - y_shift) / y_scale
        sig_eps = float(getattr(ds, "sigma_eps", 1e-3)) / y_scale
        rho = float(getattr(ds, "rho_y_noise", 0.0))

        reals.append({"mu0": m0, "mu1": m1, "sig_eps": sig_eps, "rho": rho,
                       "ate": float((m1 - m0).mean()),
                       "y_scale_raw": y_scale})

    if not reals:
        return None

    return reals


def _pool_reals(reals):
    """Concatenate per-realization mu arrays; per-row sigma tiled to match."""
    mu0 = np.concatenate([r["mu0"] for r in reals])
    mu1 = np.concatenate([r["mu1"] for r in reals])
    sig_y = np.concatenate([np.full_like(r["mu0"], r["sig_eps"]) for r in reals])
    # For τ: sigma_τ = sqrt(2·σ²·(1-ρ)) is per-realization.
    sig_tau_per_real = [float(np.sqrt(2.0 * r["sig_eps"] ** 2 * (1.0 - r["rho"])))
                          for r in reals]
    sig_tau = np.concatenate([np.full_like(r["mu0"], s)
                                for r, s in zip(reals, sig_tau_per_real)])
    tau = mu1 - mu0
    ate = np.array([r["ate"] for r in reals])
    return mu0, mu1, tau, sig_y, sig_tau, ate


def _robust_range(a: np.ndarray, lo_pct: float, hi_pct: float, pad_frac=0.05):
    lo = float(np.percentile(a, lo_pct))
    hi = float(np.percentile(a, hi_pct))
    if hi - lo < 1e-9:
        hi = lo + 1.0
    pad = pad_frac * (hi - lo)
    return lo - pad, hi + pad


def _panel(ax, x, pdf, title: str, xlabel: str, color: str):
    ax.plot(x, pdf, linewidth=1.4, color=color)
    ax.fill_between(x, 0, pdf, alpha=0.20, color=color)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, linewidth=0.5)


def main():
    args = _parse_args()
    _install_dopfn_paths(args.dopfn_root)

    n_rows = len(CASE_STUDIES)
    fig, axes = plt.subplots(n_rows, 4,
                              figsize=(4 * 3.4, n_rows * 2.4),
                              squeeze=False)

    for row, case in enumerate(CASE_STUDIES):
        reals = _load_case(os.path.join(args.pkl_root, case), args.n_max_per_case)
        if reals is None:
            for c in range(4):
                axes[row, c].text(0.5, 0.5, f"no pkls in\n{case}",
                                  ha="center", va="center",
                                  transform=axes[row, c].transAxes, fontsize=9)
                axes[row, c].axis("off")
            continue

        # Optional outlier-realization filter.
        if args.outlier_clip_sigmas > 0:
            kept = [r for r in reals
                    if max(np.abs(r["mu0"]).max(), np.abs(r["mu1"]).max())
                       <= args.outlier_clip_sigmas]
            dropped = len(reals) - len(kept)
            if dropped:
                print(f"[{case}] dropped {dropped}/{len(reals)} outlier realizations "
                      f"(|mu_scaled|.max > {args.outlier_clip_sigmas})", flush=True)
            if not kept:
                for c in range(4):
                    axes[row, c].text(0.5, 0.5, f"all realizations filtered",
                                      ha="center", va="center",
                                      transform=axes[row, c].transAxes)
                    axes[row, c].axis("off")
                continue
            reals = kept

        mu0, mu1, tau, sig_y, sig_tau, ate = _pool_reals(reals)

        # Robust axis limits (1st–99th percentile of the mixture "centers").
        y_lo, y_hi = _robust_range(np.concatenate([mu0, mu1]),
                                     args.axis_lo_pct, args.axis_hi_pct)
        tau_lo, tau_hi = _robust_range(tau, args.axis_lo_pct, args.axis_hi_pct)
        ate_lo, ate_hi = _robust_range(ate,
                                         min(5.0, args.axis_lo_pct),
                                         max(95.0, args.axis_hi_pct), pad_frac=0.15)

        y_grid   = np.linspace(y_lo, y_hi, args.grid_n)
        tau_grid = np.linspace(tau_lo, tau_hi, args.grid_n)
        ate_grid = np.linspace(ate_lo, ate_hi, args.grid_n)

        p_ydo0 = _gaussian_mixture_pdf(mu0, sig_y, y_grid)
        p_ydo1 = _gaussian_mixture_pdf(mu1, sig_y, y_grid)
        p_tau  = _gaussian_mixture_pdf(tau, sig_tau, tau_grid)
        p_ate  = _kde_pdf(ate, ate_grid)

        _panel(axes[row, 0], y_grid,   p_ydo0, "p(Ỹ | do(0))", "Ỹ (standardized)", "C0")
        _panel(axes[row, 1], y_grid,   p_ydo1, "p(Ỹ | do(1))", "Ỹ (standardized)", "C1")
        _panel(axes[row, 2], tau_grid, p_tau,  "p(τ̃)  (all queries)", "τ̃", "C2")
        _panel(axes[row, 3], ate_grid, p_ate,
               f"p(ATẼ) (R={len(reals)})", "ATẼ", "C3")

        row_label = (f"{case}\nρ={reals[0]['rho']:.2f}   "
                     f"σ̃ε≈{np.median([r['sig_eps'] for r in reals]):.3f}\n"
                     f"R={len(reals)}   N/real={len(reals[0]['mu0'])}")
        axes[row, 0].set_ylabel(row_label, fontsize=9)

    fig.suptitle("Analytical truth densities — standardized per-realization "
                 "([-1, +1] eval scale), 1st–99th pct axis clipping",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"[ok] wrote {args.out}")


if __name__ == "__main__":
    main()
