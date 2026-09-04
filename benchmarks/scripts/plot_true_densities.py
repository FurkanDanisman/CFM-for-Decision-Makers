"""Plot the 4 empirical truth densities for each of DoPFN's 6 case studies.

Truth source: `<pkl>_truth_samples.npz` sidecar files produced by
`regen_case_study_pkls.py --n-mc-samples K`. Each sidecar contains

    y_do0_samples : np.ndarray of shape (N_query, K)
    y_do1_samples : np.ndarray of shape (N_query, K)
    sigma_eps, rho_y_noise : scalars

drawn by K MC noise passes through DoPFN's SCM with a half-and-half T split
(so every query row gets K paired samples under do(0) and do(1) with the
rho=0.2 correlation from the DGP).

Per realization we standardize Y by that realization's y_int range
(y_shift, y_scale) so all realizations pool on a comparable [-1, +1] scale.
Axes use robust 1st–99th percentile clipping.

Empirical densities via Gaussian KDE:
  p(Ỹ | do(0))  KDE over pooled y_do0_scaled across (realizations × queries × K)
  p(Ỹ | do(1))  same, y_do1_scaled
  p(τ̃)          KDE over pooled (y_do1_scaled - y_do0_scaled)  — captures rho correlation
  p(ATẼ)        KDE over per-realization sample means of τ̃

Usage:
    python plot_true_densities.py \
        --pkl-root   /scratch/.../external/dopfn/data/prior_sampling_rho02 \
        --dopfn-root /scratch/.../external/dopfn \
        --out        /scratch/.../results_density_rho02/true_densities.pdf
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
                   help="Directory containing <CASE>/*.pkl (+ <CASE>/*_truth_samples.npz).")
    p.add_argument("--dopfn-root", default=os.environ.get("DOPFN_ROOT", ""),
                   help="Path to dopfn_upstream root. Falls back to $DOPFN_ROOT env var.")
    p.add_argument("--out", required=True, help="Output PDF path.")
    p.add_argument("--n-max-per-case", type=int, default=100)
    p.add_argument("--n-max-samples-per-real", type=int, default=None,
                   help="Optional cap on MC samples per realization (for speed).")
    p.add_argument("--grid-n", type=int, default=1000)
    p.add_argument("--axis-lo-pct", type=float, default=1.0)
    p.add_argument("--axis-hi-pct", type=float, default=99.0)
    return p.parse_args()


def _install_dopfn_paths(dopfn_root: str) -> None:
    if not dopfn_root or not os.path.isdir(dopfn_root):
        print(f"[warn] no valid --dopfn-root — pkl unpickling may fail.",
              file=sys.stderr)
        return
    if dopfn_root not in sys.path:
        sys.path.insert(0, dopfn_root)


def _kde_pdf(samples: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Silverman-bandwidth Gaussian KDE, chunked to bound memory."""
    n = len(samples)
    if n < 2:
        return np.zeros_like(x_grid)
    std = float(np.std(samples, ddof=1))
    if std <= 0:
        std = max(1e-6, 0.01 * float(np.abs(samples).mean() or 1.0))
    h = 1.06 * std * n ** (-1 / 5)

    G = len(x_grid)
    chunk = max(1, min(n, int(2e7 / G)))
    out = np.zeros(G, dtype=np.float64)
    for start in range(0, n, chunk):
        s = samples[start:start + chunk][:, None]
        d = (x_grid[None, :] - s) / h
        out += np.exp(-0.5 * d * d).sum(axis=0)
    return out / (n * h * np.sqrt(2 * np.pi))


def _robust_range(a: np.ndarray, lo_pct: float, hi_pct: float, pad_frac=0.05):
    lo = float(np.percentile(a, lo_pct))
    hi = float(np.percentile(a, hi_pct))
    if hi - lo < 1e-9:
        hi = lo + 1.0
    pad = pad_frac * (hi - lo)
    return lo - pad, hi + pad


def _load_case(pkl_dir: str, n_max_reals: int, n_max_samples: int):
    """Return per-realization pooled arrays for one case study.

    For each realization r that has a truth-samples sidecar:
      - Compute y_shift_r, y_scale_r from that realization's y_int.
      - Standardize its MC samples into that scale.
      - Append to pooled arrays.

    Realizations without a sidecar are skipped (with a warn message).
    """
    paths = sorted(glob.glob(os.path.join(pkl_dir, "*.pkl")))[:n_max_reals]
    if not paths:
        return None

    y_do0_pool, y_do1_pool = [], []
    ate_r = []
    n_used, n_skipped = 0, 0
    for p in paths:
        base = p[:-4] if p.endswith(".pkl") else p
        npz_path = f"{base}_truth_samples.npz"
        if not os.path.exists(npz_path):
            n_skipped += 1
            continue
        try:
            with open(p, "rb") as f:
                ds = pickle.load(f)
        except Exception as e:
            print(f"[warn] failed to load {p}: {e}", file=sys.stderr)
            n_skipped += 1
            continue

        y_int = np.asarray(ds.y_int if hasattr(ds, "y_int") else ds.target_y,
                            dtype=np.float64).reshape(-1)
        y_min, y_max = float(y_int.min()), float(y_int.max())
        y_shift = 0.5 * (y_min + y_max)
        y_scale = max(0.5 * (y_max - y_min), 1e-9)

        with np.load(npz_path) as z:
            y0 = np.asarray(z["y_do0_samples"], dtype=np.float64)
            y1 = np.asarray(z["y_do1_samples"], dtype=np.float64)
        if n_max_samples is not None and y0.shape[1] > n_max_samples:
            y0 = y0[:, :n_max_samples]
            y1 = y1[:, :n_max_samples]

        y0_std = (y0 - y_shift) / y_scale
        y1_std = (y1 - y_shift) / y_scale

        y_do0_pool.append(y0_std.reshape(-1))
        y_do1_pool.append(y1_std.reshape(-1))
        ate_r.append(float((y1_std - y0_std).mean()))
        n_used += 1

    if n_used == 0:
        return None
    return {
        "y_do0": np.concatenate(y_do0_pool),
        "y_do1": np.concatenate(y_do1_pool),
        "ate_r": np.array(ate_r),
        "n_used": n_used,
        "n_skipped": n_skipped,
    }


def _panel(ax, x, pdf, title, xlabel, color):
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
        data = _load_case(os.path.join(args.pkl_root, case),
                           args.n_max_per_case,
                           args.n_max_samples_per_real)
        if data is None:
            msg = (f"no truth-samples .npz found in\n{case}\n"
                   f"(regen with --n-mc-samples > 0 first)")
            for c in range(4):
                axes[row, c].text(0.5, 0.5, msg, ha="center", va="center",
                                  transform=axes[row, c].transAxes, fontsize=9)
                axes[row, c].axis("off")
            continue

        y0 = data["y_do0"]; y1 = data["y_do1"]
        tau = y1 - y0
        ate = data["ate_r"]

        y_lo,  y_hi  = _robust_range(np.concatenate([y0, y1]),
                                       args.axis_lo_pct, args.axis_hi_pct)
        tau_lo, tau_hi = _robust_range(tau,
                                        args.axis_lo_pct, args.axis_hi_pct)
        ate_lo, ate_hi = _robust_range(ate, 2.5, 97.5, pad_frac=0.15)

        y_grid   = np.linspace(y_lo,   y_hi,   args.grid_n)
        tau_grid = np.linspace(tau_lo, tau_hi, args.grid_n)
        ate_grid = np.linspace(ate_lo, ate_hi, args.grid_n)

        p_y0  = _kde_pdf(y0,  y_grid)
        p_y1  = _kde_pdf(y1,  y_grid)
        p_tau = _kde_pdf(tau, tau_grid)
        p_ate = _kde_pdf(ate, ate_grid)

        _panel(axes[row, 0], y_grid,   p_y0,  "p(Ỹ | do(0))", "Ỹ (standardized)", "C0")
        _panel(axes[row, 1], y_grid,   p_y1,  "p(Ỹ | do(1))", "Ỹ (standardized)", "C1")
        _panel(axes[row, 2], tau_grid, p_tau, "p(τ̃)  (pooled)", "τ̃", "C2")
        _panel(axes[row, 3], ate_grid, p_ate,
               f"p(ATẼ)  (R={data['n_used']})", "ATẼ", "C3")

        row_label = (f"{case}\n"
                     f"R={data['n_used']}   skipped={data['n_skipped']}\n"
                     f"total samples={y0.size:,}")
        axes[row, 0].set_ylabel(row_label, fontsize=9)

    fig.suptitle("Empirical truth densities — per-realization Y-standardized "
                 "([-1, +1] eval scale), Gaussian KDE, 1st–99th pct axis clipping",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"[ok] wrote {args.out}")


if __name__ == "__main__":
    main()
