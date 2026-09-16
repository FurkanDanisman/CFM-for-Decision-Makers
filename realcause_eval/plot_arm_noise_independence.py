"""Show that RealCause draws Y(0) and Y(1) with INDEPENDENT arm noise.

Why it matters: a 1D head must assume some coupling to turn two marginals
into a law for tau = Y(1) - Y(0), and

    Var(tau) = s0^2 + s1^2 - 2 rho s0 s1

so the independence convolution is only the right assumption when rho = 0.
On generators that share exogenous noise across arms rho is near 1 and that
convolution is badly too wide. RealCause is the case where it is correct, and
this figure is the evidence.

WHAT IS PLOTTED. The RealCause fitted-SCM CSVs fix X and redraw Y, giving
K=100 paired potential outcomes per unit. For each unit i and draw k we take
the within-unit residual

    e0[i,k] = y0[i,k] - mean_k y0[i,·]        (same for arm 1)

which removes mu_t(x_i) without needing it, then standardise each arm by its
own pooled sd. Under independent arm noise the joint is a spherical bivariate
standard normal, so the 95% region is a CIRCLE of radius sqrt(chi2_2(0.95)).
The empirical 95% ellipse is drawn on top: if the two coincide, the noise is
independent; a tilted ellipse is shared noise.

SMALL MULTIPLES, not one overlaid scatter. Five point clouds on shared axes
occlude each other, and the reference palette's all-pairs gate admits only
three categorical slots -- past three the guidance is to facet. Panels carry
identity, so every panel uses one hue.

    python plot_arm_noise_independence.py --causalpfn $CAUSALPFN --out fig.pdf
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np

# dataviz reference palette, slot 1 (blue). Validated all-pairs light mode:
# CVD dE 9.2, normal-vision 24.0, and unlike slot 3 it clears 3:1 on the
# light surface, so no relief rule is triggered.
SERIES = "#2a78d6"
SURFACE = "#fcfcfb"
INK = "#1a1a19"
MUTED = "#6b6b68"

PRETTY = {"lalonde_cps_sample": "CPS", "lalonde_psid_sample": "PSID",
          "ihdp_sample": "IHDP", "acic_sample": "ACIC"}


def find_csv_dir(causalpfn: str) -> str:
    for c in (os.path.join(causalpfn, "benchmarks", "realcause_datasets"),
              os.path.join(causalpfn, "src", "benchmarks", "realcause_datasets"),
              os.path.join(causalpfn, "realcause_datasets")):
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(f"no realcause_datasets under {causalpfn}")


def discover(csv_dir: str) -> dict[str, list[str]]:
    """-> {prefix: [csv paths]} for every *_sample<k>.csv family present."""
    out: dict[str, list[str]] = {}
    for p in sorted(glob.glob(os.path.join(csv_dir, "*sample*.csv"))):
        m = re.match(r"(.*sample)\d+\.csv$", os.path.basename(p))
        if m:
            out.setdefault(m.group(1), []).append(p)
    return {k: sorted(v) for k, v in out.items() if len(v) > 1}


def residuals(paths, max_files=None):
    """Within-unit standardised residuals, pooled. -> (e0, e1, rho, n_unit, K)."""
    # pandas if present, else the stdlib reader -- this figure should not
    # need a heavier dependency than the evals that produced the data.
    try:
        import pandas as pd

        def _read(path):
            df = pd.read_csv(path)
            if not {"y0", "y1"} <= set(df.columns):
                return None
            return df["y0"].to_numpy(float), df["y1"].to_numpy(float)
    except ImportError:
        import csv as _csv

        def _read(path):
            with open(path, newline="") as fh:
                rd = _csv.DictReader(fh)
                if not {"y0", "y1"} <= set(rd.fieldnames or []):
                    return None
                a, b = [], []
                for row in rd:
                    a.append(float(row["y0"])); b.append(float(row["y1"]))
            return np.asarray(a), np.asarray(b)

    y0, y1 = [], []
    for p in (paths[:max_files] if max_files else paths):
        got = _read(p)
        if got is None:
            return None
        y0.append(got[0]); y1.append(got[1])
    n = min(len(a) for a in y0)
    Y0 = np.stack([a[:n] for a in y0], axis=1)      # (n_unit, K)
    Y1 = np.stack([a[:n] for a in y1], axis=1)
    # Remove mu_t(x_i) by centring within unit -- no model of mu needed.
    e0 = Y0 - Y0.mean(axis=1, keepdims=True)
    e1 = Y1 - Y1.mean(axis=1, keepdims=True)
    s0, s1 = e0.std(), e1.std()
    if s0 <= 0 or s1 <= 0:
        return None
    e0, e1 = e0 / s0, e1 / s1
    rho = float(np.corrcoef(e0.ravel(), e1.ravel())[0, 1])
    return e0.ravel(), e1.ravel(), rho, Y0.shape[0], Y0.shape[1]


def ellipse_xy(cov, nsig, n=400):
    t = np.linspace(0, 2*np.pi, n)
    L = np.linalg.cholesky(cov + 1e-12*np.eye(2))
    return (L @ np.vstack([np.cos(t), np.sin(t)])) * nsig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--causalpfn", default=os.environ.get("CAUSALPFN", ""))
    ap.add_argument("--out", default="realcause_arm_noise.pdf")
    ap.add_argument("--max-points", type=int, default=4000,
                    help="points drawn per panel; all data is used for rho")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fams = discover(find_csv_dir(a.causalpfn))
    if not fams:
        raise SystemExit("no *_sample<k>.csv families found")

    res = {}
    for prefix, paths in fams.items():
        got = residuals(paths, a.max_files)
        if got is None:
            print(f"[skip] {prefix}: no y0/y1 columns"); continue
        res[PRETTY.get(prefix, prefix)] = got
        print(f"{PRETTY.get(prefix, prefix):8s} n_unit={got[3]:6d} K={got[4]:3d} "
              f"rho={got[2]:+.4f}")
    if not res:
        raise SystemExit("no dataset had paired y0/y1")

    # 95% contour of a spherical bivariate normal.
    R95 = float(np.sqrt(5.991464547107979))        # chi2_2(0.95)
    rng = np.random.default_rng(a.seed)
    k = len(res)
    fig, axes = plt.subplots(1, k, figsize=(2.55*k + 0.4, 3.0),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(SURFACE)

    for ax, (name, (e0, e1, rho, n_unit, K)) in zip(axes, res.items()):
        ax.set_facecolor(SURFACE)
        idx = (rng.choice(e0.size, a.max_points, replace=False)
               if e0.size > a.max_points else slice(None))
        ax.scatter(e0[idx], e1[idx], s=4, c=SERIES, alpha=0.18,
                   linewidths=0, rasterized=True)
        # theoretical independence contour
        # Reference drawn as a THICK MUTED band, the empirical as thin ink
        # dashes on top. When rho = 0 the two coincide and must still read as
        # two curves -- a same-weight solid+dashed pair hides one under the
        # other, and a white halo under the dashes erases the reference
        # altogether, which reads as a missing series.
        t = np.linspace(0, 2*np.pi, 400)
        ax.plot(R95*np.cos(t), R95*np.sin(t), color=MUTED, lw=3.2,
                alpha=0.85, zorder=3, solid_capstyle="round")
        # empirical contour -- coincides with the circle iff rho = 0
        # zorder above the circle: when rho = 0 the two coincide, and the
        # dashes must ride ON the solid curve rather than hide beneath it --
        # otherwise the independent panel reads as a missing series.
        xy = ellipse_xy(np.cov(np.vstack([e0, e1])), R95)
        ax.plot(xy[0], xy[1], color=INK, lw=1.3, ls=(0, (3.5, 2.5)), zorder=5)
        ax.set_title(name, fontsize=10, color=INK, pad=6)
        ax.text(0.04, 0.955, rf"$\hat\rho = {rho:+.3f}$", transform=ax.transAxes,
                fontsize=9, color=INK, va="top")
        ax.text(0.04, 0.875, f"n={n_unit:,}  K={K}", transform=ax.transAxes,
                fontsize=7.5, color=MUTED, va="top")
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
        ax.set_aspect("equal")
        ax.axhline(0, color=MUTED, lw=0.5, alpha=0.5, zorder=0)
        ax.axvline(0, color=MUTED, lw=0.5, alpha=0.5, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED); ax.spines[s].set_linewidth(0.8)
        ax.tick_params(colors=MUTED, labelsize=8, length=3)

    axes[0].set_ylabel(r"standardised $Y(1)$ residual", fontsize=9, color=INK)
    for ax in axes:
        ax.set_xlabel(r"standardised $Y(0)$ residual", fontsize=9, color=INK)
    # identity is carried by the panel, so the legend explains the two curves
    axes[-1].plot([], [], color=MUTED, lw=3.2, alpha=0.85,
                  label=r"95% under independence")
    axes[-1].plot([], [], color=INK, lw=1.3, ls=(0, (3.5, 2.5)),
                  label="95% empirical")
    axes[-1].legend(fontsize=7.5, frameon=False, loc="lower right",
                    labelcolor=INK, handlelength=1.8)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight", facecolor=SURFACE, dpi=200)
    print(f"\n[written] {a.out}")


if __name__ == "__main__":
    main()
