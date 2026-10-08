"""Densities of the candidate noise distributions for Experiment B, in units of the in-prior
noise s.d. sigma (= 1 here), each panel with the in-prior Gaussian N(0, 1) dashed. Two figures:

  scaled   : every distribution rescaled to mean 0 and variance 1 -> only the shape differs
  unscaled : every distribution in its textbook form with scale parameter sigma, centred at
             mean 0 but NOT rescaled -> the variance is whatever the distribution implies

Distributions (standard error laws in robust / nonparametric regression simulations):
  centred Gamma, shape k in {1, 2, 4, 8, 16}   (G - k) * sigma, G ~ Gamma(k, 1); variance k
  bimodal normal mixture                        0.5 N(-a, s^2) + 0.5 N(a, s^2)
  contaminated normal (Tukey)                   0.9 N(0, 1) + 0.1 N(0, 3^2); variance 1.8
  log-normal, centred                           exp(Z) - e^(1/2), Z ~ N(0, 1); variance (e - 1) e
  Laplace                                       Laplace(0, 1); variance 2

    python plot_noise_candidates.py --out results/noise_dists
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

p = argparse.ArgumentParser()
p.add_argument("--out", default="results/noise_dists")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
BLUE, RED = SERIES[0], "#e34948"   # RED = in-prior (baseline) Gaussian
gauss = stats.norm()
E = np.e
GAMMA_K = (1, 2, 4, 8, 16)


def rescaled(pdf, var):
    """Density of X / sqrt(var) for X with density pdf (mean 0): variance 1."""
    sd = np.sqrt(var)
    return lambda x: sd * pdf(sd * x)


# Textbook (unscaled) forms, all mean 0, with their variances.
def gamma_pdf(k):
    return lambda x: stats.gamma(k).pdf(x + k)


BIMODAL_A = 1.5   # unscaled: components keep s.d. sigma, modes at +-1.5 sigma
bimodal_pdf = lambda x: 0.5 * (stats.norm(-BIMODAL_A, 1).pdf(x) + stats.norm(BIMODAL_A, 1).pdf(x))
contam_pdf = lambda x: 0.9 * stats.norm(0, 1).pdf(x) + 0.1 * stats.norm(0, 3).pdf(x)
lognorm_pdf = lambda x: stats.lognorm(1.0).pdf(x + np.sqrt(E))
laplace_pdf = stats.laplace(0, 1).pdf
VAR = dict(bimodal=1 + BIMODAL_A ** 2, contam=0.9 + 0.1 * 9, lognorm=(E - 1) * E, laplace=2.0)

SETS = {
    "scaled": [
        ("IOD: Gaussian (training noise)", [("", gauss.pdf, 1.0)]),
        ("Gamma, centred (right skew)",
         [(f"k = {k}  (skew {2 / np.sqrt(k):.2f})", rescaled(gamma_pdf(k), k), 1.0) for k in GAMMA_K]),
        # the bimodal shape with the same mode separation as the unscaled panel, at variance 1
        ("Bimodal normal mixture", [("", rescaled(bimodal_pdf, VAR["bimodal"]), 1.0)]),
        ("Contaminated normal (Tukey)", [("", rescaled(contam_pdf, VAR["contam"]), 1.0)]),
        ("Log-normal, centred", [("", rescaled(lognorm_pdf, VAR["lognorm"]), 1.0)]),
        ("Laplace", [("", rescaled(laplace_pdf, VAR["laplace"]), 1.0)]),
    ],
    "unscaled": [
        ("IOD: Gaussian (training noise)", [("", gauss.pdf, 1.0)]),
        ("Gamma, centred: (G − k)σ", [(f"k = {k}  (variance {k})", gamma_pdf(k), float(k)) for k in GAMMA_K]),
        (f"Bimodal: ½ N(−{BIMODAL_A}σ, σ²) + ½ N({BIMODAL_A}σ, σ²)", [("", bimodal_pdf, VAR["bimodal"])]),
        ("Contaminated normal (Tukey)", [("", contam_pdf, VAR["contam"])]),
        ("Log-normal, centred: σ(e^Z − e^½)", [("", lognorm_pdf, VAR["lognorm"])]),
        ("Laplace(0, σ)", [("", laplace_pdf, VAR["laplace"])]),
    ],
}

for name, panels in SETS.items():
    xlim = (-5, 5) if name == "scaled" else (-7, 10)
    x = np.linspace(*xlim, 4001)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)
    for ax, (title, curves) in zip(axes.ravel(), panels):
        if not title.startswith("IOD"):
            ax.plot(x, gauss.pdf(x), "--", color=RED, lw=1.5, label="in-prior Gaussian")
        single = len(curves) == 1
        for c, (lab, pdf, var) in zip(SERIES, curves):
            y = pdf(x)
            col = RED if title.startswith("IOD") else (BLUE if single else c)
            if single:
                ax.fill_between(x, y, color=col, alpha=0.15, lw=0)
            ax.plot(x, y, color=col, lw=2, label=lab or None)
        if single:
            ax.text(0.97, 0.95, f"variance = {curves[0][2]:.2f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=9, color=MUTED)
        elif name == "scaled":
            ax.text(0.97, 0.95, "variance = 1.00", transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, color=MUTED)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK)
        if title.startswith("Contaminated"):
            ax.text(0.97, 0.85, "0.9 N(0, σ²) + 0.1 N(0, (3σ)²)" + (", rescaled" if name == "scaled" else ""),
                    transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=MUTED)
        ax.grid(color=GRID, lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=9)
        if not single or title.startswith("Bimodal"):
            ax.legend(fontsize=8, frameon=False, loc="upper left")
    axes[0, 0].set_ylim(0, 1.5 if name == "scaled" else 1.05)   # scaled log-normal peaks near 1.4
    axes[0, 0].set_xlim(*xlim)
    for ax in axes[1]:
        ax.set_xlabel("noise ε  (in units of the in-prior s.d. σ_ε)", fontsize=9, color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("density", fontsize=9, color=INK)
    fig.tight_layout()
    out = os.path.join(a.out, f"noise_candidates_{name}.png")
    fig.savefig(out, dpi=200, facecolor="white")
    print(out)
