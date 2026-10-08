"""Shape paths of the candidate noise families for Experiment B: in each family one shape
parameter moves the distribution further from the Gaussian over 5 levels (light -> dark).
All centred at mean 0, in units of the in-prior noise s.d. sigma (= 1); in-prior N(0, 1) dashed.

  scaled   : each level rescaled to variance 1 (X / sd(X)) -> only the shape changes
  unscaled : same parameters in textbook form with scale 1, not rescaled -> variance changes too

Families (shape parameter, levels):
  Gamma, centred           G - k, G ~ Gamma(k, 1)              k     16, 8, 4, 2, 1
  Bimodal normal mixture   0.5 N(-a, 1) + 0.5 N(a, 1)          a     1, 1.5, 2, 2.5, 3
  Contaminated (Tukey)     0.9 N(0, 1) + 0.1 N(0, c^2)          c     2, 3, 5, 7, 10
  Log-normal, centred      exp(s Z) - exp(s^2 / 2)              s     0.25, 0.5, 0.75, 1, 1.25
  Generalized normal       density ∝ exp(-|x|^beta)             beta  1.5, 1.25, 1 (Laplace), 0.75, 0.6
                           (beta = 2 is Gaussian)

    python plot_noise_shape_paths.py --out results/noise_dists
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.special import gamma as G

p = argparse.ArgumentParser()
p.add_argument("--out", default="results/noise_dists")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]   # light = closest to Gaussian
RED = "#e34948"                                                   # in-prior (baseline) Gaussian
gauss = stats.norm()


# Each family: list of (label, pdf of the unscaled mean-0 variable, its variance, shape statistic).
def gamma_fam():
    out = []
    for k in (16, 8, 4, 2, 1):
        out.append((f"k = {k}", (lambda k: lambda x: stats.gamma(k).pdf(x + k))(k), float(k),
                    f"skew {2 / np.sqrt(k):.2f}"))
    return out


def bimodal_fam():
    out = []
    for m in (1, 1.5, 2, 2.5, 3):
        var = 1 + m ** 2
        kurt = (m ** 4 + 6 * m ** 2 + 3) / var ** 2 - 3
        out.append((f"a = {m:g}", (lambda m: lambda x: 0.5 * (stats.norm(-m, 1).pdf(x)
                                                              + stats.norm(m, 1).pdf(x)))(m),
                    var, f"kurtosis {kurt:+.2f}"))
    return out


def contam_fam():
    out = []
    for c in (2, 3, 5, 7, 10):
        var = 0.9 + 0.1 * c ** 2
        kurt = 3 * (0.9 + 0.1 * c ** 4) / var ** 2 - 3
        out.append((f"c = {c}", (lambda c: lambda x: 0.9 * stats.norm(0, 1).pdf(x)
                                 + 0.1 * stats.norm(0, c).pdf(x))(c), var, f"kurtosis {kurt:+.1f}"))
    return out


def lognorm_fam():
    out = []
    for s in (0.25, 0.5, 0.75, 1.0, 1.25):
        w = np.exp(s ** 2)
        var, skew = (w - 1) * w, (w + 2) * np.sqrt(w - 1)
        out.append((f"s = {s:g}", (lambda s: lambda x: stats.lognorm(s).pdf(x + np.exp(s ** 2 / 2)))(s),
                    var, f"skew {skew:.2f}"))
    return out


def gennorm_fam():
    out = []
    for b in (1.5, 1.25, 1.0, 0.75, 0.6):
        var = G(3 / b) / G(1 / b)
        kurt = G(5 / b) * G(1 / b) / G(3 / b) ** 2 - 3
        lab = f"β = {b:g}" + (" (Laplace)" if b == 1 else "")
        out.append((lab, stats.gennorm(b).pdf, var, f"kurtosis {kurt:+.1f}"))
        out[-1] = (lab, (lambda b: stats.gennorm(b).pdf)(b), var, f"kurtosis {kurt:+.1f}")
    return out


FAMILIES = [
    ("Gamma, centred (skew)", "Gamma, centred: G − k", gamma_fam()),
    ("Bimodal normal mixture", "Bimodal: ½ N(−a, 1) + ½ N(a, 1)", bimodal_fam()),
    ("Contaminated normal (Tukey)", "Contaminated: 0.9 N(0,1) + 0.1 N(0,c²)", contam_fam()),
    ("Log-normal, centred (skew)", "Log-normal, centred: e^{sZ} − e^{s²/2}", lognorm_fam()),
    ("Generalized normal (β=1: Laplace)", "Generalized normal ∝ exp(−|x|^β)", gennorm_fam()),
]


def rescaled(pdf, var):
    sd = np.sqrt(var)
    return lambda x: sd * pdf(sd * x)


for name in ("scaled", "unscaled"):
    xlim = (-5, 5) if name == "scaled" else (-8, 12)
    x = np.linspace(*xlim, 6001)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6), sharex=True)
    ax0 = axes[0, 0]
    ax0.fill_between(x, gauss.pdf(x), color=RED, alpha=0.15, lw=0)
    ax0.plot(x, gauss.pdf(x), color=RED, lw=2)
    ax0.text(0.97, 0.95, "variance = 1.00", transform=ax0.transAxes, ha="right", va="top",
             fontsize=9, color=MUTED)
    ax0.set_title("IOD: Gaussian (training noise)", loc="left", fontsize=10, color=INK)
    for ax, (t_scaled, t_unscaled, fam) in zip(axes.ravel()[1:], FAMILIES):
        ax.plot(x, gauss.pdf(x), "--", color=RED, lw=1.5, label="in-prior Gaussian")
        for c, (lab, pdf, var, stat) in zip(RAMP, fam):
            f = rescaled(pdf, var) if name == "scaled" else pdf
            extra = stat if name == "scaled" else f"variance {var:.3g}"
            ax.plot(x, f(x), color=c, lw=1.9, label=f"{lab}  ({extra})")
        ax.set_title(t_scaled if name == "scaled" else t_unscaled, loc="left", fontsize=10, color=INK)
        ax.legend(fontsize=7.5, frameon=False, loc="upper right")
        if name == "scaled":
            ax.text(0.03, 0.95, "variance = 1", transform=ax.transAxes, ha="left", va="top",
                    fontsize=9, color=MUTED)
    for ax in axes.ravel():
        ax.set_ylim(bottom=0)
        ax.set_xlim(*xlim)
        ax.grid(color=GRID, lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=9)
    for ax in axes[1]:
        ax.set_xlabel("noise ε  (in units of the in-prior s.d. σ_ε)", fontsize=9, color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("density", fontsize=9, color=INK)
    fig.tight_layout()
    out = os.path.join(a.out, f"noise_shape_paths_{name}.png")
    fig.savefig(out, dpi=200, facecolor="white")
    print(out)
