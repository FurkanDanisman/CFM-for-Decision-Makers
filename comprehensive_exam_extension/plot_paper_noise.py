"""Densities of the paper's out-of-distribution noise (Do-PFN, Fig. 4 left), with the
parameters from the authors' code (priors/playground_scm/MakeStructuralEquations.py):
all rescaled to the in-prior s.d. sigma, except "High" (Gaussian with twice the variance).
Drawn at sigma = 1; the dashed curve in every panel is the in-prior Gaussian.

    python plot_paper_noise.py --out results/noise_dists
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

SIGMA, NU = 1.0, 3.0
EULER = 0.5772156649015329
gauss = stats.norm(0, SIGMA)
laplace = stats.laplace(0, SIGMA / np.sqrt(2))
student = stats.t(NU, 0, SIGMA * np.sqrt((NU - 2) / NU))
g_scale = np.sqrt(6) / np.pi * SIGMA
gumbel = stats.gumbel_r(-EULER * g_scale, g_scale)
high = stats.norm(0, np.sqrt(2) * SIGMA)

FOUR = (gauss, laplace, student, gumbel)
# (title, pdf, exact variance); all four components have mean 0, so the mixture's
# variance is the average of theirs.
DISTS = [
    ("IOD: Gaussian (training noise)", gauss.pdf, gauss.var()),
    ("Laplace", laplace.pdf, laplace.var()),
    ("T: Student-t, ν = 3", student.pdf, student.var()),
    ("Gumbel (mean 0)", gumbel.pdf, gumbel.var()),
    ("Mixed: ¼ each of the four above", lambda x: sum(d.pdf(x) for d in FOUR) / 4,
     np.mean([d.var() for d in FOUR])),
    ("High: Gaussian, 2× variance", high.pdf, high.var()),
]
INK, MUTED, GRID, BLUE, RED = "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6", "#e34948"   # RED = in-prior

x = np.linspace(-5, 5, 2001)
fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)
for ax, (title, pdf, var) in zip(axes.ravel(), DISTS):
    col = RED if title.startswith("IOD") else BLUE
    ax.fill_between(x, pdf(x), color=col, alpha=0.15, lw=0)
    ax.plot(x, pdf(x), color=col, lw=2)
    if not title.startswith("IOD"):
        ax.plot(x, gauss.pdf(x), "--", color=RED, lw=1.5, label="in-prior Gaussian")
    ax.text(0.97, 0.95, f"variance = {var:.2f}", transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color=MUTED)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.grid(color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
for ax in axes[1]:
    ax.set_xlabel("noise ε  (in units of the in-prior s.d. σ_ε)", fontsize=9, color=INK)
for ax in axes[:, 0]:
    ax.set_ylabel("density", fontsize=9, color=INK)
axes[0, 1].legend(fontsize=8.5, frameon=False, loc="upper left")
fig.tight_layout()
out = os.path.join(a.out, "paper_ood_noise_densities.png")
fig.savefig(out, dpi=200, facecolor="white")
print(out)
