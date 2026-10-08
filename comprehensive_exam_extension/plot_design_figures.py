"""Design figures for the four real-data experiments, drawn before any results, in the style of
the noise-shape and function-path figures: in-prior in red, perturbation levels light -> dark blue.

  treatment_balance.png      treated fraction p = 10%, ..., 90% (in-prior: 50%, a median split).
                             T = 1{s > q_(1-p)}, s = f_T(w C) + eps_T the latent treatment score;
                             only the cut-off moves.
  covariate_type.png         the observed confounder C replaced by a discretized version with
                             k equal-probability levels (k = 2 is binary), each level coded by its
                             bin mean; target = CATE given the recorded C. The
                             baseline is the continuous (in-prior) C.
  irrelevant_covariates.png  extra columns z ~ N(0, sigma_exo^2), independent of everything (one shown).
  outcome_type.png           the observed Y replaced by its discretization into k equal-probability
                             bins (k = 2 is binary), coded by bin means; target = CATE of the
                             discretized outcome.

Example curves use one Observed Confounder SCM from generation.py (seed printed).

    python plot_design_figures.py --out results/design
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch
from scipy import stats

from generation import ACTIVATIONS, _SampledSCM, build_dag

p = argparse.ArgumentParser()
p.add_argument("--out", default="results/design")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

INK, MUTED, GRID, RED, GREY = "#0b0b0b", "#52514e", "#e6e5e1", "#e34948", "#b9b8b3"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]   # light = closest to in-prior
KS = (20, 10, 5, 3, 2)                                            # discretization levels


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)


def title(ax, s):
    ax.set_title(s, loc="left", fontsize=11.5, color=INK)


# ── example Observed Confounder SCM: monotone treatment mechanism, visible effect ──────────
def example_scm():
    for seed in range(1000):
        scm = _SampledSCM(build_dag("Observed_Confounder"), N=200_000, rng=np.random.default_rng(seed))
        if scm._activation["T"] in ("tanh", "identity") and scm._activation["Y"] == "tanh":
            obs = scm.forward()
            mu0 = scm.forward(do_T=0.0, y_noiseless=True)["Y"]
            mu1 = scm.forward(do_T=1.0, y_noiseless=True)["Y"]
            if np.std(mu1 - mu0) > 0.3 * np.std(obs["Y"]):
                return seed, scm, obs
    raise RuntimeError("no example SCM found")


seed, scm, obs = example_scm()
print(f"example SCM: Observed_Confounder seed {seed}, f_T = {scm._activation['T']}, "
      f"f_Y = {scm._activation['Y']}, sigma_exo = {scm.exo_std:.2f}, sigma_eps = {scm.noise_std:.3f}")
nodes = {n.name: n for n in scm.nodes}
sx, se = scm.exo_std, scm.noise_std


def t_score_mean(c):
    """f_T(w_C c): the latent treatment score without its noise."""
    return ACTIVATIONS[scm._activation["T"]](c * scm._weights["T"][0])


def y_mean(c, t):
    w = scm._weights["Y"]   # parents (C, T)
    return ACTIVATIONS[scm._activation["Y"]](w[0] * c + w[1] * t)


def bins(x, k):
    """Equal-probability cut points of x and the mean of x inside each bin."""
    edges = np.quantile(x, np.linspace(0, 1, k + 1)[1:-1])
    idx = np.searchsorted(edges, x)
    return edges, np.array([x[idx == b].mean() for b in range(k)])


def quantize(x, edges, means):
    return means[np.searchsorted(edges, x)]


# ═════ 1. treatment balance ═════════════════════════════════════════════════════════════════
PS = (10, 20, 30, 40, 50, 60, 70, 80, 90)


def p_color(pct):
    return RED if pct == 50 else RAMP[min(abs(pct - 50) // 10, 4)]


s = t_score_mean(obs["C"]) + scm._noise["T"]
fig, ax = plt.subplots(figsize=(7.5, 4.8))
c = np.linspace(-2.5 * sx, 2.5 * sx, 1001)
for pct in PS:
    thr = np.quantile(s, 1 - pct / 100)
    ax.plot(c / sx, stats.norm.sf((thr - t_score_mean(c)) / se), color=p_color(pct),
            lw=2.6 if pct == 50 else 1.7, ls="-" if pct <= 50 else "--",
            label=f"{pct}% treated" + (" (in-prior)" if pct == 50 else ""), zorder=3 if pct == 50 else 2)
title(ax, "Propensity P(T = 1 | C) per treated fraction")
ax.set_xlabel("confounder C  (in units of σ_exo)", fontsize=10)
ax.set_ylabel("P(T = 1 | C)", fontsize=10)
ax.legend(fontsize=8, frameon=True, facecolor="white", edgecolor="none", framealpha=1, loc="lower right")
style(ax)
fig.tight_layout()
fig.savefig(os.path.join(a.out, "treatment_balance.png"), dpi=180, facecolor="white")
plt.close(fig)

# ═════ 2. covariate type ════════════════════════════════════════════════════════════════════
z = np.random.default_rng(1).standard_normal(400_000)
grid = np.linspace(-3, 3, 2001)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))

ax[0].plot(grid, grid, color=RED, lw=2.6, label="continuous (in-prior)")
for col, k in zip(RAMP, KS):
    e, m = bins(z, k)
    ax[0].plot(grid, quantize(grid, e, m), color=col, lw=1.7, drawstyle="steps-mid",
               label=f"k = {k} categories" + (" (binary)" if k == 2 else ""))
title(ax[0], "Categorical C: true confounder → recorded confounder")
ax[0].set_xlabel("true confounder C  (in units of σ_exo)", fontsize=10)
ax[0].set_ylabel("recorded C  (mean of its category)", fontsize=10)
ax[0].legend(fontsize=8.5, frameon=False, loc="upper left")

# target: CATE given the recorded C = average of the true CATE(C) over the units in that category
cate_true = y_mean(z * sx, 1) - y_mean(z * sx, 0)
sy_ = obs["Y"].std()
ax[1].plot(grid, (y_mean(grid * sx, 1) - y_mean(grid * sx, 0)) / sy_, color=RED, lw=2.6,
           label="continuous (in-prior)", zorder=3)
for col, k in zip(RAMP, KS):
    e, _ = bins(z, k)
    idx = np.searchsorted(e, z)
    cm = np.array([cate_true[idx == b].mean() for b in range(k)])
    ax[1].plot(grid, cm[np.searchsorted(e, grid)] / sy_, color=col, lw=1.7, drawstyle="steps-mid", label=f"k = {k}")
title(ax[1], "Target: CATE given the recorded confounder (one example SCM)")
ax[1].set_xlabel("true confounder C  (in units of σ_exo)", fontsize=10)
ax[1].set_ylabel("CATE  (in units of sd(Y))", fontsize=10)
ax[1].legend(fontsize=8.5, frameon=False, loc="best")
for x_ in ax:
    style(x_)
fig.tight_layout()
fig.savefig(os.path.join(a.out, "covariate_type.png"), dpi=180, facecolor="white")
plt.close(fig)

# ═════ 3. irrelevant covariates (Do-PFN Fig. 2 colours) ════════════════════════════════════
P_T, P_X, P_Y, P_U = "#e8772e", "#c1272d", "#1f3fbf", "#d9d9d9"   # treatment, covariate, outcome, hidden


def node(ax, xy, name, face):
    ax.add_patch(Circle(xy, 0.26, facecolor=face, edgecolor="#333333", lw=1.0, zorder=3))
    ax.text(*xy, name, ha="center", va="center", fontsize=12, color="white", style="italic", zorder=4)


def arrow(ax, p0, p1):
    v = np.subtract(p1, p0)
    v = v / np.linalg.norm(v) * 0.28
    ax.add_patch(FancyArrowPatch(np.add(p0, v), np.subtract(p1, v), arrowstyle="-|>", mutation_scale=12,
                                 color="black", lw=1.1, zorder=2))


def dag(ax, case):
    if case == "OC":
        pos = {"x₁": (1.0, 2.0), "t": (0.0, 0.5), "y": (2.0, 0.5)}
        edges = [("x₁", "t"), ("x₁", "y"), ("t", "y")]
        zs = [(2.4, 2.0)]
    else:
        pos = {"x₁": (0.4, 2.0), "x₂": (1.6, 2.0), "t": (0.0, 0.5), "y": (2.0, 0.5)}
        edges = [("x₁", "t"), ("x₁", "y"), ("x₂", "t"), ("x₂", "y"), ("t", "y")]
        zs = [(2.8, 2.0)]
    for e in edges:
        arrow(ax, pos[e[0]], pos[e[1]])
    for nm, xy in pos.items():
        node(ax, xy, nm, {"t": P_T, "y": P_Y, "x₂": P_U}.get(nm, P_X))
    for j, xy in enumerate(zs):
        node(ax, xy, "z", P_X)
    ax.set_xlim(-0.5, 3.3)
    ax.set_ylim(0.1, 2.5)
    ax.set_aspect("equal")
    ax.axis("off")


fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for ax_, case, name in zip(ax, ("OC", "UC"), ("Observed Confounder + irrelevant covariate z", "Unobserved Confounder + irrelevant covariate z")):
    dag(ax_, case)
    ax_.set_title(name, fontsize=12, style="italic", weight="bold", color=INK, pad=10)
fig.tight_layout()
fig.savefig(os.path.join(a.out, "irrelevant_covariates.png"), dpi=180, facecolor="white", bbox_inches="tight", pad_inches=0.15)
plt.close(fig)

# ═════ 4. outcome type ══════════════════════════════════════════════════════════════════════
Y = obs["Y"]
sy = Y.std()
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))

yg = np.linspace(np.quantile(Y, 0.001), np.quantile(Y, 0.999), 2001)
ax[0].plot(yg / sy, yg / sy, color=RED, lw=2.6, label="continuous (in-prior)")
for col, k in zip(RAMP, KS):
    e, m = bins(Y, k)
    ax[0].plot(yg / sy, quantize(yg, e, m) / sy, color=col, lw=1.7, drawstyle="steps-mid",
               label=f"k = {k} categories" + (" (binary)" if k == 2 else ""))
title(ax[0], "Categorical Y: true outcome → recorded outcome")
ax[0].set_xlabel("true outcome Y  (in units of sd(Y))", fontsize=10)
ax[0].set_ylabel("recorded Y  (mean of its category)", fontsize=10)
ax[0].legend(fontsize=8.5, frameon=False, loc="upper left")


def cate_k(cc, k):
    """E[Y_k | do(1), c] - E[Y_k | do(0), c], Y_k = bin mean of Y; Y | do(t), c ~ N(mu_t(c), se^2)."""
    if k is None:
        return y_mean(cc, 1) - y_mean(cc, 0)
    e, m = bins(Y, k)
    cuts = np.concatenate([[-np.inf], e, [np.inf]])
    out = 0
    for t, sgn in ((1, 1), (0, -1)):
        mu = y_mean(cc, t)[:, None]
        pr = np.diff(stats.norm.cdf((cuts[None, :] - mu) / se), axis=1)
        out = out + sgn * pr @ m
    return out


cc = np.linspace(-2.5 * sx, 2.5 * sx, 801)
ax[1].plot(cc / sx, cate_k(cc, None) / sy, color=RED, lw=2.6, label="continuous (in-prior)", zorder=3)
for col, k in zip(RAMP, KS):
    ax[1].plot(cc / sx, cate_k(cc, k) / sy, color=col, lw=1.7, label=f"k = {k}")
title(ax[1], "Target: CATE of the recorded outcome (one example SCM)")
ax[1].set_xlabel("confounder C  (in units of σ_exo)", fontsize=10)
ax[1].set_ylabel("CATE_k(c)  (in units of sd(Y))", fontsize=10)
ax[1].legend(fontsize=8.5, frameon=False, loc="best")
for x_ in ax:
    style(x_)
fig.tight_layout()
fig.savefig(os.path.join(a.out, "outcome_type.png"), dpi=180, facecolor="white")
plt.close(fig)
print("wrote", a.out)
