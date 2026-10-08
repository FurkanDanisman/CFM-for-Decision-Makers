"""Function paths of Experiment C, in the style of the noise-shape figures: for each training
activation f (rows) and each new function g (columns), the mechanism
    f_lambda(x) = (1 - lambda) f(x) + lambda * a * g(x),   lambda in {0, .1, .2, .3, .5, .75, 1}
with a = sd(f(x)) / sd(g(x)) so g has the same spread as f (generation.py, _SampledSCM._f).
lambda = 0 (the in-prior function) in red, then light -> dark blue.

x is the node's linear input w·pa; a is computed on x ~ N(0, 2^2), a typical input scale in
the case studies (exo_std ~ U(1, 3), Kaiming weights).

    python plot_function_paths.py --out results/function_paths
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--out", default="results/function_paths")
p.add_argument("--input-sd", type=float, default=2.0)
p.add_argument("--new", nargs="+", default=["sin", "sign", "cubic"], help="which g to draw (columns)")
p.add_argument("--name", default="function_paths")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

TRAIN = {"x²": lambda x: x ** 2, "ReLU": lambda x: np.maximum(0, x), "tanh": np.tanh, "identity": lambda x: x}
ALL_NEW = {"sin": np.sin, "sign": np.sign, "cubic": lambda x: x ** 3,
           "expsq": lambda x: np.exp(-x ** 2 / 2), "sin3": lambda x: np.sin(3 * x), "exp": np.exp}
LABEL = {"sin": "sin(x)", "sign": "sign(x)", "cubic": "x³", "expsq": "exp(−x²/2)", "sin3": "sin(3x)",
         "exp": "exp(x)"}
NEW = {k: ALL_NEW[k] for k in a.new}
LAMBDAS = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
RAMP = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
INK, MUTED, GRID, RED = "#0b0b0b", "#52514e", "#e6e5e1", "#e34948"

ref = np.random.default_rng(0).normal(0, a.input_sd, 200_000)
x = np.linspace(-2.5 * a.input_sd, 2.5 * a.input_sd, 2001)
fig, axes = plt.subplots(len(TRAIN), len(NEW), figsize=(4.4 * len(NEW), 13), sharex=True)
for i, (fn, f) in enumerate(TRAIN.items()):
    for j, (gn, g) in enumerate(NEW.items()):
        ax = axes[i, j]
        scale = f(ref).std() / g(ref).std()
        ax.plot(x, f(x), color=RED, lw=2.4, label=f"λ = 0: {fn} (in-prior)")
        for c, lam in zip(RAMP, LAMBDAS):
            ax.plot(x, (1 - lam) * f(x) + lam * scale * g(x), color=c, lw=1.8, label=f"λ = {lam:g}")
        ax.set_title(f"{fn} → {LABEL[gn]}", loc="left", fontsize=11, color=INK)
        ax.grid(color=GRID, lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=9)
        if j == 0:
            ax.set_ylabel("f_λ(x)", fontsize=10, color=INK)
        if i == len(TRAIN) - 1:
            ax.set_xlabel("x  (linear input w·pa)", fontsize=10, color=INK)
        # y range from the central 95% of inputs, so cubic's tails do not flatten the panel
        core = np.abs(x) <= 2 * a.input_sd
        ys = [f(x[core])] + [(1 - l) * f(x[core]) + l * scale * g(x[core]) for l in LAMBDAS]
        lo, hi = min(y.min() for y in ys), max(y.max() for y in ys)
        pad = 0.08 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
axes[0, 0].legend(fontsize=8.5, frameon=False, loc="upper center")
fig.tight_layout()
out = os.path.join(a.out, f"{a.name}.png")
fig.savefig(out, dpi=180, facecolor="white")
print(out)
