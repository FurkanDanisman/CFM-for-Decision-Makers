"""Figure for sim5_track.py (narrow network): learned correlation, the price of an effect error, PEHE over training,
effect error along x.   .venv/bin/python revision/theory/loss_geometry/fig_sim5.py OUT_DIR"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]; W = "narrow"; SEEDS = range(10)
R = json.load(open(os.path.join(OUT, "sim5_results.json")))
get = lambda s, l: R[f"{W}|{s}|{l}"]
xs = np.linspace(-2, 2, 401); true_rho = 0.10 + 0.85 * (xs + 2) / 4
C = {"separate": "#888888", "joint": "#c0392b"}
fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))

rh = np.array([get(s, "joint")["final"]["rho_hat"] for s in SEEDS])
ax[0].plot(xs, true_rho, "k--", label="true correlation of the noise")
ax[0].plot(xs, rh.mean(0), color=C["joint"], label="correlation learned by the joint loss")
ax[0].set(title="1. The model learns the correlation", xlabel="x", ylabel="correlation", ylim=(-0.05, 1)); ax[0].legend(fontsize=8)

ax[1].plot(xs, np.ones_like(xs), color=C["separate"], label="separate loss")
ax[1].plot(xs, 1 / (1 - rh.mean(0)), color=C["joint"], label="joint loss: 1 / (1 - learned correlation)")
ax[1].set(title="2. What an error in the effect costs", xlabel="x", ylabel="cost relative to the separate loss"); ax[1].legend(fontsize=8)

steps = [t["step"] for t in get(0, "joint")["track"]]
for l in ("separate", "joint"):
    P = np.array([[t["pehe"] for t in get(s, l)["track"]] for s in SEEDS]); m, se = P.mean(0), P.std(0, ddof=1) / np.sqrt(len(P))
    ax[2].plot(steps, m, color=C[l], label=f"{l} loss"); ax[2].fill_between(steps, m - se, m + se, color=C[l], alpha=0.25)
ax[2].set(title="3. PEHE during training", xlabel="training step", ylabel="PEHE", yscale="log"); ax[2].legend(fontsize=8)

for l in ("separate", "joint"):
    E = np.array([get(s, l)["final"]["tau_err"] for s in SEEDS]).mean(0)
    k = 21; E = np.convolve(E, np.ones(k) / k, mode="same")
    ax[3].plot(xs[k:-k], E[k:-k], color=C[l], label=f"{l} loss")
ax[3].set(title="4. Effect error along x after training", xlabel="x", ylabel="|estimated effect - true effect|"); ax[3].legend(fontsize=8)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"sim5_track.{ext}"), dpi=150)
print("saved", os.path.join(OUT, "sim5_track.png"))
