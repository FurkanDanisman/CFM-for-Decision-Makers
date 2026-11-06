"""Two panels from the SAME trained networks of sim5_track.py (narrow network, 10 seeds), both against the correlation:
left : cost of an effect error under each loss, relative to 1D (2D: 1 / (1 - rho learned by the 2D network)).
right: PEHE of the trained 1D and 2D networks, in 10 bins of the correlation (mean and SE over seeds).
   .venv/bin/python revision/theory/loss_geometry/fig_sim5_link.py OUT_DIR"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]; W = "narrow"; SEEDS = range(10)
R = json.load(open(os.path.join(OUT, "sim5_results.json")))
get = lambda s, l: R[f"{W}|{s}|{l}"]["final"]
xs = np.linspace(-2, 2, 401); rho = 0.10 + 0.85 * (xs + 2) / 4
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "paper"))
import paper_style as PS
PS.apply()
B1, B2 = PS.COL_1D, PS.COL_2D
fig, ax = plt.subplots(1, 3, figsize=(PS.TEXTWIDTH * 1.35, PS.TEXTWIDTH * 0.38))

for lab, l, c in (("1D", "separate", B1), ("2D", "joint", B2)):
    T = R[f"{W}|0|{l}"]["track"]; steps = [t["step"] for t in T]
    Lm = np.array([[t["loss"] for t in R[f"{W}|{s}|{l}"]["track"]] for s in SEEDS]).mean(0)
    ax[0].plot(steps, Lm, color=c, lw=1.8, label=f"{lab} loss")
ax[0].set(xlabel="training step", ylabel="training loss")
ax[0].legend(loc="upper right", frameon=False)
ax = ax[1:]

rh = np.array([get(s, "joint")["rho_hat"] for s in SEEDS]).mean(0)
ax[0].plot(rho, np.ones_like(rho), color=B1, lw=1.8, label="1D loss")
ax[0].plot(rho, 1 / (1 - rh), color=B2, lw=1.8, label="2D loss")
ax[0].plot(rho, 1 / (1 - rho), "--", color="k", lw=1.0, label="Theory prediction")
ax[0].set(xlabel=r"correlation $\rho$", ylabel="cost of an effect error\n(relative to 1D)", xlim=(0, 1))
ax[0].legend(loc="upper left", frameon=False)

edges = np.linspace(rho[0], rho[-1], 11); mid = (edges[:-1] + edges[1:]) / 2
idx = np.clip(np.digitize(rho, edges) - 1, 0, 9)
for lab, l, c in (("1D", "separate", B1), ("2D", "joint", B2)):
    E = np.array([get(s, l)["tau_err"] for s in SEEDS]) ** 2
    P = np.array([[np.sqrt(E[k, idx == b].mean()) for b in range(10)] for k in range(len(SEEDS))])
    m, se = P.mean(0), P.std(0, ddof=1) / np.sqrt(len(P))
    ax[1].plot(mid, m, "-o", color=c, lw=1.8, ms=3, label=lab); ax[1].fill_between(mid, m - se, m + se, color=c, alpha=0.2, lw=0)
ax[1].set(xlabel=r"correlation $\rho$", ylabel="PEHE after training", xlim=(0, 1))
ax[1].legend(loc="upper center", frameon=False)
for a in fig.axes:
    PS.frame(a); a.tick_params(axis="y", length=2, width=0.5)
fig.tight_layout()
for ext in ("png", "pdf"): fig.savefig(os.path.join(OUT, f"sim5_link.{ext}"), dpi=150)
