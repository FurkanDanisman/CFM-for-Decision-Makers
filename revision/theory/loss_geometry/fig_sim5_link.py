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
B1, B2, TH = "#1f4e9c", "#e8a09a", "#a01818"
plt.rcParams.update({"font.size": 13})
fig, ax = plt.subplots(1, 2, figsize=(13, 5))

rh = np.array([get(s, "joint")["rho_hat"] for s in SEEDS]).mean(0)
ax[0].plot(rho, np.ones_like(rho), color=B1, lw=7, label="1D loss")
ax[0].plot(rho, 1 / (1 - rh), color=B2, lw=7, label="2D loss")
ax[0].set(xlabel="correlation ρ", ylabel="cost of an effect error (relative to 1D)", xlim=(0, 1))
ax[0].legend(loc="upper left")

edges = np.linspace(rho[0], rho[-1], 11); mid = (edges[:-1] + edges[1:]) / 2
idx = np.clip(np.digitize(rho, edges) - 1, 0, 9)
for lab, l, c in (("1D", "separate", B1), ("2D", "joint", B2)):
    E = np.array([get(s, l)["tau_err"] for s in SEEDS]) ** 2
    P = np.array([[np.sqrt(E[k, idx == b].mean()) for b in range(10)] for k in range(len(SEEDS))])
    m, se = P.mean(0), P.std(0, ddof=1) / np.sqrt(len(P))
    ax[1].plot(mid, m, color=c, lw=7, label=lab); ax[1].fill_between(mid, m - se, m + se, color=c, alpha=0.3)
ax[1].set(xlabel="correlation ρ", ylabel="PEHE after training", xlim=(0, 1))
ax[1].legend(loc="upper left")
fig.tight_layout()
for ext in ("png", "pdf"): fig.savefig(os.path.join(OUT, f"sim5_link.{ext}"), dpi=150)
for b in range(10): print(f"rho {mid[b]:.2f}")
