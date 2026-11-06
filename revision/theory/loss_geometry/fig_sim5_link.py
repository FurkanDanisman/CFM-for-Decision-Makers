"""Two panels from the SAME trained networks of sim5_track.py (narrow network, 10 seeds), both against the correlation:
middle: loss of a prediction whose effect is wrong by 1 (1D: 1.25; 2D: 1 + 0.25 / (1 - rho learned by the 2D network);
        theory: 1 + 0.25 / (1 - true rho)).
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
fig, ax = plt.subplots(1, 3, figsize=(PS.TEXTWIDTH, PS.TEXTWIDTH * 0.36))

for lab, l, c in (("1D", "separate", B1), ("2D", "joint", B2)):
    T = R[f"{W}|0|{l}"]["track"]; steps = [t["step"] for t in T]
    Lm = np.array([[t["loss"] for t in R[f"{W}|{s}|{l}"]["track"]] for s in SEEDS]).mean(0)
    ax[0].plot(steps, Lm, color=c, lw=1.8, label=lab)
ax[0].set(xlabel="training step", ylabel="training loss", xticks=[0, 2000, 4000]); ax[0].set_xticklabels(["0", "2k", "4k"])
ax = ax[1:]

rh = np.array([get(s, "joint")["rho_hat"] for s in SEEDS]).mean(0)
c = 0.25                                   # error part of the 1D loss for a prediction whose effect is wrong by 1
ax[0].plot(rho, 1 + c * np.ones_like(rho), color=B1, lw=1.8, label="1D")
ax[0].plot(rho, 1 + c / (1 - rh), color=B2, lw=1.8, label="2D")
ax[0].plot(rho, 1 + c / (1 - rho), "--", color="k", lw=1.0, label="Theory")
ax[0].set(xlabel=r"correlation $\rho$", ylabel="loss", xlim=(0, 1), xticks=[0, 0.5, 1])

edges = np.linspace(rho[0], rho[-1], 11); mid = (edges[:-1] + edges[1:]) / 2
idx = np.clip(np.digitize(rho, edges) - 1, 0, 9)
for lab, l, c in (("1D", "separate", B1), ("2D", "joint", B2)):
    E = np.array([get(s, l)["tau_err"] for s in SEEDS]) ** 2
    P = np.array([[np.sqrt(E[k, idx == b].mean()) for b in range(10)] for k in range(len(SEEDS))])
    m, se = P.mean(0), P.std(0, ddof=1) / np.sqrt(len(P))
    ax[1].plot(mid, m, "-o", color=c, lw=1.8, ms=3); ax[1].fill_between(mid, m - se, m + se, color=c, alpha=0.2, lw=0)
ax[1].set(xlabel=r"correlation $\rho$", ylabel="PEHE", xlim=(0, 1), xticks=[0, 0.5, 1])
for a in fig.axes:
    PS.frame(a); a.tick_params(axis="y", length=2, width=0.5)
h, l = fig.axes[1].get_legend_handles_labels()
fig.axes[0].legend(h, l, loc="upper right", frameon=False, handlelength=1.1, handletextpad=0.4, labelspacing=0.2, borderaxespad=0.2)
fig.axes[0].set_ylim(0.7, 2.3); fig.axes[0].set_yticks([1, 1.5, 2])
fig.tight_layout(w_pad=0.6)
for ext in ("png", "pdf"): fig.savefig(os.path.join(OUT, f"sim5_link.{ext}"), dpi=150)
