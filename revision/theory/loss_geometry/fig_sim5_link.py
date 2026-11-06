"""From the trained networks of sim5_track.py (narrow network, 10 seeds):
panels 1-3: training loss of the 1D and the 2D network over training, on fixed test units with low (0.1-0.4), medium
(0.4-0.7) and high (0.7-0.95) correlation, each network's own loss as it is (mean over seeds);
panel 4: PEHE of the trained networks in 10 bins of the correlation (mean and SE over seeds).
   .venv/bin/python revision/theory/loss_geometry/fig_sim5_link.py OUT_DIR"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "paper"))
import paper_style as PS

OUT = sys.argv[1]; W = "narrow"; SEEDS = range(10)
R = json.load(open(os.path.join(OUT, "sim5_results.json")))
run = lambda s, l: R[f"{W}|{s}|{l}"]
xs = np.linspace(-2, 2, 401); rho = 0.10 + 0.85 * (xs + 2) / 4
PS.apply()
MOD = (("1D", "separate", PS.COL_1D), ("2D", "joint", PS.COL_2D))
fig, ax = plt.subplots(1, 3, figsize=(PS.TEXTWIDTH, PS.TEXTWIDTH * 0.34))

# perfect prediction (true m0, m1, sd 1, true rho) on the same fixed test units as the tracked losses
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = [sys.argv[0], OUT]
import importlib.util
spec = importlib.util.spec_from_file_location("s5", os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim5_track.py"))
S5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S5)
o = torch.stack([S5.m0(S5.XT)[:, 0], (S5.m0(S5.XT) + S5.tau(S5.XT))[:, 0], torch.zeros(len(S5.XT)), torch.zeros(len(S5.XT)),
                 torch.atanh(S5.RT / 0.999)], 1)
G = {"low": (r"$\rho$ 0.1 to 0.4", "#9ecae1"), "medium": (r"$\rho$ 0.4 to 0.7", "#4292c6"), "high": (r"$\rho$ 0.7 to 0.95", "#08306b")}
steps = [t["step"] for t in run(0, "joint")["track"]]
for a, (lab, l, _) in zip(ax[:2], MOD):
    perf = S5.nll_units(o, S5.YT, l)
    for g, (gl, col) in G.items():
        L = np.array([[t[f"loss_{g}"] for t in run(s, l)["track"]] for s in SEEDS]).mean(0)
        a.plot(steps, L, color=col, lw=1.5, label=gl)
        pv = perf[S5.GROUPS[g]].mean().item(); a.axhline(pv, color=col, lw=1.0, ls="--")
        print(f"{lab} {g:6s}: perfect {pv:.3f}  start {L[0]:.3f} (above perfect {L[0] - pv:.3f})  end {L[-1]:.3f} (above perfect {L[-1] - pv:.3f})")
    a.set(title=lab, xlabel="training step", xticks=[0, 2000, 4000]); a.set_xticklabels(["0", "2k", "4k"])
ax[0].set_ylabel("training loss")
lo = min(a.get_ylim()[0] for a in ax[:2]); hi = max(a.get_ylim()[1] for a in ax[:2])
for a in ax[:2]: a.set_ylim(0.0, hi); a.set_yticks([0.5, 1, 1.5])
KW = dict(frameon=False, handlelength=1.1, handletextpad=0.4, labelspacing=0.2, borderaxespad=0.2)
ax[0].legend(loc="lower left", **{**KW, "labelspacing": 0.05, "borderaxespad": 0.05})
GAP = {}
ax = list(ax[:2]) + [None, ax[2]]

edges = np.linspace(rho[0], rho[-1], 11); mid = (edges[:-1] + edges[1:]) / 2
idx = np.clip(np.digitize(rho, edges) - 1, 0, 9)
for lab, l, c in MOD:
    E = np.array([run(s, l)["final"]["tau_err"] for s in SEEDS]) ** 2
    P = np.array([[np.sqrt(E[k, idx == b].mean()) for b in range(10)] for k in range(len(SEEDS))])
    m, se = P.mean(0), P.std(0, ddof=1) / np.sqrt(len(P))
    ax[3].plot(mid, m, "-o", color=c, lw=1.5, ms=2.5, label=lab); ax[3].fill_between(mid, m - se, m + se, color=c, alpha=0.2, lw=0)
ax[3].set(xlabel=r"correlation $\rho$", ylabel="PEHE", xlim=(0, 1), xticks=[0, 0.5, 1])
ax[3].legend(loc="upper center", **KW)
for a in fig.axes:
    PS.frame(a); a.tick_params(axis="y", length=2, width=0.5)
fig.tight_layout(w_pad=0.4)
for ext in ("png", "pdf"): fig.savefig(os.path.join(OUT, f"sim5_link.{ext}"), dpi=150)
