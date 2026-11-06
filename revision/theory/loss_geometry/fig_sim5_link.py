"""From the trained networks of sim5_track.py (narrow network, mean over 10 seeds), over training:
row 1: training loss on fixed test units, with the loss of a perfect prediction dashed (same colour):
       1D by correlation group | 2D by correlation group | 1D and 2D over all correlations rho in [0.1, 0.95]
row 2: PEHE on a grid of x, in the same groups / models as the panel above it.
   .venv/bin/python revision/theory/loss_geometry/fig_sim5_link.py OUT_DIR"""
import importlib.util, json, os, sys
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "paper"))
import paper_style as PS

OUT = sys.argv[1]; W = "narrow"; SEEDS = range(10)
R = json.load(open(os.path.join(OUT, "sim5_results.json")))
run = lambda s, l: R[f"{W}|{s}|{l}"]
mean_track = lambda l, k: np.array([[t[k] for t in run(s, l)["track"]] for s in SEEDS]).mean(0)
steps = [t["step"] for t in run(0, "joint")["track"]]

# perfect prediction (true m0, m1, sd 1, true rho) on the same fixed test units as the tracked losses
spec = importlib.util.spec_from_file_location("s5", os.path.join(HERE, "sim5_track.py"))
S5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S5)
n = len(S5.XT)
o = torch.stack([S5.m0(S5.XT)[:, 0], (S5.m0(S5.XT) + S5.tau(S5.XT))[:, 0], torch.zeros(n), torch.zeros(n),
                 torch.atanh(S5.RT / 0.999)], 1)
perfect = lambda l, g: S5.nll_units(o, S5.YT, l)[S5.GROUPS[g]].mean().item()

PS.apply()
G = {"low": r"$\rho \in [0.1, 0.4)$", "medium": r"$\rho \in [0.4, 0.7)$", "high": r"$\rho \in [0.7, 0.95]$"}
SHADES = {"separate": ("#a6c8e8", "#3f7fbf", "#173d63"), "joint": ("#f6c48e", "#e6820e", "#8a4a05")}   # 1D blue, 2D orange
MOD = (("1D", "separate", PS.COL_1D), ("2D", "joint", PS.COL_2D))
KW = dict(frameon=False, handlelength=1.1, handletextpad=0.4, labelspacing=0.05, borderaxespad=0.05)
fig, ax = plt.subplots(2, 3, figsize=(PS.TEXTWIDTH, PS.TEXTWIDTH * 0.72), sharex=True)

for j, (lab, l, _) in enumerate(MOD):
    for (g, gl), col in zip(G.items(), SHADES[l]):
        ax[0, j].plot(steps, mean_track(l, f"loss_{g}"), color=col, lw=2.0, label=gl)
        ax[0, j].axhline(perfect(l, g), color=col, lw=1.3, ls="--")
        ax[1, j].plot(steps, mean_track(l, f"pehe_{g}"), color=col, lw=2.0)
    ax[0, j].set_title(lab)
    ax[0, j].legend(loc="upper right", **KW)
for lab, l, c in MOD:
    ax[0, 2].plot(steps, mean_track(l, "loss_all"), color=c, lw=2.0, label=lab)
    ax[0, 2].axhline(perfect(l, "all"), color=c, lw=1.3, ls="--")
    ax[1, 2].plot(steps, mean_track(l, "pehe_all"), color=c, lw=2.0)
ax[0, 2].set_title(r"$\rho \in [0.1, 0.95]$")
ax[0, 2].legend(loc="upper right", **KW)

for a in ax[0]: a.set_ylim(0.25, 2.1); a.set_yticks([0.5, 1, 1.5, 2])
top = max(a.get_ylim()[1] for a in ax[1])
for a in ax[1]: a.set_ylim(0, top); a.set_xlabel("training step")
ax[0, 0].set_ylabel("training loss"); ax[1, 0].set_ylabel("PEHE")
for a in ax.flat:
    a.set_xticks([0, 2000, 4000]); a.set_xticklabels(["0", "2k", "4k"])
    PS.frame(a); a.tick_params(axis="y", length=2, width=0.5)
fig.tight_layout(w_pad=0.4, h_pad=0.6)
for ext in ("png", "pdf"): fig.savefig(os.path.join(OUT, f"sim5_link.{ext}"), dpi=150)
for lab, l, _ in MOD:
    for g in list(G) + ["all"]:
        L, P = mean_track(l, f"loss_{g}"), mean_track(l, f"pehe_{g}")
        print(f"{lab} {g:6s}: loss above perfect start {L[0] - perfect(l, g):.3f} end {L[-1] - perfect(l, g):.3f};  "
              f"PEHE start {P[0]:.3f} end {P[-1]:.3f}")
