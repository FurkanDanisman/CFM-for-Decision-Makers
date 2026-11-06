"""Columns 1-2: separate networks per correlation range (sim6_groups.py), 1D and 2D; column 3: one network on
rho in [0.1, 0.95] (sim5_track.py), 1D and 2D. Narrow network, mean over 10 seeds.
row 1: training loss on fixed test units over training, the loss of a perfect prediction dashed (same colour);
row 2: PEHE after training: columns 1-2 one line per network over its correlation range (mean, SE band);
       column 3 against the correlation in 10 bins.
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
fig, ax = plt.subplots(2, 3, figsize=(PS.TEXTWIDTH, PS.TEXTWIDTH * 0.72))

R6 = json.load(open(os.path.join(OUT, "sim6_results.json")))
for j, (lab, l, _) in enumerate(MOD):
    for (g, gl), col in zip(G.items(), SHADES[l]):
        L = np.array([[t["loss"] for t in R6["runs"][f"{g}|{s}|{l}"]["track"]] for s in SEEDS]).mean(0)
        ax[0, j].plot(steps, L, color=col, lw=2.0, label=gl)
        ax[0, j].axhline(R6["perfect"][f"{g}|{l}"], color=col, lw=1.3, ls="--")
        P = np.sqrt((np.array([R6["runs"][f"{g}|{s}|{l}"]["tau_err"] for s in SEEDS]) ** 2).mean(1))
        m, se = P.mean(), P.std(ddof=1) / np.sqrt(len(P)); lo_, hi_ = R6["ranges"][g]
        ax[1, j].plot([lo_, hi_], [m, m], color=col, lw=2.5); ax[1, j].fill_between([lo_, hi_], m - se, m + se, color=col, alpha=0.25, lw=0)
    ax[0, j].set_title(lab)
    ax[0, j].legend(loc="upper right", **KW)
for lab, l, c in MOD:
    ax[0, 2].plot(steps, mean_track(l, "loss_all"), color=c, lw=2.0, label=lab)
    ax[0, 2].axhline(perfect(l, "all"), color=c, lw=1.3, ls="--")
ax[0, 2].set_title("1D & 2D")
ax[0, 2].text(0.97, 0.95, r"$\rho \in [0.1, 0.95]$", transform=ax[0, 2].transAxes, ha="right", va="top")

xs = np.linspace(-2, 2, 401); rho = 0.10 + 0.85 * (xs + 2) / 4
edges = np.linspace(rho[0], rho[-1], 11); mid = (edges[:-1] + edges[1:]) / 2
idx = np.clip(np.digitize(rho, edges) - 1, 0, 9)
def pehe_by_rho(l):
    E = np.array([run(s, l)["final"]["tau_err"] for s in SEEDS]) ** 2
    P = np.array([[np.sqrt(E[k, idx == b].mean()) for b in range(10)] for k in range(len(SEEDS))])
    return P.mean(0), P.std(0, ddof=1) / np.sqrt(len(P))
for lab, l, c in MOD:
    m, se = pehe_by_rho(l)
    ax[1, 2].plot(mid, m, "-o", color=c, lw=2.0, ms=3, label=lab); ax[1, 2].fill_between(mid, m - se, m + se, color=c, alpha=0.2, lw=0)
ax[1, 2].legend(loc="upper center", **KW)

for a in ax[0]: a.set_ylim(0.25, 2.1); a.set_yticks([0.5, 1, 1.5, 2])
top = max(a.get_ylim()[1] for a in ax[1])
for a in ax[1]: a.set_ylim(0, top); a.set(xlabel=r"correlation $\rho$", xlim=(0, 1), xticks=[0, 0.5, 1])
ax[0, 0].set_ylabel("training loss"); ax[1, 0].set_ylabel("PEHE")
for a in ax[0]:
    a.set(xlabel="training step", xticks=[0, 2000, 4000]); a.set_xticklabels(["0", "2k", "4k"])
for a in ax.flat:
    PS.frame(a); a.tick_params(axis="y", length=2, width=0.5)
fig.tight_layout(w_pad=0.4, h_pad=0.6)
for ext in ("png", "pdf"): fig.savefig(os.path.join(OUT, f"sim5_link.{ext}"), dpi=150)
