"""Figures for sim8_noise.py (narrow network, mean over 30 seeds), the same format as fig_sim5_link.py:
row 1 training loss with the best loss possible from x dashed (perfect_from_x.py), row 2 PEHE bars; columns 1D per noise-size range,
2D per range, 1D & 2D over the whole range.
   .venv/bin/python revision/theory/loss_geometry/fig_sim8.py MODE OUT_DIR      (MODE: cpfn or dopfn)"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "paper"))
import paper_style as PS

MODE, OUT = sys.argv[1], sys.argv[2]; SEEDS = range(30)
R = json.load(open(os.path.join(OUT, f"sim8_{MODE}_results.json")))
NZ = r"\eta" if MODE == "cpfn" else r"\epsilon"
run = lambda st, s, l: R["runs"][f"{st}|{s}|{l}"]
loss_curve = lambda st, l: np.array([[t["loss"] for t in run(st, s, l)["track"]] for s in SEEDS]).mean(0)
steps = [t["step"] for t in run("low", 0, "joint")["track"]]
def pehe(st, l):
    P = np.array([run(st, s, l)["pehe"] for s in SEEDS]); return P.mean(), P.std(ddof=1) / np.sqrt(len(P))

PS.apply()
G = {k: rf"sd$({NZ}) \in {v}$" for k, v in (("low", "[0.25, 0.75)"), ("medium", "[0.75, 1.25)"), ("high", "[1.25, 1.75]"))}
SHADES = {"separate": ("#a6c8e8", "#3f7fbf", "#173d63"), "joint": ("#f6c48e", "#e6820e", "#8a4a05")}
MOD = (("1D", "separate", PS.COL_1D), ("2D", "joint", PS.COL_2D))
KW = dict(frameon=False, handlelength=1.1, handletextpad=0.4, labelspacing=0.05, borderaxespad=0.05)
fig, ax = plt.subplots(2, 3, figsize=(PS.TEXTWIDTH, PS.TEXTWIDTH * 0.72))

for j, (lab, l, c) in enumerate(MOD):
    for (g, gl), col in zip(G.items(), SHADES[l]):
        ax[0, j].plot(steps, loss_curve(g, l), color=col, lw=2.0, label=gl)
        ax[0, j].axhline(R["perfect_x"][f"{g}|{l}"], color=col, lw=1.3, ls="--")
    ax[0, j].set_title(lab); ax[0, j].legend(loc="upper right", **KW)
    M = np.array([pehe(g, l) for g in G])
    PS.hbars(ax[1, j], M[:, 0], M[:, 1], list(SHADES[l]))
for lab, l, c in MOD:
    ax[0, 2].plot(steps, loss_curve("full", l), color=c, lw=2.0)
    ax[0, 2].axhline(R["perfect_x"][f"full|{l}"], color=c, lw=1.3, ls="--")
ax[0, 2].set_title("1D & 2D")
ax[0, 2].text(0.97, 0.95, rf"sd$({NZ}) \in [0.25, 1.75]$", transform=ax[0, 2].transAxes, ha="right", va="top")
M = np.array([pehe("full", l) for _, l, _ in MOD])
PS.hbars(ax[1, 2], M[:, 0], M[:, 1], [c for _, _, c in MOD], height=0.55)

for a in ax[0]:
    a.set(xlabel="training step", xticks=[0, 2000, 4000]); a.set_xticklabels(["0", "2k", "4k"])
    PS.frame(a); a.tick_params(axis="y", length=2, width=0.5)
lo = min(a.get_ylim()[0] for a in ax[0]); hi = max(a.get_ylim()[1] for a in ax[0])
for a in ax[0]: a.set_ylim(lo, hi + 0.35 * (hi - lo))
ax[0, 0].set_ylabel("training loss")
for a in ax[1]:
    a.set_xlim(0, 0.3); a.set_xticks([0, 0.1, 0.2, 0.3]); a.set_xlabel("PEHE")
for a in ax[1]:
    a.set_yticks([])
fig.tight_layout(w_pad=0.4, h_pad=0.6)
for ext in ("png", "pdf"): fig.savefig(os.path.join(OUT, f"sim8_{MODE}.{ext}"), dpi=150)
