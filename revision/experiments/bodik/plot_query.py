"""Plot one Twins query's predicted histogram of the other twin's outcome, one panel per model.
python plot_query.py query.npz out_prefix          (query.npz from extract_query.py)
Writes out_prefix_main.png (the paper's 8 models) and out_prefix_appendix.png (ablations, published CausalPFN, UWYK-B).
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

z = np.load(sys.argv[1]); out = sys.argv[2]
labels = list(dict.fromkeys(k.split("|")[0] for k in z.files if "|" in k))
main = [l for l in labels if "matched" not in l and "published" not in l and "UWYK-B" not in l]
rest = [l for l in labels if l not in main]
known = "heavier" if int(z["t"]) == 1 else "lighter"
other = "lighter" if int(z["t"]) == 1 else "heavier"
lo_x, hi_x = -0.5, 1.5


def panel(ax, lab):
    e, w = z[f"{lab}|edges"], z[f"{lab}|w"]
    tails = z[f"{lab}|tails"] if f"{lab}|tails" in z.files else np.zeros(2)
    c = 0.5 * (e[:-1] + e[1:])
    above = (w * np.clip((e[1:] - 0.5) / (e[1:] - e[:-1]), 0, 1)).sum() + tails[1] * (e[-1] >= 0.5) + tails[0] * (e[0] > 0.5)
    outside = w[(e[1:] <= lo_x) | (e[:-1] >= hi_x)].sum()
    ax.bar(e[:-1], w, width=np.diff(e), align="edge", color="#7b4fa3", edgecolor="white", linewidth=0.3)
    if tails.sum() > 0:
        ax.bar([e[0] - 0.08, e[-1]], [tails[0], tails[1]], width=0.08, align="edge", color="#e07b28",
               label="tail mass beyond the grid")
        ax.legend(fontsize=7, loc="upper center")
    for x in (0, 1):
        ax.axvline(x, color="black", lw=1)
    ax.axvline(0.5, color="grey", lw=0.8, ls="--")
    ax.set_xlim(lo_x, hi_x)
    ax.set_title(f"{lab}\nP(dies) = {above:.3f}   ({len(w)} bins" + (f", {outside:.2f} of mass outside the plot)" if outside > 0.005 else ")"),
                 fontsize=8)
    ax.set_xlabel(f"{other} twin's outcome", fontsize=8); ax.set_ylabel("probability per bin", fontsize=8)
    ax.tick_params(labelsize=7)


for name, labs in (("main", main), ("appendix", rest)):
    if not labs:
        continue
    nc = 4; nr = int(np.ceil(len(labs) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.2 * nr), squeeze=False)
    for ax, lab in zip(axs.ravel(), labs):
        panel(ax, lab)
    for ax in axs.ravel()[len(labs):]:
        ax.axis("off")
    fig.suptitle(f"One Twins query (d = 10): the {known} twin's outcome is known ({float(z['y_known']):.0f}); "
                 f"predicted histogram of the {other} twin's outcome (truth: {float(z['y_other']):.0f}). "
                 "Black lines: the only real outcomes, 0 and 1.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{out}_{name}.png", dpi=130)
    print("wrote", f"{out}_{name}.png")
