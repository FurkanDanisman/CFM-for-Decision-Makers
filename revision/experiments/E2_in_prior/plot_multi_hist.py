"""Histogram over SCMs of the per-world coverage (multi.sh, e2_multi_world.csv).
python plot_multi_hist.py e2_multi_world.csv out_prefix
One panel per model; bars = share of worlds per coverage bin (shared truth), outline = independent truth;
dashed line at the nominal 0.95."""
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

d = pd.read_csv(sys.argv[1]); out = sys.argv[2]
models = list(dict.fromkeys(d.model))
bins = np.linspace(0, 1, 21)
fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.0), sharey=True)
for ax, m in zip(np.atleast_1d(axes), models):
    s = d[(d.model == m) & (d.truth == "delta_shared")].coverage.values
    i = d[(d.model == m) & (d.truth == "delta_indep")].coverage.values
    ax.hist(s, bins=bins, weights=np.ones_like(s) / len(s), color="#5b3a8a", alpha=.75, label="shared noise")
    ax.hist(i, bins=bins, weights=np.ones_like(i) / len(i), histtype="step", color="#333333", lw=1.5, label="independent noise")
    ax.axvline(0.95, color="#c0392b", ls="--", lw=1.2)
    ax.set_title(f"{m}  ({len(s)} SCMs)", fontsize=11); ax.set_xlabel("coverage of the 95% interval")
    ax.spines[["top", "right"]].set_visible(False)
np.atleast_1d(axes)[0].set_ylabel("share of SCMs"); np.atleast_1d(axes)[0].legend(frameon=False, fontsize=9, loc="upper left")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{out}.{ext}", dpi=200)
print("wrote", out + ".{png,pdf}")
