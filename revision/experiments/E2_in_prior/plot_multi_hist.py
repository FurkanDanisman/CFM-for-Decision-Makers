"""Per-SCM coverage (multi.sh, e2_multi_world.csv): number of SCMs per coverage bin.
python plot_multi_hist.py e2_multi_world.csv out_prefix
Bins of width 0.01 on [0.85, 1), plus one bar for coverage below 0.85 and one for coverage exactly 1;
the band 0.93-0.97 around the nominal 0.95 is shaded. Counts are written on the bars."""
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

d = pd.read_csv(sys.argv[1]); d = d[d.truth == "delta_shared"]; out = sys.argv[2]
models = list(dict.fromkeys(d.model))
edges = np.round(np.arange(0.85, 1.0001, 0.01), 2)                    # 0.85, 0.86, ..., 1.00
labels = ["<0.85"] + [f"{a:.2f}" for a in edges[:-1]] + ["1"]
fig, axes = plt.subplots(len(models), 1, figsize=(8.2, 2.5 * len(models)), sharex=True)
for ax, m in zip(np.atleast_1d(axes), models):
    c = d[d.model == m].coverage.values
    one = c >= 1 - 1e-9
    cnt = [int((c < 0.85).sum())] + [int(((c >= a) & (c < b) & ~one).sum()) for a, b in zip(edges[:-1], edges[1:])] + [int(one.sum())]
    x = np.arange(len(cnt))
    near = [(a >= 0.93 - 1e-9) and (a < 0.97 - 1e-9) for a in edges[:-1]]
    ax.axvspan(1 + near.index(True) - 0.5, 1 + len(near) - near[::-1].index(True) - 0.5, color="#1F4E9C", alpha=0.08, lw=0)
    ax.bar(x, cnt, width=0.8, color="#1F4E9C", alpha=0.85)
    for xi, k in zip(x, cnt):
        if k: ax.text(xi, k + max(cnt) * 0.02, str(k), ha="center", va="bottom", fontsize=8)
    ax.set_title(f"{m}: {len(c)} SCMs, {int(((c >= 0.93) & (c <= 0.97)).sum())} with coverage in [0.93, 0.97]", fontsize=10)
    ax.set_ylabel("number of SCMs"); ax.set_ylim(0, max(cnt) * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=60, fontsize=8)
ax.set_xlabel("coverage of the 95% interval per SCM (bin start; shaded: 0.93–0.97)")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{out}.{ext}", dpi=200)
print("wrote", out + ".{png,pdf}")
