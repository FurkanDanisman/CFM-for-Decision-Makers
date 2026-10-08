"""Experiment A figure: the paper's own Figure 3 (row 1) panels on top, our Do-PFN (v1)
bar at N=200 underneath, drawn on the same x-axis (same limits, ticks, style).

    python plot_exp_A_paper.py --res results/exp_A --paper PDFs/dopfnpaper.pdf
"""
import argparse
import os
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--res", default="results/exp_A")
p.add_argument("--paper", default="PDFs/dopfnpaper.pdf")
p.add_argument("--n", type=int, default=0, help="0 = the single N present for each case")
p.add_argument("--boot", type=int, default=2000)
a = p.parse_args()

# Page 8 of the paper rendered at 400 dpi. Per case: crop box of the paper's panel
# (x0, y0, x1, y1) and the pixel x of the panel's left spine (= 0) and right spine,
# plus the x value at the right spine (from the 0.000 / 0.010 and 0.00 / 0.10 ticks).
# paper_med / paper_hi: the paper's Do-PFN (v1) median and upper CI, read off the figure
# by pixel measurement (approximate); row_y: that bar's row in the crop, for the label.
PANELS = {
    "Observed_Confounder":   dict(crop=(680, 395, 1170, 800), spine=(889, 1163), xmax=0.0137,
                                  ticks=[0.0, 0.005, 0.010], fmt="{:.3f}", lab="{:.4f}", paper_med=0.0010,
                                  paper_hi=0.0016, row_y=115),
    "Unobserved_Confounder": dict(crop=(1830, 395, 2152, 800), spine=(1866, 2140), xmax=0.1097,
                                  ticks=[0.0, 0.05, 0.10], fmt="{:.2f}", lab="{:.3f}", paper_med=0.028,
                                  paper_hi=0.031, row_y=115),
}
RED, ERR = "#ca3335", "#424242"

with tempfile.TemporaryDirectory() as tmp:
    subprocess.run(["pdftoppm", "-f", "8", "-l", "8", "-r", "400", "-png", a.paper,
                    os.path.join(tmp, "p")], check=True)
    page = Image.open(os.path.join(tmp, "p-08.png")).convert("RGB")
    crops = {c: page.crop(v["crop"]) for c, v in PANELS.items()}

df = pd.read_csv(os.path.join(a.res, "per_dataset.csv"))
rng = np.random.default_rng(0)

plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                     "axes.formatter.use_mathtext": True, "axes.unicode_minus": False})
widths = [PANELS[c]["crop"][2] - PANELS[c]["crop"][0] for c in PANELS]
W = 10.0
scale = W / (sum(widths) + 40)          # inches per paper pixel
img_h = (800 - 395) * scale
slot = (751 - 487) / 4 * scale          # height of one bar row in the paper panel
H = 0.45 + img_h + 0.55 + slot + 0.75
fig = plt.figure(figsize=(W, H), facecolor="white")
x_left = 20 * scale
for case, w in zip(PANELS, widths):
    P = PANELS[case]
    x0, y0, x1, y1 = P["crop"]
    h = y1 - y0
    # paper panel, unchanged
    ax_img = fig.add_axes([x_left / W, 1 - (0.45 + img_h) / H, w * scale / W, img_h / H])
    ax_img.imshow(crops[case]); ax_img.axis("off")
    px_per_x = (P["spine"][1] - P["spine"][0]) / P["xmax"]
    ax_img.annotate("median $\\approx$ " + P["lab"].format(P["paper_med"]),
                    (P["spine"][0] - x0 + P["paper_hi"] * px_per_x + 12, P["row_y"]),
                    va="center", fontsize=12, color="black")
    # ours, aligned so its 0 and right spine sit under the paper's spines
    l = x_left + (P["spine"][0] - x0) * scale
    r = x_left + (P["spine"][1] - x0) * scale
    ax = fig.add_axes([l / W, 0.75 / H, (r - l) / W, slot / H])
    dc = df[df.case == case]
    n = a.n or int(dc.n.unique().item())
    x = dc[dc.n == n].nmse_cid.values
    med = np.median(x)
    lo, hi = np.quantile(np.median(rng.choice(x, (a.boot, len(x))), axis=1), [0.025, 0.975])
    ax.barh([0], [med], height=0.8, color=RED, edgecolor="black", linewidth=1.3)
    ax.errorbar([med], [0], xerr=[[med - lo], [hi - med]], fmt="none", ecolor=ERR,
                elinewidth=2.6, capsize=5, capthick=2.6)
    ax.annotate("median = " + P["lab"].format(med), (hi, 0), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=12, color="black")
    ax.set_xlim(0, P["xmax"]); ax.set_ylim(-0.5, 0.5)
    ax.set_xticks(P["ticks"], [P["fmt"].format(t) for t in P["ticks"]], fontsize=11)
    ax.minorticks_on(); ax.tick_params(axis="y", which="both", left=False)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.set_yticks([0], ["Do-PFN (v1)\nours"] if case == "Observed_Confounder" else [""],
                  fontsize=11)
    ax.set_xlabel("MSE (CID)", fontsize=12)
    ax.set_title(f"N = {n}", fontsize=11, loc="right")
    print(f"{case}: ours N={n} median={med:.4g} 95% CI=[{lo:.4g}, {hi:.4g}]")
    x_left += w * scale + 20 * scale

fig.text(0.01, 1 - 0.12 / H, "Paper, Figure 3 (row 1), unchanged", fontsize=12, va="top")
fig.text(0.01, (0.75 + slot + 0.42) / H, "Ours: Do-PFN (v1), paper protocol, median and 95% CI over 100 datasets, same axes",
         fontsize=12, va="top")
out = os.path.join(a.res, "exp_A_paper_vs_ours.png")
fig.savefig(out, dpi=200, facecolor="white")
print(out)
