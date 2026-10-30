"""The remaining result figures in the paper's figure style (paper_style.py):

  twins_mse.pdf            Twins counterfactual MSE, d = 1 and d = 10 (bodik_twins_bodiksetup.csv)
  rho_pred_inprior_hist.pdf  predicted cross-world correlation of the 2D models in their own prior (rho_pred_inprior.csv)
  ablation_point_{pehe,eps}.pdf  matched-label / matched-resolution ablations and the 2D model (ablation_point.csv)

    python revision/paper/style_plots.py RESULTS_DIR OUT_DIR      (RESULTS_DIR = revision/results/d262t1)
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as PS                                         # noqa: E402

RES, OUT = sys.argv[1:3]
PS.apply()
os.makedirs(OUT, exist_ok=True)
FAMS = ["Do-PFN", "UWYK No-Anc", "UWYK Anc", "CausalPFN-C"]


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print("wrote", os.path.join(OUT, f"{name}.pdf"))


def row_label(fig, ax, text, x=0.012):
    p = ax.get_position()
    fig.text(x, (p.y0 + p.y1) / 2, text, rotation=90, ha="center", va="center", fontweight="bold",
             fontsize=PS.LABEL + 0.5)


# ---- Twins
tw = pd.read_csv(os.path.join(RES, "bodik_twins_bodiksetup.csv")).set_index(["setting", "model"])
TW = [("Do-PFN", "Do-PFN"), ("Do-PFN 2D", "Do-PFN 2D"), ("UWYK No-Anc", "UWYK No-Anc"),
      ("UWYK No-Anc 2D", "UWYK 2D No-Anc"), ("UWYK Anc", "UWYK Anc"), ("UWYK Anc 2D", "UWYK 2D Anc"),
      ("CausalPFN-C", "CausalPFN-C"), ("CausalPFN-C 2D", "CausalPFN-C 2D")]
fig, axes = plt.subplots(1, 2, figsize=(PS.TEXTWIDTH * 0.72, 2.3), sharey=True, gridspec_kw=dict(wspace=0.18))
for ax, d in zip(axes, ("d=1", "d=10")):
    y = PS.hbars(ax, [tw.at[(d, k), "mse"] for _, k in TW], [tw.at[(d, k), "mse_se"] for _, k in TW],
                 [PS.COL_1D, PS.COL_2D] * 4)
    ax.set_title(f"$d = {d[2:]}$", pad=3)
axes[0].set_yticks(y)
axes[0].set_yticklabels([m for m, _ in TW])
fig.supxlabel("Mean squared error", fontsize=PS.LABEL, x=(0.33 + 0.98) / 2, y=0.0)
fig.subplots_adjust(left=0.33, right=0.98, top=0.88, bottom=0.2)
save(fig, "twins_mse")

# ---- predicted rho in the own prior
rp = pd.read_csv(os.path.join(RES, "rho_pred_inprior.csv"))
models = list(dict.fromkeys(rp["model"]))
fig, axes = plt.subplots(1, len(models), figsize=(PS.TEXTWIDTH, 2.1), sharey=True, gridspec_kw=dict(wspace=0.12))
bins = np.linspace(-0.5, 1.0, 31)
for ax, m in zip(axes, models):
    v = rp.loc[rp["model"] == m, "rho_median"].to_numpy()
    w = np.full(v.size, 1 / v.size)
    ax.hist(v, bins=bins, weights=w, color=PS.COL_2D, edgecolor="black", linewidth=0.5)
    ax.set_title(m, pad=3)
    ax.set_xlim(-0.5, 1.0)
    ax.set_xticks([-0.5, 0, 0.5, 1])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(PS.fmt))
    PS.frame(ax)
    ax.tick_params(axis="y", length=2, width=0.5)
axes[0].set_ylabel("Probability")
fig.supxlabel(r"Predicted $\rho$ (median over queries)", fontsize=PS.LABEL, y=0.0)
fig.subplots_adjust(left=0.1, right=0.98, top=0.8, bottom=0.25)
save(fig, "rho_pred_inprior_hist")

# ---- ablations
ab = list(csv.DictReader(open(os.path.join(RES, "ablation_point.csv"))))
AB = {(r["benchmark"], r["setting"], r["model"], r["variant"]): r for r in ab}
VAR = [("matched label", ""), ("matched resolution", "///"), ("2D", "")]
LAB = {"matched label": "(matched label)", "matched resolution": "(matched res.)", "2D": "2D"}
BENCH = [("RealCause", ["IHDP", "ACIC", "CPS", "PSID", "PSID_bal"], ["IHDP", "ACIC", "CPS", "PSID", "PSID-B"]),
         ("CaseStudy", ["Observed_Confounder", "Observed_Mediator", "Observed_Mediator_and_Confounder",
                        "Unobserved_Confounder", "Frontdoor_Criterion", "Backdoor_Criterion"],
          ["Obs.\nConf.", "Obs.\nMed.", "Obs.\nM+C", "Unobs.\nConf.", "Front-\ndoor", "Back-\ndoor"]),
         ("ComplexMech", ["5", "10", "20", "30", "40", "50"], [f"$n={n}$" for n in (5, 10, 20, 30, 40, 50)])]
present = {k[1] for k in AB if k[0] == "RealCause"}
if "PSID_bal" not in present:                                    # setting name of PSID (balanced) in the csv
    BENCH[0][1][4] = next(s for s in present if s not in BENCH[0][1][:4])
for metric, se, lab in (("pehe", "pehe_se", r"$\sqrt{\mathrm{PEHE}}$"), ("eps_ate", "eps_se", r"$\epsilon_{\mathrm{ATE}}$")):
    fig, axes = plt.subplots(3, 6, figsize=(PS.TEXTWIDTH, 7.6), sharey=True,
                             gridspec_kw=dict(wspace=0.32, hspace=0.42))
    for r, (bm, keys, titles) in enumerate(BENCH):
        for c in range(6):
            ax = axes[r, c]
            if c >= len(keys):
                ax.axis("off")
                continue
            rows = [AB[(bm, keys[c], f, v)] for f in FAMS for v, _ in VAR]
            y = PS.hbars(ax, [float(x[metric]) for x in rows], [float(x[se]) for x in rows],
                         [PS.COL_2D if x["variant"] == "2D" else PS.COL_1D for x in rows], height=0.82)
            for bar, x in zip(ax.patches, rows):
                if x["variant"] == "matched resolution":
                    bar.set_hatch("////")
                    bar.set_edgecolor("black")
            ax.set_title(titles[c], pad=3, linespacing=1.0)
        axes[r, 0].set_yticks(y)
        axes[r, 0].set_yticklabels([f"{f} {LAB[v]}" if v != "2D" else f"{f} 2D" for f in FAMS for v, _ in VAR],
                                   fontsize=PS.YTICK - 1.5)
    plt.rcParams["hatch.linewidth"] = 0.6
    fig.subplots_adjust(left=0.3, right=0.985, top=0.95, bottom=0.06)
    for r, (bm, _, _) in enumerate(BENCH):
        row_label(fig, axes[r, 0], {"CaseStudy": "Case studies"}.get(bm, bm))
    fig.supxlabel(lab, fontsize=PS.LABEL, x=(0.3 + 0.985) / 2, y=0.0)
    save(fig, f"ablation_point_{'eps' if metric == 'eps_ate' else 'pehe'}")
