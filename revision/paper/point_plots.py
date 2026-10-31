"""Point-estimation figures in the paper's figure style (paper_style.py): one panel per dataset, eight horizontal
bars (each family's single-outcome model in blue, its 2D model in orange, mean +- SE), each panel its own axis.

  point_realcause.pdf   Table tab:rc-point as a figure: rows sqrt(PEHE), eps_ATE; columns the five datasets
  pehe_cs_cm.pdf        sqrt(PEHE): case studies (top), ComplexMech per graph size (bottom)
  eps_cs_cm.pdf         eps_ATE, same layout
Numbers: RealCause from tab:rc-point and the case studies from tab:cs-point (the paper source, as the
earlier figure script read them); ComplexMech from cm_point_allrho.csv (all rho, context 1000) with the
paper's Do-PFN 2D (262k, temperature off).

    python revision/paper/point_plots.py TEX CM_CSV OUT_DIR
"""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as PS                                         # noqa: E402

TEX, CM_CSV, OUT = sys.argv[1:4]
PS.apply()
MODELS = [("Do-PFN", "dopfn_native"), ("Do-PFN 2D", "dopfn_repro_joint2d_262k_t1"),
          ("UWYK No-Anc", "uwyk1d-noanc"), ("UWYK No-Anc 2D", "graph2d-noanc"),
          ("UWYK Anc", "uwyk1d-v3a"), ("UWYK Anc 2D", "graph2d-v3a"),
          ("CausalPFN-C", "cpfn1d_j1024"), ("CausalPFN-C 2D", "cpfn2d_eta0")]
COLORS = [PS.COL_1D, PS.COL_2D] * 4
RC = ["IHDP", "ACIC", "CPS", "PSID", "PSID-B"]
CASES = ["Obs.\nConf.", "Obs.\nMed.", "Obs.\nM+C", "Unobs.\nConf.", "Front-\ndoor", "Back-\ndoor"]
NODES = [5, 10, 20, 30, 40, 50]
CELL = re.compile(r"([\d.]+)\}?\\,\{\\scriptsize\$\\pm\$([\d.]+)")


def from_tex(label, ncol, commented=False):
    """{model: [(value, se) per cell]} from the rows of a point table (first copy of the label)"""
    lines = open(TEX).read().split("\n")
    strip = (lambda l: re.sub(r"^%\s?", "", l)) if commented else (lambda l: l)
    a = next(i for i, l in enumerate(lines) if strip(l).startswith(f"\\label{{{label}}}")
             and (l.startswith("%") or not commented))    # the commented full-precision copy
    b = next(i for i in range(a, len(lines)) if "\\end{tabular}" in strip(lines[i]))
    out = {}
    for l in map(strip, lines[a:b]):
        c = [x.strip() for x in l.rstrip("\\ ").split("&")]
        if c[0] in dict(MODELS):
            cells = [CELL.search(x) for x in c[1:1 + ncol]]
            out[c[0]] = [(float(m.group(1)), float(m.group(2))) for m in cells]
    assert all(m in out for m, _ in MODELS), (label, [m for m, _ in MODELS if m not in out])
    return out


def grid(rows, cols, cells, row_names, fname, xlabel=None):
    """cells[r][c] = [(value, se) per model]"""
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(PS.TEXTWIDTH, 1.6 * len(rows) + 0.45), sharey=True,
                             gridspec_kw=dict(wspace=0.32, hspace=0.3))
    axes = np.atleast_2d(axes)
    for r in range(len(rows)):
        for c in range(len(cols)):
            v = cells[r][c]
            y = PS.hbars(axes[r, c], [x for x, _ in v], [e for _, e in v], COLORS)
            if r == 0:
                axes[r, c].set_title(cols[c], pad=3, linespacing=1.0)
        axes[r, 0].set_yticks(y)
        axes[r, 0].set_yticklabels([m for m, _ in MODELS])
    fig.subplots_adjust(left=0.245, right=0.985, top=0.9, bottom=0.11 if xlabel else 0.07)
    for r in range(len(rows)):
        p = axes[r, 0].get_position()
        fig.text(0.012, (p.y0 + p.y1) / 2, row_names[r], rotation=90, ha="center", va="center", fontsize=PS.LABEL + 0.5)
    if xlabel:
        fig.supxlabel(xlabel, fontsize=PS.LABEL, y=0.005)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{fname}.{ext}"), dpi=300)
    plt.close(fig)
    print("wrote", os.path.join(OUT, f"{fname}.pdf"))


os.makedirs(OUT, exist_ok=True)
rc = from_tex("tab:rc-point", 10, commented=True)        # the table is commented out in the paper
cs = from_tex("tab:cs-point", 12, commented=True)
cm = pd.read_csv(CM_CSV)
cm = cm[cm["context"] == 1000].set_index(["model", "nodes"])

FAM = [("Do-PFN", "Do-PFN"), ("UWYK No-Anc", "UWYK\nNo-Anc"), ("UWYK Anc", "UWYK\nAnc"), ("CausalPFN-C", "Causal-\nPFN-C")]


def with_avg(titles, cells):
    """append the average over datasets (eps_ATE is relative, so no rescaling); SE: sqrt(sum se^2) / K"""
    K = len(cells)
    avg = [(np.mean([cells[c][m][0] for c in range(K)]), np.sqrt(np.sum([cells[c][m][1] ** 2 for c in range(K)])) / K)
           for m in range(len(MODELS))]
    return titles + ["Avg."], cells + [avg]


def deltas(cells):
    """Delta (%) per family = 100 (geometric mean over datasets of 2D / 1D - 1); lower is better, - favours 2D"""
    return [100 * (-1 + np.exp(np.mean([np.log(cells[c][2 * f + 1][0] / cells[c][2 * f][0]) for c in range(len(cells))])))
            for f in range(4)]


def fam_blocks(blocks, fname, row_h=0.6, avg=False, xlabel=None, delta=False, width=1.0, heading=None, sig=None):
    """blocks = [(block title or None, column titles, cells[c] = [(v, se) per model in MODELS order])]:
    one row per family, two bars per panel (1D blue, 2D orange), each panel its own axis"""
    nrow = 4 * len(blocks)
    K = len(blocks[0][1]) + (1 if delta else 0)
    extra = sum(0.55 if b[0] else 0.0 for b in blocks)
    two_line = any("\n" in t for b in blocks for t in b[1])
    fig = plt.figure(figsize=(PS.TEXTWIDTH * width, row_h * nrow + 0.75 + extra + (0.15 if two_line else 0)))
    gs = fig.add_gridspec(nrow, K, wspace=0.24, hspace=0.5,
                          width_ratios=[1] * (K - 1) + [1.3] if delta else None)
    gs_rows = []
    for b, (btitle, titles, cells) in enumerate(blocks):
        for f, (_, flab) in enumerate(FAM):
            r = 4 * b + f
            if delta:                                            # Average: geometric mean over datasets (+- SE)
                Vf = np.array([[cells[c][2 * f + j][0] for c in range(len(cells))] for j in (0, 1)])
                Ef = np.array([[cells[c][2 * f + j][1] for c in range(len(cells))] for j in (0, 1)])
                g, se = PS.gmean_se(Vf, Ef, sig)
                ax = fig.add_subplot(gs[r, K - 1])
                PS.avg_panel(ax, g, se, [PS.COL_1D, PS.COL_2D], title=(f == 0), kind="G")
                ax.set_ylim(-0.6, 1.6)
            for c in range(K - (1 if delta else 0)):
                ax = fig.add_subplot(gs[r, c])
                v = cells[c][2 * f: 2 * f + 2]
                y = PS.hbars(ax, [x for x, _ in v], [e for _, e in v], [PS.COL_1D, PS.COL_2D], height=0.78,
                             steps=(1, 1.5, 2, 2.5, 3, 4, 5), compact=True)
                ax.set_yticks(y)
                ax.set_yticklabels(["1D", "2D"] if c == 0 else [])
                ax.tick_params(axis="x", labelsize=PS.TICK - 0.5)
                if f == 0:
                    ax.set_title(titles[c], pad=3, linespacing=1.0, fontweight="bold" if avg and c == K - 1 - (1 if delta else 0) else "normal")
                if avg and c == K - 1 - (1 if delta else 0):
                    ax.set_facecolor(PS.COL_GREY)
                if c == 0:
                    gs_rows.append((ax, flab, btitle if f == 0 else None))
    fig.subplots_adjust(left=0.12 / width, right=0.985, top=1 - 0.55 / fig.get_figheight(), bottom=0.45 / fig.get_figheight())
    for ax, flab, btitle in gs_rows:
        p = ax.get_position()
        fig.text(0.045 / width, (p.y0 + p.y1) / 2, flab, rotation=90, ha="center", va="center", fontweight="bold",
                 fontsize=PS.LABEL, linespacing=1.0)
    for b, (btitle, _, _) in enumerate(blocks):
        if btitle:
            top = [ax for ax, _, t in gs_rows if t == btitle][0].get_position()
    if heading:                                                  # benchmark name above the column titles
        top = max(ax.get_position().y1 for ax in fig.axes)
        fig.text((0.12 + 0.985) / 2, top + (0.42 if two_line else 0.27) / fig.get_figheight(), heading, ha="center",
                 va="bottom", fontsize=PS.TITLE + 1, fontweight="bold")
    if xlabel:                                                   # just under the bottom row's tick labels
        y0 = min(ax.get_position().y0 for ax in fig.axes)
        fig.text((0.12 + 0.985) / 2, y0 - 0.2 / fig.get_figheight(), xlabel, ha="center", va="top", fontsize=PS.LABEL)
    for ext in ("pdf", "png"):                                   # cropped to the content: no blank margins
        fig.savefig(os.path.join(OUT, f"{fname}.{ext}"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", os.path.join(OUT, f"{fname}.pdf"))


import update_tables as U                                        # noqa: E402
_, SIG = U.load()
SP = U.specs(SIG)
SIGMA = {k: [c[3] for c in SP[k]] for k in ("rc", "cs", "cmall")}
PE, EA = r"$\sqrt{\mathrm{PEHE}}$", r"$\epsilon_{\mathrm{ATE}}$"
for k, tag in ((0, "pehe"), (1, "eps")):
    rc_cells = [[rc[m][2 * j + k] for m, _ in MODELS] for j in range(5)]
    if tag == "eps":                                             # average only for eps_ATE (not for PEHE)
        fam_blocks([(None, RC, rc_cells)], f"point_realcause_{tag}", xlabel=EA, delta=True)
    else:
        fam_blocks([(None, RC, rc_cells)], f"point_realcause_{tag}", xlabel=PE, delta=True, sig=SIGMA["rc"])
for k, (metric, col, sem) in enumerate((("pehe", "pehe", "pehe_sem"), ("eps", "eps_ate", "eps_sem"))):
    cs_cells = [[cs[m][2 * j + k] for m, _ in MODELS] for j in range(6)]
    cm_cells = [[(float(cm.at[(key, n), col]), float(cm.at[(key, n), sem])) for _, key in MODELS] for n in NODES]
    ea = metric == "eps"
    cs_t, cs_c = CASES, cs_cells                                 # Delta (%) instead of an average, as for PEHE
    cm_t, cm_c = [f"$n={n}$" for n in NODES], cm_cells
    fam_blocks([(None, cs_t, cs_c)], f"{metric}_cs", xlabel=EA if ea else PE, delta=True, sig=None if ea else SIGMA["cs"])
    fam_blocks([(None, cm_t, cm_c)], f"{metric}_cm", xlabel=EA if ea else PE, delta=True, sig=None if ea else SIGMA["cmall"])


# ---- Twins: counterfactual MSE, d = 1 and d = 10 (no average, no Delta)
tw = pd.read_csv(os.path.join(os.path.dirname(CM_CSV), "bodik_twins_bodiksetup.csv")).set_index(["setting", "model"])
TWK = {"Do-PFN": "Do-PFN", "Do-PFN 2D": "Do-PFN 2D", "UWYK No-Anc": "UWYK No-Anc", "UWYK No-Anc 2D": "UWYK 2D No-Anc",
       "UWYK Anc": "UWYK Anc", "UWYK Anc 2D": "UWYK 2D Anc", "CausalPFN-C": "CausalPFN-C", "CausalPFN-C 2D": "CausalPFN-C 2D"}
tw_cells = [[(tw.at[(d, TWK[m]), "mse"], tw.at[(d, TWK[m]), "mse_se"]) for m, _ in MODELS] for d in ("d=1", "d=10")]
fam_blocks([(None, ["$d = 1$", "$d = 10$"], tw_cells)], "twins_mse", xlabel="Mean squared error", width=0.7)
