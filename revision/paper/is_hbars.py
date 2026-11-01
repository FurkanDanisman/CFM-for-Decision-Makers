"""Interval score of the main calibration results in the layout of Do-PFN's Figure 3: one figure per benchmark,
one row per model family, one panel per dataset (+ the average over datasets), horizontal bars for the
single-outcome model at rho = -0.5, 0, 0.5, 1 and the 2D model (+- SE), model labels once on the left.
IS_0.05 / sd(Y), the cells of tab:rc-cal / tab:cs-cal / tab:cm-r99-cal; the average is the tables' combined score.
Each panel has its own linear x-axis from 0 with its own ticks (Do-PFN Fig. 3, row 1). Drawn at the CLeaR text width, so the font sizes are the printed sizes.

    python revision/paper/is_hbars.py OUT_DIR
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import update_tables as U                                        # noqa: E402
import coupling_tables as CT                                     # noqa: E402
import rho_rows as RR                                            # noqa: E402
import paper_style as PS                                         # noqa: E402

OUT = sys.argv[1]
TEXTWIDTH = 433.62 / 72.27
BENCH = [("rc", "realcause", ["IHDP", "ACIC", "CPS", "PSID", "PSID-B"]),
         ("cs", "case_studies", ["Obs.\nConf.", "Obs.\nMed.", "Obs.\nM+C", "Unobs.\nConf.", "Front-\ndoor", "Back-\ndoor"]),
         ("cmall", "complexmech", [f"$n={n}$" for n in U.NS])]
COL_1D, COL_2D = "#3f7fbf", "#e6820e"
YLAB = [r"$\rho=-0.5$", r"$\rho=0$", r"$\rho=0.5$", r"$\rho=1$", "2D"]
plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 10.5, "axes.labelsize": 10.5, "xtick.labelsize": 9.5,
                     "ytick.labelsize": 10, "font.family": "STIXGeneral", "mathtext.fontset": "stix", "pdf.fonttype": 42,
                     "axes.linewidth": 0.5})

FAM = {"Do-PFN": "Do-PFN", "UWYK No-Anc": "UWYK\nNo-Anc", "UWYK Anc": "UWYK\nAnc", "CausalPFN-C": "Causal-\nPFN-C"}


def THREE(top):
    """exactly three ticks 0, s, 2s, with s the largest round step that keeps 2s inside [0, top]"""
    s = max(m * 10 ** e for e in range(-3, 4) for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8) if 2 * m * 10 ** e <= top)
    return [0, s, 2 * s]


def TICKS(a, b, gap=0.55):
    """1-2-5 ticks inside [a, b], at least `gap` decades apart, so the labels never touch"""
    out = []
    for t in [m * 10 ** e for e in range(-2, 4) for m in (1, 2, 5)]:
        if a <= t <= b and (not out or np.log10(t / out[-1]) >= gap):
            out.append(t)
    return out


d, sig = U.load()
sp = U.specs(sig)
cp = CT.coupling_rows()
os.makedirs(OUT, exist_ok=True)
for key, fname, names in BENCH:
    cols = sp[key]
    get = lambda src, m: [d.get((src, bm, case, sl, m)) for bm, case, sl, _ in cols]
    getc = lambda st, m: [cp.get((st, bm, case, sl, m)) for bm, case, sl, _ in cols]
    K = len(cols)
    V = np.zeros((4, 5, K + 1)); E = np.full((4, 5, K + 1), np.nan)   # family, model, dataset (+ average)
    for f, (l1, k1, c1, l2, k2) in enumerate(CT.MAIN):
        series = [get("hist", k1) if st is None else getc(st, c1) for st, _ in RR.RHO] + [get("hist", k2)]
        for j, rs in enumerate(series):
            assert all(r is not None for r in rs), (key, l1, j)
            V[f, j, :K] = [r.is05 / c[3] for r, c in zip(rs, cols)]
            E[f, j, :K] = [r.is05_se / c[3] for r, c in zip(rs, cols)]
            g, se = PS.gmean_se(V[f, j, :K][None], E[f, j, :K][None])   # geometric mean over datasets, delta-method SE
            V[f, j, K], E[f, j, K] = g[0], se[0]
    # Delta (%) per assumed rho: 100 (Avg IS of the 2D model / Avg IS of the 1D model at that rho - 1), the averages
    # over datasets being the tables' combined score (as the Findings quote them); + favours 2D. One fixed axis
    # [-100, 100] for every Delta panel, so the zero line is in the same place everywhere.
    RAT = V[:, :4, K] / V[:, 4:5, K]                              # family x rho: mean IS of the baseline / of 2D
    PCT = 100 * (1 - 1 / RAT)                                     # + = the 2D model's mean IS is that much lower
    LIMS = PS.pct_limits(PCT)
    fig, axes = plt.subplots(4, K + 1, figsize=(TEXTWIDTH, 0.82 * 4 + (0.75 if key == "cs" else 0.6)), sharey=True,
                             gridspec_kw=dict(wspace=0.38, hspace=0.42, width_ratios=[1] * K + [1.6]))
    y = np.arange(5)[::-1]
    for c in range(K + 1):
        for f in range(4):
            ax = axes[f, c]
            if c == K:                                           # average over datasets, value printed
                PS.avg_panel(ax, V[f, :, K], E[f, :, K], [COL_1D] * 4 + [COL_2D], title=(f == 0), kind="G")
                continue
            for sp_ in ("top", "right"):                         # dotted top/right frame, as Do-PFN
                ax.spines[sp_].set_linestyle((0, (1, 1.5)))
            ax.tick_params(axis="x", length=2, width=0.5, pad=1.5)
            ax.tick_params(axis="y", length=0, pad=1.5)
            ax.barh(y, V[f, :, c], height=0.8, xerr=E[f, :, c], color=[COL_1D] * 4 + [COL_2D],
                    edgecolor="black", linewidth=0.7,
                    error_kw=dict(elinewidth=0.9, ecolor="0.2", capsize=1.6, capthick=0.9))
            top = np.nanmax(V[f, :, c] + np.nan_to_num(E[f, :, c])) * 1.08
            ax.set_xlim(0, top)                                  # each panel its own linear axis from 0
            ax.set_xticks(THREE(top))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            if f == 0:
                ax.set_title(names[c] if c < K else "Avg.", pad=3, linespacing=1.0,
                             fontweight="bold" if c == K else "normal")
            if c == K:
                ax.set_facecolor("0.93")                         # the average over datasets (with its SE)
    print(key, "baseline / 2D per family x rho (-0.5, 0, 0.5, 1):", np.round(RAT, 2).tolist())
    for f, (l1, *_rest) in enumerate(CT.MAIN):
        axes[f, 0].set_yticks(y)
        axes[f, 0].set_yticklabels(YLAB)
        axes[f, 0].set_ylabel(FAM[l1], fontsize=10.5, fontweight="bold", labelpad=3, linespacing=1.0)
    fig.supxlabel("Interval Score", fontsize=10.5, y=0.005)
    fig.subplots_adjust(left=0.165, right=0.985, top=0.91 if key == "cs" else 0.94, bottom=0.085)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"ish_{fname}.{ext}"), dpi=300)
    plt.close(fig)
    print("wrote", os.path.join(OUT, f"ish_{fname}.pdf"))
