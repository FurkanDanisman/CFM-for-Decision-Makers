"""Figures of the scaled OT ATE density, in the layout of the paper's ITE figures:
  ate_ish_{realcause,case_studies,complexmech}   as is_hbars.py: one row per model family, the single-outcome model
                                                 under rho = -0.5, 0, 0.5, 1 (blue) and the 2D model (orange)
  ate_coupling_abl_{rc,cs,cm}                    as appendix_plots.py coupling_abl: matched-resolution (green) and
                                                 matched-label (purple) baselines under each rho, and the 2D model
Value: interval score IS_0.05 / sd(Y) of the 95% interval of the SCALED barycenter (s from in-prior split conformal),
from revision/results/ate_ot/{rc,cs,cm}.csv (bench_rho.py); cells and sd(Y) are the paper's (update_tables.specs).
Bars: mean over realizations +- SE; last column Average_g (geometric mean over cells, delta-method SE), stars:
z-test against the 2D model, Holm within each row (paper_style.holm_stars), as the ITE figures.
Also prints the coverage of the scaled interval per variant and cell, and writes it to ate_coverage.txt.

    python revision/paper/ate_plots.py OUT_DIR
"""
import ast
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import update_tables as U                                        # noqa: E402
import paper_style as PS                                         # noqa: E402

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
RES = os.environ.get("ATE_OT_RES", os.path.join(HERE, "..", "results", "ate_ot"))
d, sig = U.load()
sp = U.specs(sig)
RHOS = ["m0.5", "0", "0.5", "1"]
RHOLAB = [r"$\rho=-0.5$", r"$\rho=0$", r"$\rho=0.5$", r"$\rho=1$", "2D"]
FAM = [("Do-PFN", "dopfn_native", "dopfn_joint2d_262k", "dopfn1d"),
       ("UWYK No-Anc", "uwyk1d-noanc", "graph2d-noanc", "uwyk1d|noanc"),
       ("UWYK Anc", "uwyk1d-v3a", "graph2d-v3a", "uwyk1d|v3a"),
       ("CausalPFN-C", "cpfn1d", "cpfn2d", "cpfn1d")]
FAMROW = {"Do-PFN": "Do-PFN", "UWYK No-Anc": "UWYK\nNo-Anc", "UWYK Anc": "UWYK\nAnc", "CausalPFN-C": "Causal-\nPFN-C"}
BENCH = [("rc", "rc", "realcause", ["IHDP", "ACIC", "CPS", "PSID", "PSID-B"]),
         ("cs", "cs", "case_studies", ["Obs.\nConf.", "Obs.\nMed.", "Obs.\nM+C", "Unobs.\nConf.", "Front-\ndoor", "Back-\ndoor"]),
         ("cmall", "cm", "complexmech", [f"$n={n}$" for n in U.NS])]


def abl_name(fam_key, which):
    """ablation variant prefix: dopfn1d_mr, uwyk1d_mr-noanc, cpfn1d_ml, ..."""
    base, _, tag = fam_key.partition("|")
    return f"{base}_{which}" + (f"-{tag}" if tag else "")


def series(df, key, variant):
    """(values, SEs) over the paper's cells, IS of the scaled interval / sd(Y); nan when a cell is missing"""
    v, e = [], []
    for _, case, sl, sd in sp[key]:
        cell = sl if key == "cmall" else case
        r = df[(df.variant == variant) & (df.cell == cell)]
        v.append(r.iss.iloc[0] / sd if len(r) else np.nan); e.append(r.iss_se.iloc[0] / sd if len(r) else np.nan)
    return np.array(v), np.array(e)


def rows_of(df, key, one, two):
    V, E = zip(*([series(df, key, f"{one}-rho{r}") for r in RHOS] + [series(df, key, two)]))
    return np.array(V), np.array(E)


# ---- the ablation figure function of appendix_plots.py, taken verbatim (that module runs its figures on import)
_src = open(os.path.join(HERE, "appendix_plots.py")).read()
_fn = next(n for n in ast.parse(_src).body if isinstance(n, ast.FunctionDef) and n.name == "figure")
STEPS = (1, 1.5, 2, 2.5, 3, 4, 5)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name)


exec(compile(ast.Module([_fn], []), "appendix_plots.figure", "exec"))


def THREE(top):
    s = max(m * 10 ** e for e in range(-3, 4) for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8) if 2 * m * 10 ** e <= top)
    return [0, s, 2 * s]


plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 10.5, "axes.labelsize": 10.5, "xtick.labelsize": 9.5,
                     "ytick.labelsize": 10, "font.family": "STIXGeneral", "mathtext.fontset": "stix", "pdf.fonttype": 42,
                     "axes.linewidth": 0.5})
cov_lines = []
for key, b, fname, names in BENCH:
    path = os.path.join(RES, f"{b}.csv")
    if not os.path.exists(path):
        print("missing", path); continue
    df = pd.read_csv(path)
    for v in sorted(df.variant.unique()):                         # coverage of the scaled interval, per cell
        r = df[df.variant == v]
        cov_lines.append(f"{b:2s} {v:28s} s={r.s.iloc[0]:<6} " + "  ".join(f"{c}:{x:.3f}" for c, x in zip(r.cell, r.covs)))
    # ---- main figure (is_hbars.py layout)
    K = len(sp[key])
    V = np.zeros((4, 5, K + 1)); E = np.full((4, 5, K + 1), np.nan)
    for f, (lab, one, two, _) in enumerate(FAM):
        V[f, :, :K], E[f, :, :K] = rows_of(df, key, one, two)
        for j in range(5):
            ok = np.isfinite(V[f, j, :K])
            g, se = PS.gmean_se(V[f, j, :K][ok][None], E[f, j, :K][ok][None])
            V[f, j, K], E[f, j, K] = g[0], se[0]
    SIG = np.array([PS.holm_stars(V[f, :4, K], E[f, :4, K], np.repeat(V[f, 4, K], 4), np.repeat(E[f, 4, K], 4))
                    for f in range(4)])
    print(b, "baseline / 2D Average_g per family x rho:", np.round(V[:, :4, K] / V[:, 4:5, K], 2).tolist(),
          "stars:", SIG.tolist())
    fig, axes = plt.subplots(4, K + 1, figsize=(PS.TEXTWIDTH, 0.82 * 4 + (0.75 if key == "cs" else 0.6)), sharey=True,
                             gridspec_kw=dict(wspace=0.38, hspace=0.42, width_ratios=[1] * K + [1.6]))
    y = np.arange(5)[::-1]
    for c in range(K + 1):
        for f in range(4):
            ax = axes[f, c]
            if c == K:
                PS.avg_panel(ax, V[f, :, K], E[f, :, K], [PS.COL_1D] * 4 + [PS.COL_2D], title=(f == 0), kind="G",
                             stars=list(SIG[f]) + [False], bold_min=True)
                continue
            for s_ in ("top", "right"):
                ax.spines[s_].set_linestyle((0, (1, 1.5)))
            ax.tick_params(axis="x", length=2, width=0.5, pad=1.5); ax.tick_params(axis="y", length=0, pad=1.5)
            ax.barh(y, np.nan_to_num(V[f, :, c]), height=0.8, xerr=np.nan_to_num(E[f, :, c]),
                    color=[PS.COL_1D] * 4 + [PS.COL_2D], edgecolor="black", linewidth=0.7,
                    error_kw=dict(elinewidth=0.9, ecolor="0.2", capsize=1.6, capthick=0.9))
            top = np.nanmax(V[f, :, c] + np.nan_to_num(E[f, :, c])) * 1.08 if np.isfinite(V[f, :, c]).any() else 1
            ax.set_xlim(0, top); ax.set_xticks(THREE(top))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            if f == 0:
                ax.set_title(names[c], pad=3, linespacing=1.0)
    for f, (lab, *_r) in enumerate(FAM):
        axes[f, 0].set_yticks(y); axes[f, 0].set_yticklabels(RHOLAB)
        axes[f, 0].set_ylabel(FAMROW[lab], fontsize=10.5, fontweight="bold", labelpad=3, linespacing=1.0)
    fig.supxlabel("Interval Score", fontsize=10.5, y=0.005)
    fig.subplots_adjust(left=0.165, right=0.985, top=0.91 if key == "cs" else 0.94, bottom=0.085)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"ate_ish_{fname}.{ext}"), dpi=300)
    plt.close(fig)
    print("wrote", f"ate_ish_{fname}")
    # ---- ablation figure (appendix_plots.py coupling_abl layout)
    rows = []
    for which, short in (("mr", "resolution"), ("ml", "label")):
        for lab, one, two, fk in FAM:
            Vr, Er = rows_of(df, key, abl_name(fk, which), two)
            rows.append((f"{FAMROW[lab]}\n({short})", Vr, Er))
    if all(np.isfinite(V_).all() for _, V_, _ in rows):
        figure(rows, names, RHOLAB, [PS.COL_1D] * 4 + [PS.COL_2D], f"ate_coupling_abl_{b}", "Interval Score",
               row_h=0.82, sep=4, row_colors=[[PS.COL_RES] * 4 + [PS.COL_2D]] * 4 + [[PS.COL_LAB] * 4 + [PS.COL_2D]] * 4)
    else:
        print(b, "ablation figure skipped: missing cells",
              [r[0].replace("\n", " ") for r in rows if not np.isfinite(r[1]).all()])
open(os.path.join(OUT, "ate_coverage.txt"), "w").write("\n".join(cov_lines) + "\n")
print("wrote ate_coverage.txt")
