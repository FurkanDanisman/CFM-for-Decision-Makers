"""Appendix tables as figures, in the style of the main interval-score and point figures (paper_style.py):
one row per model (family), one panel per dataset, horizontal bars (+- SE), each panel its own axis, and a last
Delta (%) column = change of the 2D model relative to each baseline (lower is better, negative favours 2D):
  interval score: 100 (mean over datasets of 2D / that of the baseline - 1)   (the tables' combined score)
  point:          100 (geometric mean over datasets of 2D / baseline - 1)       (the tables' Delta, sign flipped)

  ablation_point_{rc,cs,cm}_{pehe,eps}   matched-label / matched-resolution ablations and the 2D model (points)
  coupling_abl_{rc,cs,cm}                the ablation baselines under each assumed rho and the 2D model (IS)
  rho_ctrl_{bench}                       controlled cross-world correlation, per rho bin (IS)
  csd_{case}                             case studies by covariate dimension d (IS)
  cmp_cal_{rc,cs,cm}                     published two-stage CausalPFN / UWYK-B against the 2D models (IS)
  cmp_point_{rc,cs,cm}_{pehe,eps}        the same comparison, point estimation

    python revision/paper/appendix_plots.py OUT_DIR
"""
import csv
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paper_style as PS                                         # noqa: E402
import update_tables as U                                        # noqa: E402
import coupling_tables as CT                                     # noqa: E402
import rho_rows as RR                                            # noqa: E402

OUT = sys.argv[1]
RES = os.path.join(HERE, "..", "results", "d262t1")
TEX = os.path.join(HERE, "..", "working_revision", "overleaf", "Template_CLEAR_anon", "main.tex")
PS.apply()
os.makedirs(OUT, exist_ok=True)
RHOLAB = [r"$\rho=-0.5$", r"$\rho=0$", r"$\rho=0.5$", r"$\rho=1$", "2D"]
RC = ["IHDP", "ACIC", "CPS", "PSID", "PSID-B"]
CS_KEYS = ["Observed_Confounder", "Observed_Mediator", "Observed_Mediator_and_Confounder", "Unobserved_Confounder",
           "Frontdoor_Criterion", "Backdoor_Criterion"]
CASES = ["Obs.\nConf.", "Obs.\nMed.", "Obs.\nM+C", "Unobs.\nConf.", "Front-\ndoor", "Back-\ndoor"]
CS_NAME = {"Observed_Confounder": "Observed Confounder", "Observed_Mediator": "Observed Mediator",
           "Observed_Mediator_and_Confounder": "Observed Mediator + Confounder",
           "Unobserved_Confounder": "Unobserved Confounder", "Frontdoor_Criterion": "Frontdoor",
           "Backdoor_Criterion": "Backdoor"}
NODES = [5, 10, 20, 30, 40, 50]
COLS = {"rc": RC, "cs": CASES, "cmall": [f"$n={n}$" for n in NODES]}
STEPS = (1, 1.5, 2, 2.5, 3, 4, 5)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name)


def figure(rows, cols, ylabels, colors, fname, xlabel, kind="is", hatches=None, width=1.0, row_h=None, sig=None):
    """sig: per row, the outcome scale of each dataset (point estimates of PEHE: geometric mean of PEHE / sigma)"""
    """rows = [(row label, V[nser, K], E[nser, K])]; the last series is the 2D model (reference of Delta)."""
    per_row = isinstance(cols[0], list)                         # column titles per row (rows of different benchmarks)
    rcols = cols if per_row else [cols] * len(rows)
    nser, K, R = len(ylabels), max(len(c) for c in rcols), len(rows)
    row_h = row_h or (0.18 * nser + 0.36)
    two_line = any("\n" in c for cc in rcols for c in cc)
    extra = (0.42 if two_line else 0.25) * (R - 1) if per_row else 0   # room for each row's own titles
    fig = plt.figure(figsize=(PS.TEXTWIDTH * width, row_h * R + extra + 0.55 + (0.15 if two_line else 0)))
    gs = fig.add_gridspec(R, K + 1, wspace=0.3, hspace=(0.5 + extra / max(R - 1, 1) / row_h) if per_row else 0.5,
                          width_ratios=[1] * K + [1.6])
    D = []
    for _, V, _ in rows:
        if kind == "is":
            D.append([100 * (V[-1].mean() / V[s].mean() - 1) for s in range(nser - 1)])
        else:
            D.append([100 * (np.exp(np.mean(np.log(V[-1] / V[s]))) - 1) for s in range(nser - 1)])
    D = np.array(D)
    LIMS = PS.pct_limits(-D)
    labels = []
    for r, (rlab, V, E) in enumerate(rows):
        for c in range(len(rcols[r])):
            ax = fig.add_subplot(gs[r, c])
            y = PS.hbars(ax, V[:, c], E[:, c], colors, height=0.8, steps=STEPS, compact=True)
            if hatches:
                for p, h in zip(ax.patches, hatches):
                    if h:
                        p.set_hatch(h)
            ax.set_yticks(y)
            ax.set_yticklabels(ylabels if c == 0 else [])
            ax.tick_params(axis="x", labelsize=PS.TICK - 0.5)
            if r == 0 or per_row:
                ax.set_title(rcols[r][c], pad=3, linespacing=1.0)
            if c == 0:
                labels.append((ax, rlab))
        ax = fig.add_subplot(gs[r, K])
        if kind == "is":                                         # the average over datasets +- its SE
            PS.avg_panel(ax, V.mean(1), np.sqrt((E ** 2).sum(1)) / V.shape[1], colors, title=(r == 0 or per_row))
        else:                                                    # point estimates: geometric mean over datasets
            g, se = PS.gmean_se(V, E, None if sig is None else sig[r])
            PS.avg_panel(ax, g, se, colors, title=(r == 0 or per_row))
            if hatches:
                for p_, h in zip(ax.patches, hatches):
                    if h:
                        p_.set_hatch(h)
    nl = max(r[0].count("\n") + 1 for r in rows)              # lines of the rotated row labels
    left = (0.2 if max(len(l) for l in ylabels) > 8 else 0.15) + 0.022 * (nl - 2)
    fig.subplots_adjust(left=left / width, right=0.985, top=1 - 0.45 / fig.get_figheight(),
                        bottom=0.42 / fig.get_figheight())
    for ax, rlab in labels:
        p = ax.get_position()
        fig.text((0.02 + 0.011 * (nl - 2)) / width, (p.y0 + p.y1) / 2, rlab, rotation=90, ha="center", va="center",
                 fontweight="bold", fontsize=PS.LABEL - 1, linespacing=1.0)
    y0 = min(a.get_position().y0 for a in fig.axes)
    fig.text((left / width + 0.985) / 2, y0 - 0.2 / fig.get_figheight(), xlabel, ha="center", va="top",
             fontsize=PS.LABEL)
    save(fig, fname)
    return D


d, sig = U.load()
sp = U.specs(sig)
cp = CT.coupling_rows()
FAMROW = {"Do-PFN": "Do-PFN", "UWYK No-Anc": "UWYK\nNo-Anc", "UWYK Anc": "UWYK\nAnc", "CausalPFN-C": "Causal-\nPFN-C"}
IS = "Interval Score"


def is_rows(key, series_keys):
    """series_keys = [(source, model key)], source None = hist (rho = 0 / 2D) else a coupling stage"""
    cols = sp[key]
    V = np.array([[(d[("hist", bm, c, s, m)] if st is None else cp[(st, bm, c, s, m)]).is05 / sd
                   for bm, c, s, sd in cols] for st, m in series_keys])
    E = np.array([[(d[("hist", bm, c, s, m)] if st is None else cp[(st, bm, c, s, m)]).is05_se / sd
                   for bm, c, s, sd in cols] for st, m in series_keys])
    return V, E


# ---- coupling sensitivity of the ablation baselines (tab:coupling-abl-*)
ABLK = {a[0]: a for a in CT.ABL}
TWO = {f: k2 for _, _, _, f2, k2 in [(None,) + m[1:] for m in CT.MAIN] for f in [f2]}
FAM2D = {m[0]: m[4] for m in CT.MAIN}
for key, tag in (("rc", "rc"), ("cs", "cs"), ("cmall", "cm")):
    rows = []
    for fam in ("Do-PFN", "UWYK No-Anc", "UWYK Anc", "CausalPFN-C"):
        for var, short in (("matched res.", "res."), ("matched labels", "labels")):
            lab, k1, c1 = ABLK[f"{fam} ({var})"]
            ser = [(None, k1) if st is None else (st, c1) for st, _ in RR.RHO] + [(None, FAM2D[fam])]
            V, E = is_rows(key, ser)
            rows.append((f"{FAMROW[fam]}\n({short})", V, E))
    figure(rows, COLS[key], RHOLAB, [PS.COL_1D] * 4 + [PS.COL_2D], f"coupling_abl_{tag}", IS, row_h=0.82)

# ---- the comparison baselines: published two-stage CausalPFN and UWYK-B (tab:cpfn-2d-rho, tab:uwyk-binarization-rho)
CMPS = {"cpfn": [("", "CausalPFN (two-stage, published)", "CausalPFN 2D")],
        "uwykb": [("No-Anc", "UWYK-B No-Anc", "UWYK 2D (noanc)"), ("Anc", "UWYK-B Anc", "UWYK 2D (v3a)")]}
BNAME = {"rc": "RealCause", "cs": "Case\nstudies", "cmall": "Complex-\nMech"}
for cmp_, pairs in CMPS.items():
    rows, rcols = [], []
    for key in ("rc", "cs", "cmall"):
        for sub, b, m2 in pairs:
            V, E = is_rows(key, [(None, b) if st is None else (st, b) for st, _ in RR.RHO] + [(None, m2)])
            rows.append((BNAME[key] + (f"\n{sub}" if sub else ""), V, E)); rcols.append(COLS[key])
    figure(rows, rcols, RHOLAB, [PS.COL_1D] * 4 + [PS.COL_2D], f"cmp_{cmp_}_cal", IS, row_h=0.82)

# ---- case studies by dimension (tab:cs-d-all)
import rho_extra_tables as RX                                    # noqa: E402
for case in CS_KEYS:
    sd = sig["main"]["case_study"][case]
    rows = []
    for (l1, m1), (l2, m2) in RX.PAIRS_D:
        ser = [(None, m1) if st is None else (st, m1) for st, _ in RR.RHO] + [(None, m2)]
        V = np.array([[(d[("hist", "CaseStudy", case, f"d{x}", m)] if st is None else
                        cp[(st, "CaseStudy", case, f"d{x}", m)]).is05 / sd for x in RX.DS] for st, m in ser])
        E = np.array([[(d[("hist", "CaseStudy", case, f"d{x}", m)] if st is None else
                        cp[(st, "CaseStudy", case, f"d{x}", m)]).is05_se / sd for x in RX.DS] for st, m in ser])
        rows.append((FAMROW.get(l1, l1), V, E))
    figure(rows, [f"$d={x}$" for x in RX.DS], RHOLAB, [PS.COL_1D] * 4 + [PS.COL_2D],
           "csd_" + {"Observed_Confounder": "oc", "Observed_Mediator": "om", "Observed_Mediator_and_Confounder": "omc",
                     "Unobserved_Confounder": "uc", "Frontdoor_Criterion": "fd", "Backdoor_Criterion": "bd"}[case],
           IS, row_h=0.82)

# ---- controlled cross-world correlation, per rho bin (tab:rho-ctrl-cal and its per-rho copies)
RHO_FILES = [("gauss_-0.5", "rho_ctrl_gauss_-0.5.csv"), (None, "rho_ctrl.csv"), ("gauss_0.5", "rho_ctrl_gauss_0.5.csv"),
             ("comonotonic", "rho_ctrl_comonotonic.csv")]
RC_ = {st: pd.read_csv(os.path.join(RES, f)).set_index(["benchmark", "bin", "model"]) for st, f in RHO_FILES}
BINS = ["[-0.5, 0)", "[0, 0.5)", "[0.5, 1]"]
BINLAB = [r"$\rho\in[-0.5,0)$", r"$\rho\in[0,0.5)$", r"$\rho\in[0.5,1]$"]
CTRL = [("Do-PFN", "Do-PFN", "Do-PFN 2D"), ("UWYK\nNo-Anc", "UWYK No-Anc", "UWYK 2D No-Anc"),
        ("UWYK\nAnc", "UWYK Anc", "UWYK 2D Anc"), ("Causal-\nPFN-C", "CausalPFN-C", "CausalPFN-C 2D"),
        ("C-PFN-C\n(m. res.)", "CausalPFN-C (matched resolution)", "CausalPFN-C 2D")]
for bench in RC_[None].index.get_level_values(0).unique():
    rows = []
    for lab, m1, m2 in CTRL:
        if (bench, BINS[0], m1) not in RC_[None].index:
            continue
        V = np.array([[RC_[st].at[(bench, b, m1), "IS"] for b in BINS] for st, _ in RHO_FILES]
                     + [[RC_[None].at[(bench, b, m2), "IS"] for b in BINS]])
        E = np.array([[RC_[st].at[(bench, b, m1), "IS_se"] for b in BINS] for st, _ in RHO_FILES]
                     + [[RC_[None].at[(bench, b, m2), "IS_se"] for b in BINS]])
        rows.append((lab, V, E))
    tag = re.sub(r"[^a-z0-9]+", "_", bench.lower()).strip("_")
    figure(rows, BINLAB, RHOLAB, [PS.COL_1D] * 4 + [PS.COL_2D], f"rho_ctrl_{tag}", IS, row_h=0.82)

# ---- ablation point estimates (ablation_point.csv)
AB = {(r["benchmark"], r["setting"], r["model"], r["variant"]): r for r in csv.DictReader(open(os.path.join(RES, "ablation_point.csv")))}
BKEYS = [("RealCause", ["IHDP", "ACIC", "CPS", "PSID", "PSID_bal"], RC, "rc"),
         ("CaseStudy", CS_KEYS, CASES, "cs"), ("ComplexMech", [str(n) for n in NODES], COLS["cmall"], "cm")]
VARS = ["matched label", "matched resolution", "2D"]
for bm, keys, cols, tag in BKEYS:
    for metric, se, lab, mt in (("pehe", "pehe_se", r"$\sqrt{\mathrm{PEHE}}$", "pehe"), ("eps_ate", "eps_se", r"$\epsilon_{\mathrm{ATE}}$", "eps")):
        rows = []
        for fam in ("Do-PFN", "UWYK No-Anc", "UWYK Anc", "CausalPFN-C"):
            V = np.array([[float(AB[(bm, k, fam, v)][metric]) for k in keys] for v in VARS])
            E = np.array([[float(AB[(bm, k, fam, v)][se]) for k in keys] for v in VARS])
            rows.append((FAMROW[fam], V, E))
        sg = [[c[3] for c in sp[{"rc": "rc", "cs": "cs", "cm": "cmall"}[tag]]]] * len(rows) if mt == "pehe" else None
        figure(rows, cols, ["label", "res.", "2D"], [PS.COL_1D, PS.COL_1D, PS.COL_2D], f"ablation_point_{tag}_{mt}",
               lab, kind="point", hatches=["", "////", ""], row_h=0.66, sig=sg)

# ---- comparison point estimates, from the paper's tables (tab:cpfn-2d, tab:uwyk-binarization)
CELL = re.compile(r"([\d.]+)\}?\\,\{\\scriptsize\$\\pm\$([\d.]+)")


def table_rows(label):
    """{section: [(dataset, [(v, se) per cell])]} from a comparison point table"""
    s = open(TEX).read()
    i = s.index(f"\\label{{{label}}}")
    b = s.index(r"\end{tabular}", i)
    out, sec = {}, None
    for l in s[i:b].splitlines():
        l = re.sub(r"^%\s?", "", l)                             # the table itself is commented out in the paper
        if l.lstrip().startswith("%"):
            continue
        m = re.search(r"\\textit\{([^}]*)\}", l)
        if "multicolumn" in l and m:
            sec = m.group(1); out[sec] = []
        elif sec and "&" in l and not l.startswith(r"$\Delta"):
            c = [x.strip() for x in l.rstrip().rstrip("\\").split("&")]
            cells = [CELL.search(x) for x in c[1:]]
            if all(cells):
                out[sec].append((c[0], [(float(m.group(1)), float(m.group(2))) for m in cells]))
    return out


cp_t, ub_t = table_rows("tab:cpfn-2d"), table_rows("tab:uwyk-binarization")
SEC = [("RealCause", "rc", "RealCause"), ("Case studies", "cs", "Case\nstudies"), ("ComplexMech", "cm", "Complex-\nMech")]
CCOL = {"rc": RC, "cs": CASES, "cm": COLS["cmall"]}
for k, (mt, lab) in enumerate((("pehe", r"$\sqrt{\mathrm{PEHE}}$"), ("eps", r"$\epsilon_{\mathrm{ATE}}$"))):
    rows, rcols = [], []                                         # CausalPFN: base PEHE, base eps, 2D PEHE, 2D eps
    for sec, tag, bl in SEC:
        a = cp_t[sec]
        rows.append((bl, np.array([[r[1][k][0] for r in a], [r[1][2 + k][0] for r in a]]),
                     np.array([[r[1][k][1] for r in a], [r[1][2 + k][1] for r in a]]))); rcols.append(CCOL[tag])
    sg = [[c[3] for c in sp[{"rc": "rc", "cs": "cs", "cm": "cmall"}[t]]] for _, t, _ in SEC] if mt == "pehe" else None
    figure(rows, rcols, ["baseline", "2D"], [PS.COL_1D, PS.COL_2D], f"cmp_cpfn_point_{mt}", lab, kind="point", row_h=0.8,
           sig=sg)
    rows, rcols = [], []                                         # UWYK-B: No-Anc B, 2D | Anc B, 2D (PEHE, eps each)
    for sec, tag, bl in SEC:
        b = ub_t[sec]
        for off, sub in ((0, "No-Anc"), (4, "Anc")):
            rows.append((f"{bl}\n{sub}", np.array([[r[1][off + k][0] for r in b], [r[1][off + 2 + k][0] for r in b]]),
                         np.array([[r[1][off + k][1] for r in b], [r[1][off + 2 + k][1] for r in b]]))); rcols.append(CCOL[tag])
    sg = [[c[3] for c in sp[{"rc": "rc", "cs": "cs", "cm": "cmall"}[t]]] for _, t, _ in SEC for _ in (0, 1)] if mt == "pehe" else None
    figure(rows, rcols, ["UWYK-B", "2D"], [PS.COL_1D, PS.COL_2D], f"cmp_uwykb_point_{mt}", lab, kind="point", row_h=0.8,
           sig=sg)


# ---- case studies by covariate dimension d, point estimation (fig:dsplit-pehe / fig:dsplit-ate)
# revision/results/d262t1/cs_point_by_d.csv: the merged CSV of D262k stage 19 (cs_point_total.py --by-d), copied from
# $SCRATCH/cs_point_by_d_d262t1.csv. Model keys as plot_fig3_dotgrid.py (first key present wins).
PBD = os.path.join(RES, "cs_point_by_d.csv")
if os.path.exists(PBD):
    pbd = pd.read_csv(PBD)
    pbd["d"] = pbd["d"].astype(int)
    KEYS = {"Do-PFN": ("dopfn_native",),
            "Do-PFN 2D": ("dopfn_repro_joint2d_262k_t1", "dopfn_repro_joint2d_262k", "dopfn_repro_joint2d"),
            "UWYK No-Anc": ("uwyk1d-noanc", "uwyk_noanc"), "UWYK No-Anc 2D": ("graph2d-noanc", "graph2d_noanc"),
            "UWYK Anc": ("uwyk1d-v3a", "uwyk_v3a"), "UWYK Anc 2D": ("graph2d-v3a", "graph2d_v3a"),
            "CausalPFN-C": ("cpfn1d_j1024", "cpfn1d", "cpfn1d_perarm"), "CausalPFN-C 2D": ("cpfn2d_eta0", "cpfn2d_pooled", "cpfn2d")}
    have = set(pbd["model"])
    key = {k: next(x for x in v if x in have) for k, v in KEYS.items()}
    DS = sorted(pbd["d"].unique())
    T = pbd.set_index(["case", "d", "model"])
    TAG = {"Observed_Confounder": "oc", "Observed_Mediator": "om", "Observed_Mediator_and_Confounder": "omc",
           "Unobserved_Confounder": "uc", "Frontdoor_Criterion": "fd", "Backdoor_Criterion": "bd"}
    for metric, se, lab, mt in (("pehe", "pehe_sem", r"$\sqrt{\mathrm{PEHE}}$", "pehe"),
                                ("eps_ate", "eps_sem", r"$\epsilon_{\mathrm{ATE}}$", "eps")):
        for case in CS_KEYS:
            rows, sg = [], []
            for fam in ("Do-PFN", "UWYK No-Anc", "UWYK Anc", "CausalPFN-C"):
                ms = (key[fam], key[fam + " 2D"])
                V = np.array([[T.at[(case, d, m), metric] for d in DS] for m in ms])
                E = np.array([[T.at[(case, d, m), se] for d in DS] for m in ms])
                rows.append((FAMROW[fam], V, E))
                sg.append([sig["main"]["case_study"][case]] * len(DS))
            figure(rows, [f"$d={d}$" for d in DS], ["1D", "2D"], [PS.COL_1D, PS.COL_2D], f"dsplit_{mt}_{TAG[case]}",
                   lab, kind="point", row_h=0.6, sig=sg if mt == "pehe" else None)
