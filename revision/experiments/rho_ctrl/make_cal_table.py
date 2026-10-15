"""LaTeX calibration table of the controlled-rho benchmarks (tab:rho-ctrl-cal) from rho_ctrl.csv.
python make_cal_table.py rho_ctrl.csv > table.tex"""
import csv, math, sys
r = list(csv.DictReader(open(sys.argv[1])))
PAIRS = [("Do-PFN", "Do-PFN 2D", "Do-PFN", "Do-PFN 2D"),
         ("UWYK No-Anc", "UWYK 2D No-Anc", "UWYK No-Anc", "UWYK No-Anc 2D"),
         ("UWYK Anc", "UWYK 2D Anc", "UWYK Anc", "UWYK Anc 2D"),
         ("CausalPFN-C", "CausalPFN-C 2D", "CausalPFN-C", "CausalPFN-C 2D"),
         ("CausalPFN-C (matched resolution)", "CausalPFN-C 2D", "CausalPFN-C (matched resolution)", "CausalPFN-C 2D")]
B = ["[-0.5, 0)", "[0, 0.5)", "[0.5, 1]"]; BL = [r"$\rho\in[-0.5,0)$", r"$\rho\in[0,0.5)$", r"$\rho\in[0.5,1]$"]
ORDER = ["Observed Confounder", "Backdoor Criterion", "Frontdoor Criterion", "Unobserved Confounder", "ComplexMech n=5"]
TITLE = {"ComplexMech n=5": "ComplexMech ($n{=}5$)", "Backdoor Criterion": "Backdoor", "Frontdoor Criterion": "Frontdoor"}
g = lambda b, m, bl: [y for y in r if y["benchmark"] == b and y["model"] == m and y["bin"] == bl][0]
F = float
def cell(v, se, d, bold):
    s = f"{v:.{d}f}"; s = r"\textbf{" + s + "}" if bold else s
    return s + (r"\,{\scriptsize$\pm$" + f"{se:.{d}f}" + "}" if math.isfinite(se) else "")
hdr = " & ".join([r"\multicolumn{2}{c}{" + x + "}" for x in BL] + [r"\multicolumn{2}{c}{\textbf{Average}}", r"\multicolumn{1}{c}{\textbf{Combined score}}"])
sub = " & ".join([r"$|\hat{c}{-}0.95|\downarrow$ & Length$\downarrow$"] * 4 + [r"$\mathrm{IS}_{0.05}\downarrow$"])
C = [r"\begin{table*}[p]\color{red}", r"\centering",
     r"\caption{\small 95\% posterior predictive interval calibration for the ITE on the benchmarks with a controlled cross-world correlation, reported per bin of $\rho$ (computed exactly per dataset) ($|\hat c - 0.95|$, Length), computed as in Table~\ref{tab:rc-cal}. Average reports the mean $|\hat c - 0.95|$ and length over the three bins; the combined score is the mean interval score $\mathrm{IS}_{0.05}$ over the three bins, in outcome units. The last pair of each block compares CausalPFN-C 2D with its baseline at matched resolution. Each bin holds $180$ datasets per case study ($162$ in the lowest bin of Unobserved Confounder) and $100$ for ComplexMech. Mean $\pm$ standard error over datasets; bold marks the better model of each pair.}",
     r"\label{tab:rho-ctrl-cal}", r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}\renewcommand{\arraystretch}{0.82}", r"\resizebox{\textwidth}{!}{%",
     r"\begin{tabular}{l cc cc cc cc c}", r"\toprule", "& " + hdr + r" \\",
     r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-10}", r"\textbf{Model} & " + sub + r" \\"]
for b in ORDER:
    C += [r"\midrule", r"\multicolumn{10}{l}{\textit{" + TITLE.get(b, b) + r"}} \\", r"\midrule"]
    for pi, (a1, a2, n1, n2) in enumerate(PAIRS):
        V = {}
        for m in (a1, a2):
            xs = [g(b, m, bl) for bl in B]
            v = V[m] = dict(dev=[abs(F(x["coverage"]) - 0.95) for x in xs], dse=[F(x["coverage_se"]) for x in xs],
                            L=[F(x["length"]) for x in xs], Lse=[F(x["length_se"]) for x in xs], IS=[F(x["IS"]) for x in xs])
            v["a"] = sum(v["dev"]) / 3; v["aL"] = sum(v["L"]) / 3; v["aIS"] = sum(v["IS"]) / 3
        for m, lab in ((a1, n1), (a2, n2)):
            v = V[m]; o = V[a2 if m == a1 else a1]; row = [lab]
            for i in range(3):
                row += [cell(v["dev"][i], v["dse"][i], 3, v["dev"][i] < o["dev"][i]), cell(v["L"][i], v["Lse"][i], 2, v["L"][i] < o["L"][i])]
            row += [cell(v["a"], math.nan, 3, v["a"] < o["a"]), cell(v["aL"], math.nan, 2, v["aL"] < o["aL"]), cell(v["aIS"], math.nan, 2, v["aIS"] < o["aIS"])]
            C.append(" & ".join(row) + r" \\")
        if pi < len(PAIRS) - 1:
            C.append(r"\cmidrule(lr){1-10}")
C += [r"\bottomrule", r"\end{tabular}%", "}", r"\end{table*}"]
print("\n".join(C))
