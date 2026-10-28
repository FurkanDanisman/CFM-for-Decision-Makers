"""Calibration of the two comparison appendices (published two-stage CausalPFN, UWYK-B) in the format of the main
calibration tables, with the single-outcome model at rho = -0.5, 0, 0.5, 1 (as rho_rows.py), followed by the 2D
model. New tables tab:cpfn-2d-cal and tab:uwyk-binarization-cal are inserted after tab:cpfn-2d /
tab:uwyk-binarization, whose calibration columns are removed (they keep point estimation).

  rho = 0 and 2D rows: update_tables.load() (the cells the old tables showed; their averages are checked below)
  rho = -0.5, 0.5, 1:  revision/results/d262t1/coupling.csv

    python revision/paper/cmp_tables.py [--dry-run]
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import update_tables as U                                        # noqa: E402
import coupling_tables as CT                                     # noqa: E402
import rho_rows as RR                                            # noqa: E402

TEX = CT.TEX
PANELS = [("rc", "RealCause"), ("cs", "Case studies"), ("cmall", "ComplexMech")]
CMP = {
    "tab:cpfn-2d": dict(new="tab:cpfn-2d-cal", fam=[("CausalPFN", "CausalPFN (two-stage, published)", "CausalPFN-C 2D", "CausalPFN 2D")],
                        what="CausalPFN-C 2D against the published two-stage CausalPFN checkpoint"),
    "tab:uwyk-binarization": dict(new="tab:uwyk-binarization-cal",
                                  fam=[("UWYK-B No-Anc", "UWYK-B No-Anc", "UWYK No-Anc 2D", "UWYK 2D (noanc)"),
                                       ("UWYK-B Anc", "UWYK-B Anc", "UWYK Anc 2D", "UWYK 2D (v3a)")],
                                  what="UWYK 2D against the re-trained binary-treatment baseline (UWYK-B)"),
}


def panel(key, title, fam, d, cp, sp):
    cols = sp[key]; K = len(cols)
    get = lambda src, m: [d.get((src, bm, case, sl, m)) for bm, case, sl, _ in cols]
    getc = lambda st, m: [cp.get((st, bm, case, sl, m)) for bm, case, sl, _ in cols]
    out = [r"\begin{tabular}{l" + " cc" * K + " ccc}", r"\toprule",
           r"\multicolumn{" + str(2 * K + 4) + r"}{l}{\textit{" + title + r"}} \\", r"\midrule",
           "& " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{n}}}" for n in CT.NAMES[key])
           + r" & \multicolumn{2}{c}{\textbf{Average}} & \multicolumn{1}{c}{\textbf{Combined score}} \\",
           "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(K))
           + f"\\cmidrule(lr){{{2 + 2 * K}-{3 + 2 * K}}}\\cmidrule(lr){{{4 + 2 * K}-{4 + 2 * K}}}",
           r"\textbf{Model} & " + " & ".join([r"$|\hat{c}{-}0.95|\downarrow$ & Length$\downarrow$"] * K)
           + r" & $|\hat{c}{-}0.95|\downarrow$ & Length$/\sigma_Y\downarrow$ & IS$_{0.05}/\sigma_Y\downarrow$ \\"]
    avg = {}
    for l1, k1, l2, k2 in fam:
        out.append(r"\midrule")
        rows = []
        for st, r in RR.RHO:
            rs = get("hist", k1) if st is None else getc(st, k1)
            assert all(x is not None for x in rs), (key, k1, st)
            rows.append((RR.lab1(l1, r), RR.main_vals(rs, cols)))
        rows.append((l2, RR.main_vals(get("hist", k2), cols)))
        out += RR.block(rows)
        avg[l1] = [v for v, _, _ in rows[1][1][-3:]]
        avg[l2] = [v for v, _, _ in rows[4][1][-3:]]
    out += [r"\bottomrule", r"\end{tabular}"]
    return out, avg


def old_averages(s, label):
    """the old table's calibration averages per section: {section: [cells after the point columns]}"""
    i = s.index(f"\\label{{{label}}}"); z = s.index(r"\end{tabular}", i)
    res, sec = {}, None
    for l in s[i:z].splitlines():
        m = re.search(r"\\textit\{([^}]*)\}", l)
        if "multicolumn" in l and m:
            sec = m.group(1)
        elif l.startswith(r"$\Delta$") and sec:
            res[sec] = [float(re.sub(r"\\textbf\{([^}]*)\}", r"\1", c).strip(" \\$")) for c in l.split("&")[1:]
                        if re.fullmatch(r"\s*(\\textbf\{)?[\d.]+\}?\s*(\\\\)?\s*", c)]
    return res


def strip_cal(s, label):
    """drop the calibration columns of a comparison table (keep the point columns)"""
    i = s.index(f"\\label{{{label}}}"); a = s.index(r"\begin{tabular}", i); z = s.index(r"\end{tabular}", a)
    body = s[a:z].split("\n")
    uw = label == "tab:uwyk-binarization"
    keep = [0, 1, 2, 3, 4, 11, 12, 13, 14] if uw else [0, 1, 2, 3, 4]
    out = []
    for l in body:
        if l.startswith(r"\begin{tabular}"):
            out.append(r"\begin{tabular}{l cc cc || cc cc}" if uw else r"\begin{tabular}{l cc cc}")
        elif l.startswith(r"\cmidrule"):
            out.append(r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}" if uw and "{2-11}" in l else
                       r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}" + (r"\cmidrule(lr){6-7}\cmidrule(lr){8-9}" if uw else ""))
        elif uw and l.startswith(r"& \multicolumn{10}"):
            out.append(r"& \multicolumn{4}{c}{No-Anc} & \multicolumn{4}{c}{Anc} \\")
        elif l.startswith(r"\multicolumn{") and "textit" in l:
            out.append(re.sub(r"\\multicolumn\{\d+\}", r"\\multicolumn{%d}" % len(keep), l))
        elif "&" in l and not l.lstrip().startswith("%"):
            c = [x.strip() for x in l.rstrip().rstrip("\\").split("&")]
            if uw and l.startswith("& \\multicolumn{2}"):          # model-name header row: 4 + 4 groups of 2/3 cols
                g = [x.strip() for x in l.rstrip().rstrip("\\").split("&")][1:]
                out.append("& " + " & ".join(g[:2] + g[4:6]) + r" \\")
                continue
            if l.startswith("& \\multicolumn{2}"):
                g = [x.strip() for x in l.rstrip().rstrip("\\").split("&")][1:]
                out.append("& " + " & ".join(g[:2]) + r" \\")
                continue
            c = [c[k] for k in keep]
            if c[0].startswith(r"$\Delta$"):
                c[0] = r"$\Delta$"
            out.append(" & ".join(c) + r" \\")
        else:
            out.append(l)
    return s[:a] + "\n".join(out) + s[z:]


def main():
    d, sig = U.load()
    sp = U.specs(sig)
    cp = CT.coupling_rows()
    s = open(TEX).read()
    for label, c in CMP.items():
        old = old_averages(s, label)
        tabs = []
        for key, title in PANELS:
            t, avg = panel(key, title, c["fam"], d, cp, sp)
            tabs.append("\n".join(t))
            sec = {"rc": "RealCause", "cs": "Case studies", "cmall": "ComplexMech"}[key]
            flat = [round(v, 3 if j == 0 else 2) for f in c["fam"] for m in (f[0], f[2]) for j, v in enumerate(avg[m])]
            assert [round(v, 3 if j % 3 == 0 else 2) for j, v in enumerate(old[sec])] == flat, (label, sec, old[sec], flat)
        new = "\n".join([
            r"\begin{table*}[t]", r"\centering",
            r"\caption{\small 95\% posterior predictive interval calibration of " + c["what"]
            + r" ($|\hat c - 0.95|$, Length), computed as in Table~\ref{tab:rc-cal}. As in Tables~\ref{tab:rc-cal}--\ref{tab:cm-r99-cal}, the single-outcome model is reported under each assumed $\rho$ and the 2D model with its learned dependence; bold marks the best model of each block.}",
            r"\label{" + c["new"] + "}", r"\small", r"\setlength{\tabcolsep}{2.5pt}",
            r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{c}", ("\\\\[4pt]\n").join(tabs), r"\end{tabular}%", "}", r"\end{table*}"])
        s = strip_cal(s, label)
        e = s.index(r"\end{table*}", s.index(f"\\label{{{label}}}")) + len(r"\end{table*}")
        s = s[:e] + "\n\n" + new + s[e:]
        # caption of the old table: point estimation only now
        i = s.index(f"\\label{{{label}}}"); c0 = s.rindex(r"\caption{", 0, i)
        cap = s[c0:i]
        old_txt = (r"Left columns report point estimation ($\sqrt{\mathrm{PEHE}}$, $\epsilon_{\mathrm{ATE}}$; lower is better), "
                   r"right columns 95\% posterior predictive interval calibration ($|\hat c - 0.95|$, $\mathrm{Length}/\sigma_Y$, "
                   r"$\mathrm{IS}_{0.05}/\sigma_Y$), computed as in Tables~\ref{tab:rc-point} and~\ref{tab:rc-cal}. $\Delta$ / Average rows")
        new_txt = (r"Point estimation ($\sqrt{\mathrm{PEHE}}$, $\epsilon_{\mathrm{ATE}}$; lower is better), computed as in "
                   r"Table~\ref{tab:rc-point}; Table~\ref{" + c["new"] + r"} reports the calibration. $\Delta$ rows")
        assert cap.count(old_txt) == 1, ("caption text not found", label)
        cap2 = cap.replace(old_txt, new_txt)
        assert cap2 != cap, ("caption not rewritten", label)
        s = s[:c0] + cap2 + s[i:]
    if "--dry-run" in sys.argv:
        p = os.path.join(os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "tmp", "cmp_dry.tex"); open(p, "w").write(s); print("dry run:", p)
    else:
        open(TEX, "w").write(s); print("rewrote", TEX)


if __name__ == "__main__":
    main()
