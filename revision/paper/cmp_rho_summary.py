"""Compact rho tables for the published-CausalPFN and UWYK-B comparisons, inserted after tab:cpfn-2d and
tab:uwyk-binarization (those tables are left as they are). Per benchmark only the averages of the main tables:
|c - 0.95|, Length/sigma_Y, IS_0.05/sigma_Y; rows: the single-outcome model at rho = -0.5, 0, 0.5, 1 and the 2D model.
rho = 0 / 2D from update_tables.load(), the others from coupling.csv; bold = best of each block.
    python revision/paper/cmp_rho_summary.py [--dry-run]"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import update_tables as U, coupling_tables as CT, rho_rows as RR   # noqa: E402

TEX = CT.TEX
CMP = [("tab:cpfn-2d", "tab:cpfn-2d-rho", "CausalPFN-C 2D and the published two-stage CausalPFN",
        [("CausalPFN", "CausalPFN (two-stage, published)", "CausalPFN-C 2D", "CausalPFN 2D")]),
       ("tab:uwyk-binarization", "tab:uwyk-binarization-rho", "UWYK 2D and the re-trained binary baseline UWYK-B",
        [("UWYK-B No-Anc", "UWYK-B No-Anc", "UWYK No-Anc 2D", "UWYK 2D (noanc)"),
         ("UWYK-B Anc", "UWYK-B Anc", "UWYK Anc 2D", "UWYK 2D (v3a)")])]


def table(new, what, fam, d, cp, sp):
    out = [r"\begin{table}[t]", r"\centering",
           r"\caption{\small Calibration of " + what + r" under each assumed $\rho$ for the single-outcome model, as in Tables~\ref{tab:rc-cal}--\ref{tab:cm-r99-cal}; the 2D model uses its learned dependence. Averages over the datasets of each benchmark: $|\hat c - 0.95|$, Length$/\sigma_Y$ and the combined score IS$_{0.05}/\sigma_Y$; bold marks the best of each block.}",
           r"\label{" + new + "}", r"\scriptsize", r"\setlength{\tabcolsep}{3.5pt}",
           r"\begin{tabular}{l ccc ccc ccc}", r"\toprule",
           r"& \multicolumn{3}{c}{RealCause} & \multicolumn{3}{c}{Case studies} & \multicolumn{3}{c}{ComplexMech} \\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
           r"\textbf{Model} & " + " & ".join([r"$|\hat{c}{-}0.95|$ & Len$/\sigma_Y$ & IS$/\sigma_Y$"] * 3) + r" \\"]
    for l1, k1, l2, k2 in fam:
        out.append(r"\midrule")
        rows = []
        for st, r in RR.RHO:
            vals = []
            for key in ("rc", "cs", "cmall"):
                cols = sp[key]
                rs = [d.get(("hist", bm, c, sl, k1)) if st is None else cp.get((st, bm, c, sl, k1)) for bm, c, sl, _ in cols]
                assert all(x is not None for x in rs), (k1, st, key)
                vals += [(v, None, dd) for v, _, dd in RR.main_vals(rs, cols)[-3:]]
            rows.append((RR.lab1(l1, r), vals))
        vals = []
        for key in ("rc", "cs", "cmall"):
            cols = sp[key]
            vals += [(v, None, dd) for v, _, dd in RR.main_vals([d.get(("hist", bm, c, sl, k2)) for bm, c, sl, _ in cols], cols)[-3:]]
        rows.append((l2, vals))
        out += RR.block(rows)
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def split_uwyk(s):
    """tab:uwyk-binarization: one 21-column tabular -> two stacked 11-column panels (No-Anc, Anc), same cells."""
    i = s.index(r"\label{tab:uwyk-binarization}"); a = s.index(r"\begin{tabular}", i); z = s.index(r"\end{tabular}", a)
    lines = s[a:z].split("\n")
    assert lines[0].startswith(r"\begin{tabular}{l cc cc | ccc ccc || cc cc | ccc ccc}"), lines[0]
    panels = []
    for half, title in ((0, "No-Anc"), (1, "Anc")):
        out = [r"\begin{tabular}{l cc cc | ccc ccc}", r"\toprule",
               r"\multicolumn{11}{l}{\textbf{" + title + r"}} \\", r"\midrule"]
        for l in lines[1:]:
            st = l.strip()
            if st in (r"\toprule", ""):
                continue
            if st.startswith(r"& \multicolumn{10}"):
                continue                                          # the No-Anc / Anc header becomes the panel title
            if st.startswith(r"\cmidrule(lr){2-11}"):
                continue
            if st.startswith(r"\cmidrule"):
                rng = [tuple(map(int, m)) for m in __import__("re").findall(r"\{(\d+)-(\d+)\}", st)]
                keep = [r for r in rng if (r[0] <= 11) == (half == 0)]
                out.append("".join(f"\\cmidrule(lr){{{a - 10 * half}-{b - 10 * half}}}" for a, b in keep)); continue
            if st.startswith(r"& \multicolumn{2}"):                # model-name header: 4 groups per variant
                g = [x.strip() for x in st.rstrip("\\").rstrip().rstrip("\\").split("&")][1:]
                out.append("& " + " & ".join(g[4 * half: 4 * half + 4]) + r" \\"); continue
            if "&" in st and not st.startswith("%"):
                c = [x.strip() for x in st.rstrip("\\").rstrip().rstrip("\\").split("&")]
                if len(c) == 21:
                    out.append(" & ".join([c[0]] + c[1 + 10 * half: 11 + 10 * half]) + r" \\"); continue
            out.append(l.replace(r"\multicolumn{21}", r"\multicolumn{11}"))
        out.append(r"\end{tabular}")
        panels.append("\n".join(out))
    return (s[:a] + r"\begin{tabular}{@{}c@{}}" + "\n" + panels[0] + "\n\\\\[8pt]\n" + panels[1] + "\n"
            + r"\end{tabular}" + s[z + len(r"\end{tabular}"):])


def main():
    d, sig = U.load(); sp = U.specs(sig); cp = CT.coupling_rows()
    s = open(TEX).read()
    s = split_uwyk(s)
    for old, new, what, fam in CMP:
        assert f"\\label{{{new}}}" not in s, f"{new} already in the tex"
        e = s.index(r"\end{table*}", s.index(f"\\label{{{old}}}")) + len(r"\end{table*}")
        s = s[:e] + "\n\n" + table(new, what, fam, d, cp, sp) + s[e:]
    if "--dry-run" in sys.argv:
        p = os.path.join(os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "tmp", "cmp_rho_dry.tex"); open(p, "w").write(s); print("dry:", p)
    else:
        open(TEX, "w").write(s); print("rewrote", TEX)


if __name__ == "__main__":
    main()
