"""tab:cs-d-all (case studies by dimension) with the single-outcome models at rho = -0.5, 0, 0.5, 1 as rows, as in
the main tables, split over two tables (three scenarios each). Removes the per-rho copies (tab:cs-d-all-m05/-p05/-p1).
rho = 0 and 2D rows must equal the current tab:cs-d-all (checked). Bold = best of each block.
    python revision/paper/csd_inline.py [--dry-run] [--parts N]"""
import os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import update_tables as U, coupling_tables as CT, rho_rows as RR, rho_extra_tables as RX   # noqa: E402

TEX = CT.TEX
NAMES = {"Observed_Confounder": "Observed Confounder", "Observed_Mediator": "Observed Mediator",
         "Observed_Mediator_and_Confounder": "Observed Mediator + Confounder", "Unobserved_Confounder": "Unobserved Confounder",
         "Frontdoor_Criterion": "Frontdoor", "Backdoor_Criterion": "Backdoor"}
strip = lambda l: re.sub(r"\\textbf\{([^}]*)\}", r"\1", l).strip()


def vals(rs, sd):
    v = [c for r in rs for c in ((r.abs_dev, r.se, 3), (r.length, r.length_se, 0 if r.length >= 1000 else 2))]
    return v + [(np.mean([r.abs_dev for r in rs]), None, 3), (np.mean([r.length for r in rs]) / sd, None, 2),
                (np.mean([r.is05 for r in rs]) / sd, None, 2)]


def blocks(d, sig, cp, case):
    sd = sig["main"]["case_study"][case]; out = []
    for (l1, m1), (l2, m2) in RX.PAIRS_D:
        rows = []
        for st, rv in RR.RHO:
            rs = [d[("hist", "CaseStudy", case, f"d{x}", m1)] if st is None else cp[(st, "CaseStudy", case, f"d{x}", m1)] for x in RX.DS]
            rows.append((RR.lab1(l1, rv), vals(rs, sd)))
        rows.append((l2, vals([d[("hist", "CaseStudy", case, f"d{x}", m2)] for x in RX.DS], sd)))
        out.append(RR.block(rows))
    return out


def main():
    parts = int(sys.argv[sys.argv.index("--parts") + 1]) if "--parts" in sys.argv else 2
    d, sig = U.load(); cp = CT.coupling_rows()
    s = open(TEX).read()
    for suf in ("-m05", "-p05", "-p1"):                                  # drop the per-rho copies
        a, z = RX.table_span(s, "tab:cs-d-all" + suf); s = s[:a].rstrip() + "\n\n" + s[z:].lstrip()
    a, z = RX.table_span(s, "tab:cs-d-all"); tab = s[a:z]
    head = tab[:tab.index(r"\midrule")]                                  # caption, label, header rows
    old = [strip(l) for l in tab[tab.index(r"\midrule"):tab.index(r"\bottomrule")].split("\n") if " & " in l and not l.lstrip().startswith("%")]
    allb = {c: blocks(d, sig, cp, c) for c in U.CASES}
    # check: the rho = 0 and 2D rows equal the table as it stands
    new0 = []
    for c in U.CASES:
        for b in allb[c]:
            for k in (1, 4):
                new0.append(strip(b[k]).split(" & ", 1)[1])
    assert [o.split(" & ", 1)[1] for o in old] == new0, "rho = 0 / 2D rows differ from the current table"
    K = 6
    cases = U.CASES; per = -(-len(cases) // parts); tables = []
    for p in range(parts):
        cs = cases[p * per:(p + 1) * per]
        lab = "tab:cs-d-all" if p == 0 else f"tab:cs-d-all-{chr(ord('a') + p)}"
        h = head.replace(r"\label{tab:cs-d-all}", f"\\label{{{lab}}}")
        c0 = h.index(r"\caption{"); c1 = h.index(r"\label{", c0)
        cap = h[c0:c1].rstrip().rstrip("}")
        if p == 0:
            cap += (r" Each single-outcome model is reported under each assumed $\rho$, as in Tables~\ref{tab:rc-cal}--\ref{tab:cm-r99-cal}; "
                    r"bold marks the best of each block. " + ", ".join(NAMES[c] for c in cs) + r"; continued in Table~\ref{tab:cs-d-all-b}.")
        else:
            cap = (r"\caption{\small Table~\ref{tab:cs-d-all}, continued: " + ", ".join(NAMES[c] for c in cs) + ".")
        h = h[:c0] + cap + "}\n" + h[c1:]
        body = []
        for c in cs:
            body += [r"\midrule", r"\multicolumn{" + str(2 * K + 4) + r"}{l}{\textit{" + NAMES[c] + r"}} \\"]
            for i, b in enumerate(allb[c]):
                body.append(r"\midrule" if i == 0 else r"\cmidrule(lr){1-" + str(2 * K + 4) + "}")
                body += b
        tables.append(h + "\n".join(body) + "\n" + tab[tab.index(r"\bottomrule"):])
    s = s[:a] + "\n\n".join(tables) + s[z:]
    if "--dry-run" in sys.argv:
        q = os.path.join(os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "tmp", "csd_dry.tex"); open(q, "w").write(s); print("dry:", q)
    else:
        open(TEX, "w").write(s); print("rewrote", TEX)


if __name__ == "__main__":
    main()
