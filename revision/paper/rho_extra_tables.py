"""Extra-rho copies of two appendix tables: tab:cs-d-all (case studies by dimension) and tab:rho-ctrl-cal
(controlled rho). For each assumed rho in {-0.5, 0.5, 1} a table in the identical format is inserted right after
the original: the single-outcome rows combine their two predicted outcomes under that rho (Gaussian copula,
rho = 1 comonotone); the 2D rows are unchanged; bold marks the better of each pair, as in the originals.
Before writing, the rho = 0 version is rebuilt with the same code and must equal the table in the tex.
    python revision/paper/rho_extra_tables.py [--dry-run]"""
import csv, math, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import update_tables as U, coupling_tables as CT                 # noqa: E402

TEX = CT.TEX
RES = os.path.join(HERE, "..", "results", "d262t1")
STAGES = [("gauss_-0.5", "-0.5", "m05"), ("gauss_0.5", "0.5", "p05"), ("comonotonic", "1", "p1")]
strip = lambda l: re.sub(r"\\textbf\{([^}]*)\}", r"\1", l).strip()


def bold_pairs(rows):
    """rows: [(label, [(value, text) ...])] in pairs; bold the smaller value of each pair, per column"""
    out = []
    for k in range(0, len(rows), 2):
        (la, a), (lb, b) = rows[k], rows[k + 1]
        ca, cb = [], []
        for (va, ta), (vb, tb) in zip(a, b):
            ra, rb = round(va, 6), round(vb, 6)
            ca.append(r"\textbf{" + ta.split(r"\,")[0] + "}" + ta[len(ta.split(r"\,")[0]):] if ra < rb else ta)
            cb.append(r"\textbf{" + tb.split(r"\,")[0] + "}" + tb[len(tb.split(r"\,")[0]):] if rb < ra else tb)
        out.append((la, ca)); out.append((lb, cb))
    return out


# ---------------- tab:cs-d-all ----------------
PAIRS_D = [(("Do-PFN", "Do-PFN 1D"), ("Do-PFN 2D", "Do-PFN 2D (262k)")),
           (("UWYK No-Anc", "UWYK 1D (noanc)"), ("UWYK No-Anc 2D", "UWYK 2D (noanc)")),
           (("UWYK Anc", "UWYK 1D (v3a)"), ("UWYK Anc 2D", "UWYK 2D (v3a)")),
           (("CausalPFN-C", "CausalPFN 1D"), ("CausalPFN-C 2D", "CausalPFN 2D"))]
DS = ["5", "10", "20", "30", "40", "50"]


def csd_rows(d, sig, cp, stage):
    blocks = []
    for case in U.CASES:
        sd = sig["main"]["case_study"][case]
        rows = []
        for (l1, m1), (l2, m2) in PAIRS_D:
            r1 = [d[("hist", "CaseStudy", case, f"d{x}", m1)] if stage is None else cp[(stage, "CaseStudy", case, f"d{x}", m1)] for x in DS]
            r2 = [d[("hist", "CaseStudy", case, f"d{x}", m2)] for x in DS]
            for lab, rs in ((l1, r1), (l2, r2)):
                cells = [c for r in rs for c in ((r.abs_dev, U.fdev(r)), (r.length, U.flen(r)))]
                av = (np.mean([r.abs_dev for r in rs]), np.mean([r.length for r in rs]) / sd, np.mean([r.is05 for r in rs]) / sd)
                cells += [(av[0], f"{av[0]:.3f}"), (av[1], f"{av[1]:.2f}"), (av[2], f"{av[2]:.2f}")]
                rows.append((lab, cells))
        blocks.append((case, bold_pairs(rows)))
    return blocks


def csd_body(blocks, old_lines):
    """re-emit the data part of the table, keeping every non-data line (titles, rules) of the original"""
    it = iter([r for _, b in blocks for r in b])
    out = []
    for l in old_lines:
        st = l.strip()
        if "&" in st and not st.startswith("%") and not st.startswith(r"\textbf{Model}") and not st.startswith("&"):
            lab, cells = next(it)
            old_lab = st.split("&")[0].strip()
            assert re.sub(r"\\textbf\{([^}]*)\}", r"\1", old_lab) == lab, (old_lab, lab)
            out.append(old_lab + " & " + " & ".join(cells) + r" \\")
        else:
            out.append(l)
    return out


# ---------------- tab:rho-ctrl-cal ----------------
PAIRS_C = [("Do-PFN", "Do-PFN 2D", "Do-PFN", "Do-PFN 2D"), ("UWYK No-Anc", "UWYK 2D No-Anc", "UWYK No-Anc", "UWYK No-Anc 2D"),
           ("UWYK Anc", "UWYK 2D Anc", "UWYK Anc", "UWYK Anc 2D"), ("CausalPFN-C", "CausalPFN-C 2D", "CausalPFN-C", "CausalPFN-C 2D"),
           ("CausalPFN-C (matched resolution)", "CausalPFN-C 2D", "CausalPFN-C (matched resolution)", "CausalPFN-C 2D")]
B = ["[-0.5, 0)", "[0, 0.5)", "[0.5, 1]"]
ORDER = ["Observed Confounder", "Backdoor Criterion", "Frontdoor Criterion", "Unobserved Confounder", "ComplexMech n=5"]


def ctrl_rows(stage):
    base = list(csv.DictReader(open(os.path.join(RES, "rho_ctrl.csv"))))
    one = base if stage is None else list(csv.DictReader(open(os.path.join(RES, f"rho_ctrl_{stage}.csv"))))
    g = lambda rows, b, m, bl: [y for y in rows if y["benchmark"] == b and y["model"] == m and y["bin"] == bl][0]
    F = float
    def cells(rows, b, m):
        xs = [g(rows, b, m, bl) for bl in B]
        dev = [abs(F(x["coverage"]) - 0.95) for x in xs]; L = [F(x["length"]) for x in xs]; IS = [F(x["IS"]) for x in xs]
        c = []
        for x, dv, l in zip(xs, dev, L):
            c += [(dv, f"{dv:.3f}" + r"\,{\scriptsize$\pm$" + f"{F(x['coverage_se']):.3f}" + "}"),
                  (l, f"{l:.2f}" + r"\,{\scriptsize$\pm$" + f"{F(x['length_se']):.2f}" + "}")]
        a = (sum(dev) / 3, sum(L) / 3, sum(IS) / 3)
        return c + [(a[0], f"{a[0]:.3f}"), (a[1], f"{a[1]:.2f}"), (a[2], f"{a[2]:.2f}")]
    blocks = []
    for b in ORDER:
        rows = []
        for a1, a2, n1, n2 in PAIRS_C:
            rows += [(n1, cells(one, b, a1)), (n2, cells(base, b, a2))]
        blocks.append((b, bold_pairs(rows)))
    return blocks


def table_span(s, label):
    i = s.index(f"\\label{{{label}}}")
    a = s.rindex(r"\begin{table", 0, i); z = s.index(r"\end{table", i); z = s.index("}", z) + 1
    return a, z


def make_copies(s, label, rows_fn, suffix_note):
    a, z = table_span(s, label)
    tab = s[a:z]
    ta = tab.index(r"\midrule"); tz = tab.index(r"\bottomrule")
    old_lines = tab[ta:tz].split("\n")
    # check: the rho = 0 rebuild reproduces the table as it stands (bold aside)
    rebuilt = csd_body(rows_fn(None), old_lines)
    assert [strip(x) for x in rebuilt] == [strip(x) for x in old_lines], f"{label}: rho = 0 rebuild differs from the tex"
    copies = []
    for stage, rv, suf in STAGES:
        body = "\n".join(csd_body(rows_fn(stage), old_lines))
        t = tab[:ta] + body + tab[tz:]
        t = t.replace(f"\\label{{{label}}}", f"\\label{{{label}-{suf}}}")
        c0 = t.index(r"\caption{"); c1 = t.index(r"\label{", c0)
        cap = t[c0:c1].rstrip().rstrip("}")
        cap = re.sub(r" The single-outcome baselines are reported with their default setting.*?independent\.", "", cap)
        t = t[:c0] + cap + suffix_note.format(rv=rv) + "}\n" + t[c1:]
        copies.append(t)
    return s[:z] + "\n\n" + "\n\n".join(copies) + s[z:]


def main():
    d, sig = U.load(); cp = CT.coupling_rows()
    s = open(TEX).read()
    for lab in ("tab:cs-d-all", "tab:rho-ctrl-cal"):
        assert f"\\label{{{lab}-m05}}" not in s, f"{lab} copies already in the tex"
    note = r" Here the single-outcome baselines combine their two predicted outcomes under $\rho = {rv}$ (Gaussian copula; $\rho = 1$ is the comonotone combination); the 2D models are as in Table~\ref{{TAB}}."
    s = make_copies(s, "tab:cs-d-all", lambda st: csd_rows(d, sig, cp, st), note.replace("TAB", "tab:cs-d-all"))
    s = make_copies(s, "tab:rho-ctrl-cal", ctrl_rows, note.replace("TAB", "tab:rho-ctrl-cal"))
    if "--dry-run" in sys.argv:
        p = os.path.join(os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "tmp", "rho_extra_dry.tex"); open(p, "w").write(s); print("dry:", p)
    else:
        open(TEX, "w").write(s); print("rewrote", TEX)


if __name__ == "__main__":
    main()
