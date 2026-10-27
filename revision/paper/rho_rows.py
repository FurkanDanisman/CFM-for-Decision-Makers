"""Main calibration tables (tab:rc-cal, tab:cs-cal, tab:cm-r99-cal) and the controlled-rho table (tab:rho-ctrl-cal)
with every single-outcome model at the couplings rho = -0.5, 0, 0.5, 1 (Gaussian copula; rho = 0 = independent,
rho = 1 = comonotone), followed by its 2D model. Bold = best of the block (the 1D rows and the 2D row).

  main tables: rho = 0 and 2D rows from update_tables.load() (the rows the tables already had; checked below),
               rho = -0.5, 0.5, 1 from revision/results/d262t1/coupling.csv; commented LD=Off rows kept as comments
  rho-ctrl:    rho = 0 and 2D from rho_ctrl.csv, the others from rho_ctrl_{gauss_-0.5,gauss_0.5,comonotonic}.csv

    python revision/paper/rho_rows.py [--dry-run]
"""
import csv
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import update_tables as U                                        # noqa: E402
import coupling_tables as CT                                     # noqa: E402

TEX = CT.TEX
RES = os.path.join(HERE, "..", "results", "d262t1")
RHO = [("gauss_-0.5", "-0.5"), (None, "0"), ("gauss_0.5", "0.5"), ("comonotonic", "1")]
lab1 = lambda name, r: f"{name} ($\\rho{{=}}{r}$)"


def b(s, on):
    return r"\textbf{" + s + "}" if on else s


def cell(v, se, d):
    return (f"{v:.{d}f}", r"\,{\scriptsize$\pm$" + f"{se:.{d}f}" + "}" if se is not None and math.isfinite(se) else "")


def block(rows):
    """rows: [(label, [(value, se, digits) ...])]; bold the minimum of every column over the block."""
    K = len(rows[0][1])
    best = [min(round(r[1][j][0], r[1][j][2]) for r in rows) for j in range(K)]
    out = []
    for lab, vals in rows:
        cs = []
        for j, (v, se, d) in enumerate(vals):
            s, e = cell(v, se, d)
            cs.append(b(s, round(v, d) == best[j]) + e)
        out.append(lab + " & " + " & ".join(cs) + r" \\")
    return out


def main_vals(rs, cols):
    v = []
    for r in rs:
        dl = 0 if r.length >= 1000 else 2
        v += [(r.abs_dev, r.se, 3), (r.length, r.length_se, dl)]
    v += [(np.mean([r.abs_dev for r in rs]), None, 3),
          (np.mean([r.length / c[3] for r, c in zip(rs, cols)]), None, 2),
          (np.mean([r.is05 / c[3] for r, c in zip(rs, cols)]), None, 2)]
    return v


def strip(line):
    return re.sub(r"\\textbf\{([^}]*)\}", r"\1", line).strip()


def main_tables(s, d, cp, sp):
    for label, key in (("tab:rc-cal", "rc"), ("tab:cs-cal", "cs"), ("tab:cm-r99-cal", "cmall")):
        i = s.index(f"\\label{{{label}}}")
        a = s.index(r"\midrule", s.index(r"\begin{tabular}", i))
        z = s.index(r"\bottomrule", a)
        old = s[a:z].splitlines()
        cols = sp[key]
        get = lambda src, m: [d.get((src, bm, case, sl, m)) for bm, case, sl, _ in cols]
        getc = lambda st, m: [cp.get((st, bm, case, sl, m)) for bm, case, sl, _ in cols]
        new = []
        for fi, (l1, k1, c1, l2, k2) in enumerate(CT.MAIN):
            rows = []
            for st, r in RHO:
                rs = get("hist", k1) if st is None else getc(st, c1)
                assert all(x is not None for x in rs), (label, l1, st)
                rows.append((lab1(l1, r), main_vals(rs, cols)))
            rows.append((l2, main_vals(get("hist", k2), cols)))
            lines = block(rows)
            # the rows the table had: rho = 0 (old label l1) and the 2D row must be unchanged
            had = {strip(x).split(" & ", 1)[0]: strip(x).split(" & ", 1)[1] for x in old
                   if " & " in x and not x.lstrip().startswith("%")}
            assert had[l1] == strip(lines[1]).split(" & ", 1)[1], (label, l1, had[l1], strip(lines[1]))
            assert had[l2] == strip(lines[4]).split(" & ", 1)[1], (label, l2)
            ldoff = [x for x in old if x.lstrip().startswith("%") and x.lstrip("% ").startswith(l2 + " (LD=Off)")]
            new += ([r"\midrule"] if True else []) + lines + ldoff
        s = s[:a] + "\n".join(new) + "\n" + s[z:]
    return s


def rho_ctrl(s):
    load = lambda f: list(csv.DictReader(open(os.path.join(RES, f))))
    R = {None: load("rho_ctrl.csv")}
    for st, _ in RHO:
        if st:
            R[st] = load(f"rho_ctrl_{st}.csv")
    M = __import__("importlib.util").util
    spec = M.spec_from_file_location("mct", os.path.join(HERE, "..", "experiments", "rho_ctrl", "make_cal_table.py"))
    src = open(spec.origin).read()
    PAIRS = eval(re.search(r"PAIRS = (\[.*?\])\n", src, re.S).group(1))
    B = ["[-0.5, 0)", "[0, 0.5)", "[0.5, 1]"]
    ORDER = ["Observed Confounder", "Backdoor Criterion", "Frontdoor Criterion", "Unobserved Confounder", "ComplexMech n=5"]
    TITLE = {"ComplexMech n=5": "ComplexMech ($n{=}5$)", "Backdoor Criterion": "Backdoor", "Frontdoor Criterion": "Frontdoor"}

    def vals(rows, bm, m):
        xs = [[y for y in rows if y["benchmark"] == bm and y["model"] == m and y["bin"] == bl][0] for bl in B]
        F = lambda x, k: float(x[k])
        v = []
        for x in xs:
            v += [(abs(F(x, "coverage") - 0.95), F(x, "coverage_se"), 3), (F(x, "length"), F(x, "length_se"), 2)]
        v += [(np.mean([abs(F(x, "coverage") - 0.95) for x in xs]), None, 3),
              (np.mean([F(x, "length") for x in xs]), None, 2), (np.mean([F(x, "IS") for x in xs]), None, 2)]
        return v

    i = s.index(r"\label{tab:rho-ctrl-cal}")
    a = s.index(r"\midrule", s.index(r"\begin{tabular}", i))
    z = s.index(r"\bottomrule", a)
    old = s[a:z]
    out = []
    for bm in ORDER:
        out += [r"\midrule", r"\multicolumn{10}{l}{\textit{" + TITLE.get(bm, bm) + r"}} \\", r"\midrule"]
        for pi, (a1, a2, n1, n2) in enumerate(PAIRS):
            rows = [(lab1(n1, r), vals(R[st], bm, a1)) for st, r in RHO] + [(n2, vals(R[None], bm, a2))]
            lines = block(rows)
            out += lines
            if pi < len(PAIRS) - 1:
                out.append(r"\cmidrule(lr){1-10}")
    # the rho = 0 and 2D cells must equal the table's (bold aside)
    oldv = [strip(x).split(" & ", 1)[1] for x in old.splitlines() if " & " in x]
    newv = [strip(x).split(" & ", 1)[1] for x in out if " & " in x and "rho{=}0$)" in x.split(" & ")[0]
            or (" & " in x and "rho{=}" not in x.split(" & ")[0])]
    assert oldv == newv, "rho-ctrl: rho = 0 / 2D rows differ from the table"
    return s[:a] + "\n".join(out) + "\n" + s[z:]


def main():
    d, sig = U.load()
    sp = U.specs(sig)
    cp = CT.coupling_rows()
    s = open(TEX).read()
    s = main_tables(s, d, cp, sp)
    s = rho_ctrl(s)
    if "--dry-run" in sys.argv:
        p = os.path.join(os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "tmp", "rho_rows_dry.tex")
        open(p, "w").write(s); print("dry run:", p)
    else:
        open(TEX, "w").write(s); print("rewrote", TEX)


if __name__ == "__main__":
    main()
