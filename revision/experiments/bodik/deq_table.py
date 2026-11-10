"""Jittered Twins: 2D (conditioned on the known twin) against 1D conditioned through a Gaussian copula at
rho = -0.5, 0, 0.5, 1 and the estimated rho (2D also through the copula at it), MSE and IS, from the score_twins_deq
CSVs; per jitter w and averaged over w = 0.1, 0.25, 0.5.
python deq_table.py CSV [CSV ...]

Significance (the paper's test, Appendix F): 2D (learned) against each 1D column, two-sided z-test on the difference
of the means with SE = sqrt(SE_2D^2 + SE_1D^2) (the two models treated as independent, as in the paper). Averaged over
w, the SE of a model's average is bounded by the average of its per-w SEs (exact under perfect correlation across w, an
upper bound otherwise, so no assumption on the dependence across w is needed). Hochberg correction within each model
and metric, over all its comparisons in that table (1D columns x d, and x w in the per-w table). Marks on a 1D cell:
+ 2D lower (better), - 2D higher (worse); * at 0.1, ** at 0.05; no mark = not significant.
"""
import csv
import sys

import numpy as np
from scipy.stats import norm

PAIRS = [("Do-PFN", "Do-PFN 2D"), ("CausalPFN-C", "CausalPFN-C 2D"),
         ("UWYK No-Anc", "UWYK 2D No-Anc"), ("UWYK Anc", "UWYK 2D Anc")]
COLS = [("2D", "{2}"), ("2D ρ=est", "{2} copula ρ=est"), ("1D ρ=-0.5", "{1} copula ρ=-0.5"), ("1D ρ=0", "{1}"),
        ("1D ρ=0.5", "{1} copula ρ=0.5"), ("1D ρ=1", "{1} copula ρ=1.0"), ("1D ρ=est", "{1} copula ρ=est")]
TESTED = [i for i, (c, _) in enumerate(COLS) if c.startswith("1D")]
WS = ("0.1", "0.25", "0.5")


def hochberg(p):
    """number of stars per test (0, 1, 2): Hochberg step-up at 0.1 and 0.05"""
    p = np.asarray(p, float)
    out = np.zeros(p.size, int)
    order = np.argsort(p)
    M = p.size
    for a in (0.1, 0.05):
        ks = [k for k in range(1, M + 1) if p[order[k - 1]] <= a / (M - k + 1)]
        if ks:
            out[order[:max(ks)]] += 1
    return out


rows = {}
for path in sys.argv[1:]:
    for r in csv.DictReader(open(path)):
        if r["variant"] == "twins":
            rows[(r["w"], r["setting"], r["model"])] = r


def cell(w, d, model, metric):
    r = rows.get((w, d, model))
    return (float(r[metric]), float(r[metric + "_se"])) if r else (np.nan, np.nan)


for metric in ("mse", "IS"):
    for avg in (False, True):
        lines, tests = [], {}
        for one, two in PAIRS:
            for w in (("avg",) if avg else WS):
                for d in ("d=10", "d=1"):
                    v = []
                    for _, t in COLS:
                        name = t.format(None, one, two)
                        if avg:
                            c = [cell(x, d, name, metric) for x in WS]
                            v.append((np.mean([a for a, _ in c]), np.mean([s for _, s in c])))
                        else:
                            v.append(cell(w, d, name, metric))
                    if all(np.isnan(a) for a, _ in v):
                        continue
                    lines.append((one, w, d, v))
                    for i in TESTED:
                        (m2, s2), (m1, s1) = v[0], v[i]
                        if not (np.isnan(m1) or np.isnan(m2)):
                            z = (m2 - m1) / np.sqrt(s2 ** 2 + s1 ** 2)
                            tests.setdefault(one, []).append(((one, w, d, i), 2 * norm.sf(abs(z)), m2 < m1))
        marks = {}
        for one, ts in tests.items():
            for (key, _, lower), k in zip(ts, hochberg([p for _, p, _ in ts])):
                marks[key] = ("+" if lower else "-") + "*" * k if k else ""
        print(f"\n{metric.upper()}, {'averaged over w = 0.1, 0.25, 0.5' if avg else 'per jitter w'}"
              "   (1D cells: + 2D lower, - 2D higher; * 0.1, ** 0.05, Hochberg; no mark = not significant)")
        print(f"{'model':12s} {'w':>4s} {'d':>4s} " + " ".join(f"{c:>12s}" for c, _ in COLS))
        for one, w, d, v in lines:
            print(f"{one:12s} {w:>4s} {d[2:]:>4s} " + " ".join(
                f"{a:9.3f}{marks.get((one, w, d, i), ''):3s}" for i, (a, _) in enumerate(v)))
