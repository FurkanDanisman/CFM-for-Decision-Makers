"""Jittered Twins, MSE: 2D against its 1D baselines evaluated over the range of rho compatible with the observed data
(score_twins_deq.py ... range: per w, RANGES[w]), per covariate set d; averaged over w = 0.1, 0.25, 0.5.
python twins_range_test.py CSV [CSV ...]   (the *_range CSVs of the 1D baselines and the 2D CSVs)

Two summaries of a 1D baseline per w: the average of its MSEs over the range ("avg of scores") and the MSE of its
prediction averaged over the range ("avg prediction", the row "copula ρ=avg"). Over w: the mean of the per-w values,
its SE bounded by the mean of the per-w SEs (for "avg of scores" the per-w SE is bounded the same way by the mean SE
over the range); both bounds hold under any dependence across w and rho. Significance (the paper's test, Appendix F):
2D against each baseline, two-sided z-test with SE = sqrt(SE_2D^2 + SE_1D^2); Hochberg within each model and summary,
over its baselines (full resolution, matched resolution, matched label) and d; * at 0.1, ** at 0.05.
"""
import csv
import math
import sys

WS = ("0.1", "0.25", "0.5")
RANGES = {"0.1": (-4, 8), "0.25": (-7, 9), "0.5": (-9, 10)}   # = score_twins_deq.RANGES (tenths)
FAM = [("Do-PFN", "Do-PFN 2D"), ("UWYK No-Anc", "UWYK 2D No-Anc"), ("UWYK Anc", "UWYK 2D Anc"),
       ("CausalPFN-C", "CausalPFN-C 2D")]
BASE = [("full resolution", ""), ("matched resolution", " (matched resolution)"), ("matched label", " (matched label)")]
mean = lambda x: sum(x) / len(x)

rows = {}
for path in sys.argv[1:]:
    for r in csv.DictReader(open(path)):
        if r["variant"] == "twins":
            rows[(r["w"], r["setting"], r["model"])] = (float(r["mse"]), float(r["mse_se"]))


def over_w(per_w):
    """per_w: {w: (mean, se)} -> (mean over w, mean SE), None if a w is missing"""
    if any(per_w.get(w) is None for w in WS):
        return None
    return mean([per_w[w][0] for w in WS]), mean([per_w[w][1] for w in WS])


def hochberg(p):
    M = len(p); out = [0] * M; order = sorted(range(M), key=lambda i: p[i])
    for a in (0.1, 0.05):
        ks = [k for k in range(1, M + 1) if p[order[k - 1]] <= a / (M - k + 1)]
        for i in (order[:max(ks)] if ks else []):
            out[i] += 1
    return out


for summary in ("avg of scores",):                             # the averaged prediction is not reported
    print(f"\n== 1D summarised as: {summary} over the observed-data range of rho; MSE averaged over w")
    for one, two in FAM:
        tests = []
        for d in ("d=10", "d=1"):
            t = over_w({w: rows.get((w, d, two)) for w in WS})
            for blab, suf in BASE:
                per_w = {}
                for w in WS:
                    if summary == "avg prediction":
                        per_w[w] = rows.get((w, d, f"{one}{suf} copula ρ=avg"))
                    else:
                        lo, hi = RANGES[w]
                        g = [rows.get((w, d, f"{one}{suf} copula ρ={k / 10}")) for k in range(lo, hi + 1)]
                        per_w[w] = None if any(x is None for x in g) else (mean([m for m, _ in g]), mean([s for _, s in g]))
                a = over_w(per_w)
                if t is None or a is None:
                    print(f"  missing: {one}{suf} {d}"); continue
                p = math.erfc(abs(t[0] - a[0]) / math.hypot(t[1], a[1]) / math.sqrt(2))
                tests.append((d, blab, t, a, p))
        for (d, blab, t, a, p), k in zip(tests, hochberg([x[-1] for x in tests])):
            v = ("2D lower " if t[0] < a[0] else "2D higher ") + "*" * k if k else "not significant"
            print(f"  {one:12s} {d:5s} vs 1D {blab:19s} 2D {t[0]:.3f} ± {t[1]:.3f}   1D {a[0]:.3f} ± {a[1]:.3f}"
                  f"   p {p:.2g}   {v}")
