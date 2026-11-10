"""Jittered Twins: 2D (conditioned on the known twin) against 1D conditioned through a Gaussian copula at
rho = -0.5, 0, 0.5, 1 and the estimated rho (2D also through the copula at it), MSE and IS, from the score_twins_deq CSVs.
python deq_table.py CSV [CSV ...]"""
import csv
import sys

PAIRS = [("Do-PFN", "Do-PFN 2D"), ("CausalPFN-C", "CausalPFN-C 2D"),
         ("UWYK No-Anc", "UWYK 2D No-Anc"), ("UWYK Anc", "UWYK 2D Anc")]
COLS = [("2D", "{2}"), ("2D ρ=est", "{2} copula ρ=est"), ("1D ρ=-0.5", "{1} copula ρ=-0.5"), ("1D ρ=0", "{1}"),
        ("1D ρ=0.5", "{1} copula ρ=0.5"), ("1D ρ=1", "{1} copula ρ=1.0"), ("1D ρ=est", "{1} copula ρ=est")]
rows = {}
for p in sys.argv[1:]:
    for r in csv.DictReader(open(p)):
        if r["variant"] == "twins":
            rows[(r["w"], r["setting"], r["model"])] = r
for metric in ("mse", "IS"):
    print(f"\n{metric.upper()}  (lowest per row marked <)")
    print(f"{'model':12s} {'w':>4s} {'d':>4s} " + " ".join(f"{c:>10s}" for c, _ in COLS))
    for one, two in PAIRS:
        for w in ("0.1", "0.25", "0.5"):
            for d in ("d=10", "d=1"):
                v = [rows.get((w, d, t.format(None, one, two))) for _, t in COLS]
                if all(x is None for x in v):
                    continue
                x = [float(r[metric]) if r else float("nan") for r in v]
                best = min((a for a in x if a == a), default=None)
                print(f"{one:12s} {w:>4s} {d[2:]:>4s} " + " ".join(
                    f"{a:9.3f}{'<' if a == best else ' '}" for a in x))
