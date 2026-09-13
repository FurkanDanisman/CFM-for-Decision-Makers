"""Compare models in a combined_*_density.csv: pooled, by d, and by N.

    python case_study/density_eval/compare_density.py <csv> [--models a b]
"""
from __future__ import annotations
import argparse, csv, math
from collections import defaultdict
from statistics import mean

M = ["cate_cov95", "cate_len95", "cate_wis", "cate_crps",
     "ate_cov95", "ate_len95", "ate_wis", "ate_crps", "ate_bias"]
SHORT = {"cate_cov95": "C.cov%", "cate_len95": "C.len", "cate_wis": "C.wis",
         "cate_crps": "C.crps", "ate_cov95": "A.cov%", "ate_len95": "A.len",
         "ate_wis": "A.wis", "ate_crps": "A.crps", "ate_bias": "A.bias"}


def _f(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _avg(vals):
    v = [x for x in vals if x is not None]
    return mean(v) if v else float('nan')


def _fmt(metric, v):
    return f"{100 * v:10.1f}" if metric.endswith("cov95") else f"{v:10.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--models", nargs="*", default=None)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv)))
    if a.models:
        rows = [r for r in rows if r["model"] in a.models]
    if not rows:
        raise SystemExit("no rows")
    models = sorted({r["model"] for r in rows})
    print(f"{len(rows)} rows   models: {', '.join(models)}")
    print(f"   n per row: {rows[0].get('n')}   "
          f"(realizations x shifts pooled)\n")

    print("=== pooled over all d, N, cases ===")
    print(f"{'model':16s}" + "".join(f"{SHORT[m]:>10s}" for m in M))
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for m in M:
            agg[r["model"]][m].append(_f(r.get(m)))
    for mod in models:
        print(f"{mod:16s}" + "".join(_fmt(m, _avg(agg[mod][m])) for m in M))

    for key, label in (("d", "d"), ("N", "N")):
        print(f"\n=== CATE by {label}:  coverage% / length / CRPS ===")
        hdr = f"{label:>5s} " + "".join(f"{mod:^33s}" for mod in models)
        print(hdr)
        print(f"{'':>5s} " + "".join(f"{'cov%':>11s}{'len':>11s}{'crps':>11s}"
                                     for _ in models))
        buck = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for r in rows:
            for m in ("cate_cov95", "cate_len95", "cate_crps"):
                buck[int(r[key])][r["model"]][m].append(_f(r.get(m)))
        for k in sorted(buck):
            line = f"{k:5d} "
            for mod in models:
                c = buck[k].get(mod)
                if not c:
                    line += " " * 33
                    continue
                line += (f"{100 * _avg(c['cate_cov95']):11.1f}"
                         f"{_avg(c['cate_len95']):11.3f}"
                         f"{_avg(c['cate_crps']):11.4f}")
            print(line)

    if len(models) == 2:
        a_, b_ = models
        print(f"\n=== {b_} relative to {a_} (ratio; >1 means {b_} is larger) ===")
        for m in ("cate_len95", "cate_wis", "cate_crps",
                  "ate_len95", "ate_wis", "ate_crps"):
            va, vb = _avg(agg[a_][m]), _avg(agg[b_][m])
            print(f"  {SHORT[m]:>8s}  {va:9.4f} -> {vb:9.4f}   "
                  f"ratio {vb / va if va else float('nan'):.3f}")
        for m in ("cate_cov95", "ate_cov95"):
            print(f"  {SHORT[m]:>8s}  {100*_avg(agg[a_][m]):8.1f}% -> "
                  f"{100*_avg(agg[b_][m]):8.1f}%")


if __name__ == "__main__":
    main()
