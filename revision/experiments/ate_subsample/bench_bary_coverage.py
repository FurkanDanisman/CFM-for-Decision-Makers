"""Coverage of the OT ATE density on the benchmarks: the barycenter as is (s = 1) and scaled by the
in-prior split-conformal s (conformal_bary.py, "s on all 500 datasets").
python bench_bary_coverage.py --root DIR [--match TOKEN] --s S [--label NAME] [--alpha 0.05] [--list]

Walks DIR for harness dumps (*r###.npz), keeps paths containing TOKEN, groups them by their directory
(one benchmark cell, e.g. a RealCause dataset or a case-study case/d/shift). Per dump: the barycenter of
the per-query ITE densities (as reduce_bary.py) centred at theta = mean of the per-query ITE means; the
truth is the mean of the dump's true_cate_per_query over the same queries. Interval: the central 1-alpha
interval of Q_s(u) = theta + s (Q(u) - theta). --list only counts the dumps per group.
"""
import argparse, glob, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../../UWYK_Fig3_4"))

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--match", default="")
ap.add_argument("--s", type=float, default=1.0); ap.add_argument("--label", default="")
ap.add_argument("--alpha", type=float, default=0.05); ap.add_argument("--list", action="store_true")
a = ap.parse_args()
files = [f for f in glob.glob(os.path.join(a.root, "**", "*r[0-9][0-9][0-9].npz"), recursive=True)
         if a.match in os.path.relpath(f, a.root)]
groups = {}
for f in sorted(files):
    groups.setdefault(os.path.relpath(os.path.dirname(f), a.root), []).append(f)
print(f"{a.label or a.root}: {len(files)} dumps in {len(groups)} groups (match '{a.match}'), s = {a.s:g}")
if a.list:
    for g, fs in groups.items():
        print(f"  {g}: {len(fs)}")
    sys.exit()

from cate_density_metrics import _query_pmfs
U = np.linspace(0.0025, 0.9975, 399)
lo_u, hi_u = a.alpha / 2, 1 - a.alpha / 2


def quantiles(atoms, p):
    p = np.clip(p, 0, None); p = p / p.sum()
    cdf = np.cumsum(p) - p / 2
    keep = np.r_[True, np.diff(cdf) > 0]
    return np.interp(U, cdf[keep], atoms[keep])


tot = {1: [], "s": []}
for g, fs in groups.items():
    c1, cs, w1 = [], [], []
    for f in fs:
        got = _query_pmfs(f)
        if got is None:
            continue
        atoms, pmfs, y = got
        ok = np.isfinite(pmfs).all(1) & (pmfs.sum(1) > 0)
        if not ok.any() or not np.isfinite(y).all():
            continue
        pm = pmfs[ok]; yt = y[ok[: y.size]] if y.size >= ok.size else y
        th = ((pm @ atoms) / pm.sum(1)).mean(); tr = yt.mean()
        Q = np.mean([quantiles(atoms, p) for p in pm], axis=0)
        ql, qh = np.interp(lo_u, U, Q), np.interp(hi_u, U, Q)
        c1.append(ql <= tr <= qh)
        cs.append(th + a.s * (ql - th) <= tr <= th + a.s * (qh - th)); w1.append(qh - ql)
    if not c1:
        print(f"  {g}: no usable dumps"); continue
    c1, cs, w1 = map(np.asarray, (c1, cs, w1))
    tot[1].append(c1); tot["s"].append(cs)
    print(f"  {g}: n={c1.size}  coverage s=1 {c1.mean():.3f}  s={a.s:g} {cs.mean():.3f}"
          f"   median width s=1 {np.median(w1):.4g}  s={a.s:g} {a.s * np.median(w1):.4g}")
if tot[1]:
    p1, ps = np.concatenate(tot[1]), np.concatenate(tot["s"])
    se = lambda c: np.sqrt(c.mean() * (1 - c.mean()) / c.size)
    print(f"  pooled: n={p1.size}  coverage s=1 {p1.mean():.3f} +- {se(p1):.3f}   s={a.s:g} {ps.mean():.3f} +- {se(ps):.3f}")
