"""Coverage of the OT ATE density on the benchmarks: the barycenter as is (s = 1) and scaled by the
in-prior split-conformal s (conformal_bary.py, "s on all 500 datasets").
python bench_bary_coverage.py --root DIR --s S [--label NAME] [--alpha 0.05] [--pool K] [--quiet]
                              [--cmech-data DATA_ROOT] [--tag TAG] [--list]

Walks DIR for harness dumps (*r###.npz) and groups them by directory (one benchmark cell). Per
realization: the barycenter of the per-query ITE densities (each query's quantile function on its own
atoms, averaged), centred at theta = mean of the per-query ITE means; the truth is the mean of the dumps'
true_cate_per_query over the same queries. Interval: the central 1-alpha interval of
Q_s(u) = theta + s (Q(u) - theta).
--cmech-data: ComplexMech dumps come in two directories per cell, CMECH_n<k>_nonzero and _zero, each
  numbering its realizations among those that have queries of that subset. Each dump index is mapped back
  to its source realization (benchmarks/uwyk_fig34_dataset.py, the loader the harnesses used), and the
  two subsets of a realization are joined, so the ATE is over all of its queries (as --subset total).
--pool K: also pool the groups by their K-th path component (e.g. the case-study case).
"""
import argparse, glob, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "../../..")
sys.path.insert(0, os.path.join(REPO, "UWYK_Fig3_4")); sys.path.insert(0, os.path.join(REPO, "benchmarks"))

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--s", type=float, default=1.0)
ap.add_argument("--label", default=""); ap.add_argument("--alpha", type=float, default=0.05)
ap.add_argument("--pool", type=int, default=None); ap.add_argument("--quiet", action="store_true")
ap.add_argument("--cmech-data", default=None); ap.add_argument("--list", action="store_true")
ap.add_argument("--tag", default=None, help="density tag of multi-mode dumps (UWYK: noanc | v3a)")
a = ap.parse_args()
files = sorted(glob.glob(os.path.join(a.root, "**", "*r[0-9][0-9][0-9].npz"), recursive=True))
ridx = lambda f: int(re.search(r"r(\d+)\.npz$", f).group(1))

# groups: {group: {realization key: [files]}}
groups = {}
for f in files:
    g = os.path.relpath(os.path.dirname(f), a.root)
    key = ridx(f)
    if a.cmech_data:
        m = re.search(r"CMECH_n(\d+)_(nonzero|zero)$", g)
        if not m:
            continue
        from uwyk_fig34_dataset import UWYKFig34Dataset
        cache = groups.setdefault("_map", {})
        if g not in cache:
            cache[g] = UWYKFig34Dataset(f"CMECH_n{m.group(1)}_{m.group(2)}", data_root=a.cmech_data).source_realizations
        key = cache[g][key]
        g = g[: m.start()] + f"CMECH_n{m.group(1)}"
    groups.setdefault(g, {}).setdefault(key, []).append(f)
maps = groups.pop("_map", {})
if a.cmech_data:                                                # drop realizations with a subset missing
    need = {}
    for g, src in maps.items():
        m = re.search(r"CMECH_n(\d+)_(nonzero|zero)$", g)
        for r in src:
            need.setdefault((g[: m.start()] + f"CMECH_n{m.group(1)}", r), set()).add(m.group(2))
    for g, rs in groups.items():
        for r in [r for r, fs in rs.items() if len(fs) < len(need.get((g, r), ()))]:
            print(f"  skip {g} realization {r}: {len(rs[r])} of {len(need[(g, r)])} subset dumps")
            del rs[r]
print(f"{a.label or a.root}: {len(files)} dumps, {sum(len(v) for v in groups.values())} realizations "
      f"in {len(groups)} groups, s = {a.s:g}")
if a.list:
    for g, rs in groups.items():
        print(f"  {g}: {len(rs)} realizations, files per realization {sorted({len(v) for v in rs.values()})}")
    sys.exit()

from cate_density_metrics import _query_pmfs
U = np.linspace(0.0025, 0.9975, 399)
lo_u, hi_u = a.alpha / 2, 1 - a.alpha / 2


def quantiles(atoms, p):
    p = np.clip(p, 0, None); p = p / p.sum()
    cdf = np.cumsum(p) - p / 2
    keep = np.r_[True, np.diff(cdf) > 0]
    return np.interp(U, cdf[keep], atoms[keep])


def realization(fs):
    q, th, yt = [], [], []
    for f in fs:
        got = _query_pmfs(f, a.tag)
        if got is None:
            return None                                         # a missing part would change the query set
        atoms, pmfs, y = got
        ok = np.isfinite(pmfs).all(1) & (pmfs.sum(1) > 0) & np.isfinite(y)
        pm = pmfs[ok]
        q += [quantiles(atoms, p) for p in pm]; th += list((pm @ atoms) / pm.sum(1)); yt += list(y[ok])
    if not q:
        return None
    Q = np.mean(q, axis=0); t = float(np.mean(th)); tr = float(np.mean(yt))
    ql, qh = np.interp(lo_u, U, Q), np.interp(hi_u, U, Q)
    return ql <= tr <= qh, t + a.s * (ql - t) <= tr <= t + a.s * (qh - t), qh - ql


se = lambda c: np.sqrt(c.mean() * (1 - c.mean()) / c.size)
pools, allr = {}, []
for g in sorted(groups):
    res = [r for r in (realization(fs) for fs in groups[g].values()) if r is not None]
    if not res:
        print(f"  {g}: no usable dumps"); continue
    c1, cs, w = map(np.asarray, zip(*res))
    allr.append((c1, cs))
    if a.pool is not None:
        pools.setdefault(g.split(os.sep)[a.pool], []).append((c1, cs))
    if not a.quiet:
        print(f"  {g}: n={c1.size}  coverage s=1 {c1.mean():.3f}  s={a.s:g} {cs.mean():.3f}"
              f"   median width s=1 {np.median(w):.4g}  s={a.s:g} {a.s * np.median(w):.4g}")
for k, v in sorted(pools.items()):
    c1, cs = np.concatenate([x[0] for x in v]), np.concatenate([x[1] for x in v])
    print(f"  [{k}] n={c1.size}  coverage s=1 {c1.mean():.3f} +- {se(c1):.3f}   s={a.s:g} {cs.mean():.3f} +- {se(cs):.3f}")
if allr:
    c1, cs = np.concatenate([x[0] for x in allr]), np.concatenate([x[1] for x in allr])
    print(f"  pooled: n={c1.size}  coverage s=1 {c1.mean():.3f} +- {se(c1):.3f}   s={a.s:g} {cs.mean():.3f} +- {se(cs):.3f}")
