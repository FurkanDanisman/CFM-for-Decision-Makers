"""OT ATE density on a benchmark, every variant of one model in one pass: coverage and interval score of the
barycenter as is (s = 1) and scaled by the variant's in-prior split-conformal s, per benchmark cell.
python bench_rho.py --root DUMPS --bench rc|cs|cm --model NAME [--tags noanc,v3a] [--couplings ...]
                    --inprior DIR --out CSV [--cmech-data DATA] [--alpha 0.05] [--workers N]

Variants: the model's tags x couplings, named as reduce_bary.py names its in-prior results (NAME[-TAG][-rho<r>]);
the variant's s is read from DIR/<variant>/coverage.txt ("s on all <N> datasets"). Couplings: indep (rho = 0)
or gauss:<rho>, the paper's Gaussian copula for single-outcome models; omit them for a 2D model (learned joint).
Cells (the columns of the paper's figures): rc the dataset, cs the case (pooled over d and shifts), cm the
number of nodes (ComplexMech's two query subsets joined per source realization, as bench_bary_coverage.py).
Per realization (one ATE): the barycenter of the per-query ITE densities, centred at theta = mean of the
per-query ITE means; the truth is the mean of the dumps' true_cate_per_query. Interval: the central 1-alpha
interval of Q_s(u) = theta + s (Q(u) - theta); interval score IS = width + (2/alpha) x the miss.
Appends one CSV row per variant x cell: model, variant, tag, rho, s, cell, n, cov1, is1, is1_se, covs, iss,
iss_se, width1 (median). Per-realization values go to <out>.<variant>.npz.
"""
import argparse, csv, glob, os, re, sys
import multiprocessing as mp
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "../../..")
sys.path.insert(0, os.path.join(REPO, "UWYK_Fig3_4")); sys.path.insert(0, os.path.join(REPO, "benchmarks"))
sys.path.insert(0, HERE)
from reduce_bary import quantiles, rho_label, U                  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--bench", required=True, choices=("rc", "cs", "cm"))
ap.add_argument("--model", required=True); ap.add_argument("--tags", default=""); ap.add_argument("--couplings", default="")
ap.add_argument("--inprior", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--cmech-data", default=None); ap.add_argument("--alpha", type=float, default=0.05)
ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
a = ap.parse_args()
tags = [t for t in a.tags.split(",") if t] or [None]
coups = [c for c in a.couplings.split(",") if c] or [None]
VARS = [(t, c) for t in tags for c in coups]
vname = lambda t, c: a.model + (f"-{t}" if t else "") + (f"-rho{rho_label(c)}" if c else "")


def s_of(name):
    try:
        txt = open(os.path.join(a.inprior, name, "coverage.txt")).read()
        return float(re.search(r"s on all \d+ datasets: ([0-9.]+)", txt).group(1))
    except (OSError, AttributeError):
        return None


S = {v: s_of(vname(*v)) for v in VARS}
for v, s in S.items():
    print(f"{vname(*v)}: s = {s}")

files = sorted(glob.glob(os.path.join(a.root, "**", "*r[0-9][0-9][0-9].npz"), recursive=True))
ridx = lambda f: int(re.search(r"r(\d+)\.npz$", f).group(1))
real = {}                                                         # (cell, realization key) -> [files]
need = {}
if a.bench == "cm":
    from uwyk_fig34_dataset import UWYKFig34Dataset
    maps = {}
for f in files:
    g = os.path.relpath(os.path.dirname(f), a.root)
    if a.bench == "cm":
        m = re.search(r"CMECH_n(\d+)_(nonzero|zero)$", g)
        if not m:
            continue
        if g not in maps:
            maps[g] = UWYKFig34Dataset(f"CMECH_n{m.group(1)}_{m.group(2)}", data_root=a.cmech_data).source_realizations
            for r in maps[g]:
                need.setdefault((f"n{m.group(1)}", r), set()).add(m.group(2))
        cell, key = f"n{m.group(1)}", maps[g][ridx(f)]
    else:
        cell, key = g.split(os.sep)[-1], (g, ridx(f))            # rc: dataset; cs: case (d, shift in the path)
    real.setdefault((cell, key), []).append(f)
if a.bench == "cm":                                               # a realization with a missing subset is dropped
    for k in [k for k, fs in real.items() if len(fs) < len(need.get(k, ()))]:
        print(f"skip {k}: {len(real[k])} of {len(need[k])} subset dumps"); del real[k]


def one(item):
    (cell, key), fs = item
    from cate_density_metrics import _query_pmfs
    out = {}
    for v in VARS:
        q, th, yt, ok_all = [], [], [], True
        for f in fs:
            got = _query_pmfs(f, v[0], v[1] or "indep")
            if got is None:
                ok_all = False; break
            atoms, pmfs, y = got
            ok = np.isfinite(pmfs).all(1) & (pmfs.sum(1) > 0) & np.isfinite(y)
            pm = pmfs[ok]
            q += [quantiles(atoms, p) for p in pm]; th += list((pm @ atoms) / pm.sum(1)); yt += list(y[ok])
        if not ok_all or not q:
            out[v] = None; continue
        Q = np.mean(q, axis=0); t = float(np.mean(th)); tr = float(np.mean(yt))
        ql, qh = np.interp(a.alpha / 2, U, Q), np.interp(1 - a.alpha / 2, U, Q)
        s = S[v] if S[v] is not None else np.nan
        sl, sh = t + s * (ql - t), t + s * (qh - t)
        isc = lambda l, h: (h - l) + 2 / a.alpha * (max(l - tr, 0.0) + max(tr - h, 0.0))
        out[v] = (ql <= tr <= qh, isc(ql, qh), sl <= tr <= sh, isc(sl, sh), qh - ql, t, tr)
    return cell, out


items = sorted(real.items(), key=lambda kv: str(kv[0]))
# fork: the workers inherit the parsed arguments and variant table (this script has no __main__ guard)
res = list(mp.get_context("fork").Pool(a.workers).imap(one, items, chunksize=8)) if a.workers > 1 else [one(i) for i in items]
new = not os.path.exists(a.out)
with open(a.out, "a", newline="") as fh:
    w = csv.writer(fh)
    if new:
        w.writerow(["model", "variant", "tag", "rho", "s", "cell", "n", "cov1", "is1", "is1_se", "covs", "iss", "iss_se", "width1"])
    for v in VARS:
        name = vname(*v)
        rec = [(c, o[v]) for c, o in res if o[v] is not None]
        cells = sorted({c for c, _ in rec})
        np.savez(f"{a.out}.{name}.npz", cell=np.array([c for c, _ in rec]),
                 **{k: np.array([r[j] for _, r in rec], dtype=float)
                    for j, k in enumerate(("cov1", "is1", "covs", "iss", "width1", "theta", "truth"))})
        for c in cells:
            R = np.array([r for cc, r in rec if cc == c], dtype=float)
            n = len(R); se = lambda x: x.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
            rho = "learned" if v[1] is None else ("0" if v[1] == "indep" else v[1].split(":", 1)[1])
            w.writerow([a.model, name, v[0] or "", rho, S[v], c, n, R[:, 0].mean(), R[:, 1].mean(), se(R[:, 1]),
                        R[:, 2].mean(), R[:, 3].mean(), se(R[:, 3]), np.median(R[:, 4])])
            print(f"  {name} [{c}] n={n}  cov s=1 {R[:, 0].mean():.3f}  s={S[v]} {R[:, 2].mean():.3f}"
                  f"   IS s=1 {R[:, 1].mean():.4g} +- {se(R[:, 1]):.2g}   s {R[:, 3].mean():.4g} +- {se(R[:, 3]):.2g}")
