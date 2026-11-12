"""Coverage of the subsampling ATE interval.
python ate_coverage.py RES_DIR [--m 500] [--n 1000]

RES_DIR holds b_full.npz (full context) and b_<k>.npz (context subsample k of m out of n rows, without
replacement), each from reduce_ate.py. Per dataset: theta_hat from the full context, theta_b from the
subsamples. The spread of a subsample estimate around the full one is rescaled to the full-context size
by sqrt(m / (n - m)) (variance of a mean over m of n rows without replacement is s^2 (1/m - 1/n);
at the full context it is s^2 / n), which is 1 at m = n / 2.
Intervals at 95%:
  normal     theta_hat +- 1.96 * k * sd(theta_b)
  percentile theta_hat - k * (q_{0.975} - mean, q_{0.025} - mean) of theta_b (basic bootstrap form)
Truth: mean over the queries of the dump's true_cate_per_query.
"""
import argparse, glob, os, re
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("res"); ap.add_argument("--m", type=int, default=500); ap.add_argument("--n", type=int, default=1000)
a = ap.parse_args()
full = np.load(os.path.join(a.res, "b_full.npz"))
subs = [np.load(f) for f in sorted(glob.glob(os.path.join(a.res, "b_[0-9]*.npz")),
                                   key=lambda f: int(re.search(r"b_(\d+)", f).group(1)))]
k = np.sqrt(a.m / (a.n - a.m))
z = 1.959963984540054
rows = []
for j, r in enumerate(full["idx"]):
    tb = [s["theta"][np.flatnonzero(s["idx"] == r)[0]] for s in subs if r in s["idx"]]
    if len(tb) < 2:
        continue
    tb = np.asarray(tb); th, tr = full["theta"][j], full["truth"][j]
    sd = k * tb.std(ddof=1)
    c = tb.mean()
    lo_p, hi_p = th - k * (np.quantile(tb, 0.975) - c), th - k * (np.quantile(tb, 0.025) - c)
    rows.append((abs(th - tr) <= z * sd, 2 * z * sd, lo_p <= tr <= hi_p, hi_p - lo_p, len(tb)))
if not rows:
    raise SystemExit("no dataset has >= 2 subsamples")
cn, wn, cp, wp, nb = map(np.asarray, zip(*rows))
se = lambda c: np.sqrt(c.mean() * (1 - c.mean()) / c.size)
print(f"{os.path.basename(a.res.rstrip('/'))}: {cn.size} datasets, subsamples per dataset {nb.min()}-{nb.max()}, m={a.m} of n={a.n}")
print(f"  normal     95%: coverage {cn.mean():.3f} +- {se(cn):.3f}   mean width {wn.mean():.4g}")
print(f"  percentile 95%: coverage {cp.mean():.3f} +- {se(cp):.3f}   mean width {wp.mean():.4g}")
