"""Scale-calibrated OT ATE density: split conformal over in-prior datasets.
python conformal_bary.py RES_DIR [--alpha 0.05] [--folds 5]

RES_DIR holds shard*.npz from reduce_bary.py. The scaled density keeps the barycenter's centre theta and
shape and changes its width: Q_s(u) = theta + s (Q(u) - theta). Its central 1-alpha interval is
[theta + s (Q_lo - theta), theta + s (Q_hi - theta)]. Per dataset the smallest covering s is
  r = max((theta - truth) / (theta - Q_lo), (truth - theta) / (Q_hi - theta)).
s is the ceil((1-alpha)(N+1))-th smallest r over the N calibration datasets (split conformal). Reported:
the barycenter as is (s = 1), and s calibrated on the other folds and tested on each held-out fold;
coverage, width and the interval score (mean and median over datasets; outcome scale).
"""
import argparse, glob, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("res"); ap.add_argument("--alpha", type=float, default=0.05); ap.add_argument("--folds", type=int, default=5)
a = ap.parse_args()
zs = [np.load(f) for f in sorted(glob.glob(os.path.join(a.res, "shard*.npz")))]
th = np.concatenate([z["theta"] for z in zs]); tr = np.concatenate([z["truth"] for z in zs])
Q = np.concatenate([z["Q"] for z in zs]); U = zs[0]["U"]
qlo = np.array([np.interp(a.alpha / 2, U, q) for q in Q]); qhi = np.array([np.interp(1 - a.alpha / 2, U, q) for q in Q])
eps = 1e-12
dl, dh = np.maximum(th - qlo, eps), np.maximum(qhi - th, eps)
r = np.maximum((th - tr) / dl, (tr - th) / dh)
w1 = qhi - qlo
N = r.size
se = lambda c: np.sqrt(c.mean() * (1 - c.mean()) / c.size)


def s_of(rc):
    k = int(np.ceil((1 - a.alpha) * (rc.size + 1)))
    return np.inf if k > rc.size else np.sort(rc)[k - 1]


perm = np.random.default_rng(0).permutation(N)
cov, wid, ss = np.zeros(N, bool), np.zeros(N), []
for f in np.array_split(perm, a.folds):
    cal = np.setdiff1d(perm, f); s = s_of(r[cal]); ss.append(s)
    cov[f] = r[f] <= s; wid[f] = s * w1[f]
c1 = r <= 1
isc = lambda lo, hi, y: (hi - lo) + 2 / a.alpha * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))
is1 = isc(qlo, qhi, tr)
iss = np.zeros(N)
for f, s in zip(np.array_split(perm, a.folds), ss):
    iss[f] = isc(th[f] - s * dl[f], th[f] + s * dh[f], tr[f])
print(f"{os.path.basename(a.res.rstrip('/'))}: {N} datasets, {int(100 * (1 - a.alpha))}% intervals")
print(f"  barycenter (s = 1):   coverage {c1.mean():.3f} +- {se(c1):.3f}   mean width {w1.mean():.4g}")
print(f"  calibrated s, {a.folds}-fold: coverage {cov.mean():.3f} +- {se(cov):.3f}   mean width {wid.mean():.4g}"
      f"   s per fold {', '.join(f'{s:.3f}' for s in ss)}")
print(f"  s on all {N} datasets: {s_of(r):.3f}   (width relative to the barycenter = s)")
print(f"  interval score ({int(100 * (1 - a.alpha))}%): barycenter mean {is1.mean():.4g} median {np.median(is1):.4g}"
      f"   calibrated (held-out folds) mean {iss.mean():.4g} median {np.median(iss):.4g}")
