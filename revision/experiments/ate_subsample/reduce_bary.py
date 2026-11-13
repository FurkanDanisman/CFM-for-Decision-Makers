"""Reduce one shard of harness dumps to the OT (2-Wasserstein barycenter) ATE density of each dataset.
python reduce_bary.py DUMP_DIR OUT_NPZ [TAG]   (TAG: density tag of multi-mode dumps, UWYK noanc | v3a)

Per dump (one dataset): the per-query ITE densities (cate_density_metrics._query_pmfs, as score_shard.py),
each turned into a quantile function on the grid U (CDF linear between atoms, mass of an atom centred on
it); the 1D barycenter is the average of these quantile functions (Section 3.2, Eq. ate-quantile).
Saved: idx, theta (mean of the per-query ITE means = the barycenter's centre), truth (mean of the dump's
true_cate_per_query), U, Q (n_datasets, len(U)) the barycenter quantiles.
"""
import glob, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../../UWYK_Fig3_4"))
from cate_density_metrics import _query_pmfs

U = np.linspace(0.0025, 0.9975, 399)


def quantiles(atoms, p):
    p = np.clip(p, 0, None); p = p / p.sum()
    cdf = np.cumsum(p) - p / 2                                  # mass of each atom centred on it
    keep = np.r_[True, np.diff(cdf) > 0]                        # strictly increasing for interp
    return np.interp(U, cdf[keep], atoms[keep])


dump_dir, out = sys.argv[1], sys.argv[2]
tag = sys.argv[3] if len(sys.argv) > 3 else None
idx, theta, truth, Q = [], [], [], []
for f in sorted(glob.glob(os.path.join(dump_dir, "*r[0-9][0-9][0-9].npz"))):
    got = _query_pmfs(f, tag)
    if got is None:
        continue
    atoms, pmfs, y = got
    ok = np.isfinite(pmfs).all(1) & (pmfs.sum(1) > 0)
    if not ok.any() or not np.isfinite(y).all():
        continue
    pm = pmfs[ok]
    tau_hat = (pm @ atoms) / pm.sum(1)
    q = np.mean([quantiles(atoms, p) for p in pm], axis=0)
    idx.append(int(re.search(r"r(\d+)\.npz$", f).group(1)))
    theta.append(tau_hat.mean()); truth.append(y.mean()); Q.append(q)
if not idx:
    sys.exit(f"{dump_dir}: no usable dumps")
np.savez(out, idx=np.array(idx), theta=np.array(theta), truth=np.array(truth), U=U, Q=np.array(Q))
print(f"{out}: {len(idx)} datasets")
