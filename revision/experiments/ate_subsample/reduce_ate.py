"""Reduce one run of harness dumps to per-dataset ATE estimates, so the dumps can be deleted.
python reduce_ate.py DUMP_DIR OUT_NPZ

Per dump (one dataset): the per-query ITE means tau_hat_i = sum(atoms * pmf_i) of the predicted ITE
density (the same density score_shard.py reads, via cate_density_metrics._query_pmfs), the ATE estimate
theta = mean_i tau_hat_i, and the truth = mean_i of the dump's true_cate_per_query.
Saved: idx, theta, truth, tau_hat (n_datasets, n_queries), y_true (n_datasets, n_queries).
"""
import glob, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../../UWYK_Fig3_4"))
from cate_density_metrics import _query_pmfs

dump_dir, out = sys.argv[1], sys.argv[2]
idx, theta, truth, th, yt = [], [], [], [], []
for f in sorted(glob.glob(os.path.join(dump_dir, "*r[0-9][0-9][0-9].npz"))):
    got = _query_pmfs(f)
    if got is None:                                            # no usable density (scorer's own gate)
        continue
    atoms, pmfs, y = got
    tau_hat = (pmfs @ atoms) / pmfs.sum(1)
    if not (np.isfinite(tau_hat).all() and np.isfinite(y).all()):
        continue
    idx.append(int(re.search(r"r(\d+)\.npz$", f).group(1)))
    theta.append(tau_hat.mean()); truth.append(y.mean()); th.append(tau_hat); yt.append(y)
if not idx:
    sys.exit(f"{dump_dir}: no usable dumps")
np.savez(out, idx=np.array(idx), theta=np.array(theta), truth=np.array(truth),
         tau_hat=np.array(th, dtype=np.float32), y_true=np.array(yt, dtype=np.float32))
print(f"{out}: {len(idx)} datasets")
