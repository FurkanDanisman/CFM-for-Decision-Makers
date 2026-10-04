"""Read one shard of harness dumps, keep only each query's 95% interval, so the dumps can be deleted.
python score_shard.py DUMP_DIR OUT_NPZ

Interval = hist: the exact tau distribution of the predicted histogram (hist_interval.py).
Saved per dataset (index within the shard, as the harness numbers them): lo, hi (n_queries,),
and the dump's true_cate (to check alignment with the truth files later).
"""
import glob, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../../UWYK_Fig3_4"))
from cate_density_metrics import _query_pmfs, interval_95
from hist_interval import interval_hist

dump_dir, out = sys.argv[1], sys.argv[2]
idx, lo, hi, y, flag = [], [], [], [], []
for f in sorted(glob.glob(os.path.join(dump_dir, "*r[0-9][0-9][0-9].npz"))):
    got = _query_pmfs(f)
    if got is None:                                            # no usable density (scorer's own gate)
        continue
    atoms, pmfs, y_dump = got
    uniform = np.allclose(np.diff(atoms), atoms[1] - atoms[0], rtol=1e-3)
    iv = np.array([interval_hist(atoms, p) if uniform else interval_95(atoms, p) for p in pmfs])
    idx.append(int(re.search(r"r(\d+)\.npz$", f).group(1)))
    lo.append(iv[:, 0]); hi.append(iv[:, 1]); y.append(y_dump); flag.append(uniform)
np.savez(out, idx=np.array(idx), lo=np.array(lo, dtype=np.float32), hi=np.array(hi, dtype=np.float32),
         y_dump=np.array(y, dtype=np.float32), hist=np.array(flag))
print(f"{out}: {len(idx)} datasets scored")
