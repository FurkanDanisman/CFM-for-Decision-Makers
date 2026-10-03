"""Coverage of each model's raw central 95% interval against each truth.
python score_e2.py DATA_ROOT DUMP_ROOT

One replication = one world: hit at each of its 100 queries, averaged within the world.
Coverage = mean over worlds; SE = sd over worlds / sqrt(n_worlds).
Intervals come from the paper's scorer (UWYK_Fig3_4/cate_density_metrics.py), unchanged.
"""
import glob, os, re, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../UWYK_Fig3_4"))
from cate_density_metrics import _query_pmfs, interval_95

data_root, dump_root = sys.argv[1], sys.argv[2]
SHARD = 100
#        model          prior      truths scored
MODELS = [("dopfn_native", "dopfn", ["delta_shared", "delta_indep"]),
          ("dopfn_joint2d", "dopfn", ["delta_shared", "delta_indep"]),
          ("uwyk1d", "uwyk_1d", ["delta_shared", "delta_indep"]),
          ("graph2d", "uwyk_2d", ["delta_shared", "delta_indep"]),
          ("cpfn1d", "cpfn_1d", ["delta_shared", "delta_indep", "tau"]),
          ("cpfn2d", "cpfn_2d", ["delta_shared", "delta_indep"])]

print(f"{'model':14s} {'truth':13s} {'worlds':>6s} {'coverage':>9s} {'SE':>6s} {'|c-.95|':>8s} {'length':>9s}")
for model, prior, truths in MODELS:
    per_world = {k: [] for k in truths}
    length, dropped = [], 0
    for f in sorted(glob.glob(os.path.join(dump_root, model, "shard*", "*r[0-9][0-9][0-9].npz"))):
        shard = int(re.search(r"shard(\d+)", f).group(1))
        r = shard * SHARD + int(re.search(r"r(\d+)\.npz$", f).group(1))   # harness numbers worlds 0..99 per shard
        got = _query_pmfs(f)
        if got is None:
            dropped += 1
            continue
        atoms, pmfs, y_dump = got
        lo, hi = np.array([interval_95(atoms, p) for p in pmfs]).T
        tr = np.load(os.path.join(data_root, prior, "truths", f"r{r:03d}.npz"))
        assert np.allclose(y_dump, tr["delta_shared"], rtol=1e-4, atol=1e-5), f"{f}: dump/truth mismatch"
        for k in truths:
            per_world[k].append(np.mean((lo <= tr[k]) & (tr[k] <= hi)))
        length.append(np.mean(hi - lo))
    for k in truths:
        c = np.array(per_world[k])
        if c.size == 0:
            continue
        se = c.std(ddof=1) / np.sqrt(c.size)
        print(f"{model:14s} {k:13s} {c.size:6d} {c.mean():9.3f} {se:6.3f} {abs(c.mean() - .95):8.3f} {np.mean(length):9.3g}")
    if dropped:
        print(f"{'':14s} ({dropped} worlds without a usable density)")
