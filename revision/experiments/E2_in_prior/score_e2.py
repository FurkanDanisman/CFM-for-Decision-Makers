"""Coverage of each model's central 95% interval against each truth.
python score_e2.py DATA_ROOT DUMP_ROOT [N_QUERIES]

One replication = one world: hit at each of its first N_QUERIES queries (default all), averaged
within the world. Coverage = mean over worlds; SE = sd over worlds / sqrt(n_worlds).
Two intervals from the same prediction:
  raw    the paper's scorer (cate_density_metrics.interval_95): diagonal mass as points
  hist   the exact tau distribution of the predicted histogram (hist_interval.py)
length = median over worlds of (interval length / sd of that world's observed outcomes).
"""
import glob, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../../UWYK_Fig3_4"))
from cate_density_metrics import _query_pmfs, interval_95
from hist_interval import interval_hist

data_root, dump_root = sys.argv[1], sys.argv[2]
NQ = int(sys.argv[3]) if len(sys.argv) > 3 else None
SHARD = 100
#        model          prior      truths scored
MODELS = [("dopfn_native", "dopfn", ["delta_shared", "delta_indep"]),
          ("dopfn_joint2d", "dopfn", ["delta_shared", "delta_indep"]),
          ("uwyk1d", "uwyk_1d", ["delta_shared", "delta_indep"]),
          ("graph2d", "uwyk_2d", ["delta_shared", "delta_indep"]),
          ("cpfn1d", "cpfn_1d", ["delta_shared", "delta_indep", "tau"]),
          ("cpfn2d", "cpfn_2d", ["delta_shared", "delta_indep"])]

print(f"queries per world: {NQ or 'all'}")
print(f"{'model':14s} {'truth':13s} {'worlds':>6s} | {'raw':>6s} {'SE':>5s} {'len':>6s} | {'hist':>6s} {'SE':>5s} {'len':>6s}")
for model, prior, truths in MODELS:
    hits = {(k, v): [] for k in truths for v in ("raw", "hist")}
    length = {"raw": [], "hist": []}
    dropped = nonuniform = 0
    for f in sorted(glob.glob(os.path.join(dump_root, model, "shard*", "*r[0-9][0-9][0-9].npz"))):
        shard = int(re.search(r"shard(\d+)", f).group(1))
        r = shard * SHARD + int(re.search(r"r(\d+)\.npz$", f).group(1))   # harness numbers worlds 0..99 per shard
        got = _query_pmfs(f)
        if got is None:
            dropped += 1
            continue
        atoms, pmfs, y_dump = got
        tr = np.load(os.path.join(data_root, prior, "truths", f"r{r:03d}.npz"))
        assert np.allclose(y_dump, tr["delta_shared"], rtol=1e-4, atol=1e-5), f"{f}: dump/truth mismatch"
        pmfs = pmfs[:NQ]
        iv = {"raw": np.array([interval_95(atoms, p) for p in pmfs])}
        if np.allclose(np.diff(atoms), atoms[1] - atoms[0], rtol=1e-3):
            iv["hist"] = np.array([interval_hist(atoms, p) for p in pmfs])
        else:
            nonuniform += 1
            iv["hist"] = iv["raw"]
        w = np.load(os.path.join(data_root, prior, f"shard{shard}", "complexmech/5node/path_TY/hide_0.0",
                                 f"r{r:03d}.npz"))
        sd = float(np.std(w["Y_train"]))
        for v, (lo, hi) in ((v, x.T) for v, x in iv.items()):
            length[v].append(np.mean(hi - lo) / sd)
            for k in truths:
                t = tr[k][:NQ]
                hits[(k, v)].append(np.mean((lo <= t) & (t <= hi)))
    if not hits[(truths[0], "raw")]:
        continue
    for k in truths:
        row = f"{model:14s} {k:13s} {len(hits[(k, 'raw')]):6d}"
        for v in ("raw", "hist"):
            c = np.array(hits[(k, v)])
            row += f" | {c.mean():6.3f} {c.std(ddof=1) / np.sqrt(c.size):5.3f} {np.median(length[v]):6.2f}"
        print(row)
    if dropped or nonuniform:
        print(f"{'':14s} ({dropped} worlds without a usable density; {nonuniform} on a non-uniform grid, hist = raw)")
