"""E2 tables from the per-shard intervals.
python e2_tables.py DATA_ROOT RESULTS_ROOT R_LIST

One replication = one dataset: its first Q queries are scored against each truth and the hits
averaged within the dataset. Coverage = mean over the first R datasets; SE = sd / sqrt(R).
  new-world run : datasets are worlds    -> R_LIST e.g. 1000,10000
  same-world run: datasets of one world  -> R_LIST e.g. 1000
Q = 10, 100, 500. length = median over datasets of (mean interval length / sd of context outcomes).
"""
import glob, os, re, sys
import numpy as np

data_root, res_root = sys.argv[1], sys.argv[2]
R_LIST = [int(x) for x in sys.argv[3].split(",")]
QS, SHARD = (10, 100, 500), 100
#        model          prior      truths
MODELS = [("dopfn_native", "dopfn", ["delta_shared", "delta_indep"]),
          ("dopfn_joint2d", "dopfn", ["delta_shared", "delta_indep"]),
          ("uwyk1d", "uwyk_1d", ["delta_shared", "delta_indep"]),
          ("graph2d", "uwyk_2d", ["delta_shared", "delta_indep"]),
          ("cpfn1d", "cpfn_1d", ["delta_shared", "delta_indep", "tau"]),
          ("cpfn2d", "cpfn_2d", ["delta_shared", "delta_indep"])]

cols = [(R, Q) for R in R_LIST for Q in QS]
print("coverage (SE) of the hist 95% interval;  columns = datasets R x queries Q")
print(f"{'model':14s} {'truth':13s} " + " ".join(f"{f'R={R} Q={Q}':>15s}" for R, Q in cols) + f" {'length':>7s}")
for model, prior, truths in MODELS:
    rows = {}                                                  # global dataset index -> (lo, hi)
    for f in glob.glob(os.path.join(res_root, model, "shard*.npz")):
        k = int(re.search(r"shard(\d+)\.npz$", f).group(1))
        z = np.load(f)
        for i, lo, hi, y in zip(z["idx"], z["lo"], z["hi"], z["y_dump"]):
            r = k * SHARD + int(i)
            tr = np.load(os.path.join(data_root, prior, "truths", f"r{r:03d}.npz"))
            assert np.allclose(y, tr["delta_shared"], rtol=1e-4, atol=1e-5), f"{model} r={r}: dump/truth mismatch"
            w = np.load(os.path.join(data_root, prior, f"shard{k}", "complexmech/5node/path_TY/hide_0.0", f"r{r:03d}.npz"))
            rows[r] = (lo, hi, {t: tr[t] for t in truths}, float(np.std(w["Y_train"])))
    if not rows:
        continue
    order = sorted(rows)
    for t in truths:
        line = f"{model:14s} {t:13s} "
        for R, Q in cols:
            use = order[:R]
            c = np.array([np.mean((rows[r][0][:Q] <= rows[r][2][t][:Q]) & (rows[r][2][t][:Q] <= rows[r][1][:Q]))
                          for r in use])
            cell = f"{c.mean():.3f} ({c.std(ddof=1) / np.sqrt(c.size):.3f})" if len(use) == R else "n/a"
            line += f"{cell:>15s} "
        length = np.median([np.mean(rows[r][1] - rows[r][0]) / rows[r][3] for r in order])
        print(line + f"{length:7.2f}")
    print(f"{'':14s} ({len(order)} datasets scored)")
