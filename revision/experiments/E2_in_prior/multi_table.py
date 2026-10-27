"""Per-SCM in-prior calibration (multi.sh): 100 worlds per prior, 1000 observational sets each.
python multi_table.py ROOT NW

Per world: coverage = mean over its 1000 datasets of the share of the 500 queries whose truth falls in the
hist 95% interval (as e2_tables.py), against both truths; length = mean over datasets of mean interval length
/ sd of the context outcomes. Across worlds: mean (SE over worlds), sd, min, max, and the number of worlds with
coverage within 0.02 of 0.95.
"""
import glob, os, re, sys
import numpy as np

root, NW = sys.argv[1], int(sys.argv[2])
SHARD = 100
MODELS = [("dopfn_native", "Do-PFN", "dopfn"), ("dopfn_joint2d_262k", "Do-PFN 2D", "dopfn"),
          ("uwyk1d", "UWYK (binary T)", "uwyk_1d"), ("uwyk1d_cont", "UWYK (continuous T)", "uwyk_1dc"),
          ("graph2d", "UWYK 2D", "uwyk_2d"), ("cpfn1d", "CausalPFN-C", "cpfn_1d"), ("cpfn2d", "CausalPFN-C 2D", "cpfn_2d")]
TRUTHS = ("delta_shared", "delta_indep")
sd_cache = {}


def world(model, prior, w):
    """coverage per truth and length of one world, or None if incomplete"""
    files = glob.glob(os.path.join(root, "results", f"w{w}", model, "shard*.npz"))
    if len(files) < 10:
        return None
    hit = {t: [] for t in TRUTHS}; L = []
    for f in files:
        k = int(re.search(r"shard(\d+)\.npz$", f).group(1))
        z = np.load(f)
        for i, lo, hi, y in zip(z["idx"], z["lo"], z["hi"], z["y_dump"]):
            r = k * SHARD + int(i)
            tr = np.load(os.path.join(root, "data", f"w{w}", prior, "truths", f"r{r:03d}.npz"))
            assert np.allclose(y, tr["delta_shared"], rtol=1e-4, atol=1e-5), f"{model} w{w} r{r}: dump/truth mismatch"
            for t in TRUTHS:
                hit[t].append(np.mean((lo <= tr[t]) & (tr[t] <= hi)))
            key = (prior, w, r)
            if key not in sd_cache:
                d = os.path.join(root, "data", f"w{w}", prior, f"shard{k}", "complexmech/5node/path_TY/hide_0.0", f"r{r:03d}.npz")
                sd_cache[key] = float(np.std(np.load(d)["Y_train"]))
            L.append(np.mean(hi - lo) / sd_cache[key])
    if len(L) != 1000:
        return None
    return {t: float(np.mean(hit[t])) for t in TRUTHS}, float(np.mean(L))


print(f"In-prior calibration per SCM: {NW} worlds per prior, 1000 observational sets x 500 queries each.")
print("coverage per world = mean over its sets; across worlds: mean (SE), sd, [min, max], worlds with |c - 0.95| <= 0.02\n")
print(f"{'model':20s} {'truth':13s} {'worlds':>6s} {'coverage':>15s} {'sd':>6s} {'range':>15s} {'within 0.02':>12s} {'length':>13s}")
for model, name, prior in MODELS:
    got = [world(model, prior, w) for w in range(NW)]
    got = [g for g in got if g is not None]
    if not got:
        print(f"{name:20s} (no complete world)"); continue
    Ls = np.array([g[1] for g in got])
    for t in TRUTHS:
        c = np.array([g[0][t] for g in got]); n = len(c)
        inside = int(np.sum(np.abs(c - 0.95) <= 0.02))
        print(f"{name:20s} {t:13s} {n:6d} {c.mean():7.3f} ({c.std(ddof=1) / np.sqrt(n):.3f}) {c.std(ddof=1):6.3f} "
              f"[{c.min():.3f}, {c.max():.3f}] {inside:5d}/{n:<6d} {Ls.mean():6.2f} ({Ls.std(ddof=1) / np.sqrt(n):.2f})")
