"""Same SCM, fixed query units: coverage per query unit over the 1000 observational sets.
python fixq_table.py DATA_ROOT RESULTS_ROOT
Truths as in the new-world table (tab:inprior-new): independent noise for the single-outcome models, shared noise
for the 2D models and for CausalPFN-C. Per model: per-set coverage (mean over the 500 queries; mean and number of
sets in [0.93, 0.97]) and per-query coverage (mean over the 1000 sets; how many queries are covered in every set,
in [0.93, 0.97], and in under half of the sets)."""
import glob, os, re, sys
import numpy as np

data_root, res_root = sys.argv[1:3]
MODELS = [("dopfn_native", "Do-PFN", "dopfn", "delta_indep"), ("dopfn_joint2d_262k", "Do-PFN 2D", "dopfn", "delta_shared"),
          ("uwyk1d", "UWYK (binary T)", "uwyk_1d", "delta_indep"), ("uwyk1d_cont", "UWYK (continuous T)", "uwyk_1dc", "delta_indep"),
          ("graph2d", "UWYK 2D", "uwyk_2d", "delta_shared"), ("cpfn1d", "CausalPFN-C", "cpfn_1d", "delta_shared"),
          ("cpfn2d", "CausalPFN-C 2D", "cpfn_2d", "delta_shared")]
print("per-query coverage (same 500 query units in every set; truths as in tab:inprior-new)")
print(f"{'model':20s} {'truth':12s} {'per set':>7s} {'sets in band':>12s} {'per query':>9s} {'q always':>8s} "
      f"{'q in band':>9s} {'q < 0.5':>7s}")
for model, name, prior, truth in MODELS:
    hits = {}
    for f in glob.glob(os.path.join(res_root, model, "shard*.npz")):
        k = int(re.search(r"shard(\d+)\.npz$", f).group(1))
        z = np.load(f)
        for i, lo, hi in zip(z["idx"], z["lo"], z["hi"]):
            r = k * 100 + int(i)
            y = np.load(os.path.join(data_root, prior, "truths", f"r{r:03d}.npz"))[truth]
            hits[r] = (lo <= y) & (y <= hi)
    if len(hits) < 1000:
        print(f"{name:20s} ({len(hits)}/1000 sets)"); continue
    H = np.array([hits[r] for r in sorted(hits)], dtype=float)  # (sets, queries)
    cs, cq = H.mean(1), H.mean(0)
    band = lambda c: int(((c >= .93) & (c <= .97)).sum())
    print(f"{name:20s} {truth:12s} {cs.mean():7.3f} {band(cs):7d}/1000 {cq.mean():9.3f} {int((cq >= .999).sum()):5d}/{cq.size}"
          f" {band(cq):5d}/{cq.size} {int((cq < .5).sum()):4d}/{cq.size}")
