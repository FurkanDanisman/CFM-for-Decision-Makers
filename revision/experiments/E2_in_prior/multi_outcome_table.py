"""Per-SCM outcome-level coverage (multi.sh eval_y): Do-PFN per outcome (y0, y1), Do-PFN 2D for the joint.
python multi_outcome_table.py ROOT NW OUT_CSV
Per world: mean over its 1000 datasets of the per-dataset coverage (score_outcomes.py). Across worlds: mean (SE),
sd, [min, max], and the number of worlds with coverage in [0.93, 0.97], in (0.97, 1) and exactly 1."""
import glob, os, sys
import numpy as np

root, NW, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
SPEC = [("dopfn_native", "Do-PFN", ["cov_y0", "cov_y1"]), ("dopfn_joint2d_262k", "Do-PFN 2D", ["cov_joint"])]
NAME = {"cov_y0": "y0", "cov_y1": "y1", "cov_joint": "joint"}
rows = []
print(f"{'model':10s} {'target':6s} {'worlds':>6s} {'coverage':>15s} {'sd':>6s} {'range':>15s} {'[0.93,0.97]':>11s} {'(0.97,1)':>9s} {'=1':>5s}")
for model, name, keys in SPEC:
    per = {k: [] for k in keys}; outside = []
    for w in range(NW):
        fs = sorted(glob.glob(os.path.join(root, "results_y", f"w{w}", model, "outcomes_shard*.npz")))
        if len(fs) < 10:
            continue
        zs = [np.load(f) for f in fs]
        if sum(z["idx"].size for z in zs) != 1000:
            continue
        for k in keys:
            c = float(np.concatenate([z[k] for z in zs]).mean()); per[k].append(c); rows.append((name, NAME[k], w, c))
        if "outside" in zs[0].files:
            outside.append(float(np.concatenate([z["outside"] for z in zs]).mean()))
    for k in keys:
        c = np.array(per[k]); n = c.size
        if not n:
            print(f"{name:10s} {NAME[k]:6s} (no complete world)"); continue
        one = c >= 1 - 1e-9
        print(f"{name:10s} {NAME[k]:6s} {n:6d} {c.mean():7.3f} ({c.std(ddof=1) / np.sqrt(n):.3f}) {c.std(ddof=1):6.3f} "
              f"[{c.min():.3f}, {c.max():.3f}] {int(((c >= 0.93) & (c <= 0.97)).sum()):11d} "
              f"{int(((c > 0.97) & ~one).sum()):9d} {int(one.sum()):5d}")
    if outside:
        print(f"{'':10s} true pair outside the 2D grid: {100 * np.mean(outside):.2f}% of query units (counted as not covered)")
with open(out, "w") as fh:
    fh.write("model,truth,world,coverage,length\n")
    fh.writelines(f"{m},{t},{w},{c:.6f},nan\n" for m, t, w, c in rows)
print("per-world values:", out)
