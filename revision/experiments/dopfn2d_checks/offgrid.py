"""Do-PFN 2D on ComplexMech: how often is a query unit's true outcome outside the model's grid?
python offgrid.py DUMP_ROOT DATA_ROOT        (e.g. $SCRATCH/cmech_dumps/dopfn_repro_joint2d $SCRATCH/cmech_data_v2)

The dump keeps only the J x J grid cells (tails dropped, cells renormalised), so an outcome outside
[edges[0], edges[-1]] can never be inside the predicted support. For each node count: share of query
units with Y(0) or Y(1) off the grid, and hist coverage of Delta for on-grid vs off-grid units.
Dumps are matched to data files through the dataset's own _paths (as benchmarks/cmech_coverage_by_rho.py).
"""
import glob, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "../../..")
sys.path[:0] = [os.path.join(REPO, "benchmarks"), os.path.join(REPO, "UWYK_Fig3_4"),
                os.path.join(REPO, "revision/experiments/E2_in_prior")]
from uwyk_fig34_dataset import UWYKFig34Dataset
from cate_density_metrics import _query_pmfs
from hist_interval import interval_hist

root, data_root = sys.argv[1], sys.argv[2]
print(f"{'n':>3s} {'queries':>8s} {'off-grid':>9s} {'cov all':>8s} {'cov on':>8s} {'cov off':>8s}")
for n in (5, 10, 20, 30, 40, 50):
    off, hit = [], []
    for sub in ("nonzero", "zero"):
        d = os.path.join(root, "N1000", "dopfn_native", f"CMECH_n{n}_{sub}")
        if not os.path.isdir(d):
            continue
        ds = UWYKFig34Dataset(f"CMECH_n{n}_{sub}", data_root=data_root)
        for f in sorted(glob.glob(os.path.join(d, "r[0-9]*.npz"))):
            idx = int(re.search(r"r(\d+)", os.path.basename(f)).group(1))
            got = _query_pmfs(f)
            if got is None or idx >= len(ds._paths):
                continue
            atoms, pmfs, y = got
            with np.load(f) as z:
                lo, hi = float(z["edges"][0]), float(z["edges"][-1])
            with np.load(ds._paths[idx]) as z:
                m = ds._mask_for(np.asarray(z["true_cate"]).ravel())
                y0, y1 = np.asarray(z["Y_do0"]).ravel()[m], np.asarray(z["Y_do1"]).ravel()[m]
            assert np.allclose(y1 - y0, y, atol=1e-4), f"{f}: data/dump mismatch"
            iv = np.array([interval_hist(atoms, p) for p in pmfs])
            off.append((y0 < lo) | (y0 > hi) | (y1 < lo) | (y1 > hi))
            hit.append((iv[:, 0] <= y) & (y <= iv[:, 1]))
    if not off:
        continue
    o, h = np.concatenate(off), np.concatenate(hit)
    f = lambda v: f"{v.mean():.3f}" if v.size else "—"
    print(f"{n:3d} {o.size:8d} {o.mean():9.3f} {f(h):>8s} {f(h[~o]):>8s} {f(h[o]):>8s}")
