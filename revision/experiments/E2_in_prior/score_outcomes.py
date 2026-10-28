"""Outcome-level coverage of one shard of Do-PFN dumps (run before the dumps are deleted).
python score_outcomes.py DUMP_DIR TRUTH_Y_DIR SHARD OUT_NPZ

1D dump (p_y0_scaled, p_y1_scaled): per query, the 95% interval of each outcome's predicted distribution (2.5% and
  97.5% quantiles, on bin edges, as the paper's hist read-out) and whether the true y0, resp. y1, falls in it.
2D dump (p_joint_scaled): per query, the 95% region of the joint: the fewest grid cells whose predicted mass reaches
  0.95 (all cells if the grid holds less); covered if the true pair (y0, y1) lies in one of them. A true pair
  outside the grid is not covered, and is counted.
Saves per dataset (idx): cov_y0, cov_y1 (1D) or cov_joint, outside (2D), each a mean over the dataset's queries.
"""
import glob, os, re, sys
import numpy as np

A = 0.05
dump_dir, truth_dir, shard, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]


def interval(p, e):
    F = np.concatenate([[0.0], np.cumsum(p / p.sum())])
    lo = int(np.clip(np.searchsorted(F, A / 2, side="left") - 1, 0, p.size - 1))
    hi = int(np.clip(np.searchsorted(F, 1 - A / 2, side="left") - 1, 0, p.size - 1))
    return e[lo], e[hi + 1]


res = {}
for f in sorted(glob.glob(os.path.join(dump_dir, "*r[0-9][0-9][0-9].npz"))):
    i = int(re.search(r"r(\d+)\.npz$", f).group(1))
    t = np.load(os.path.join(truth_dir, f"r{shard * 100 + i:03d}.npz"))
    y0, y1 = t["y0"].astype(np.float64), t["y1"].astype(np.float64)
    with np.load(f, allow_pickle=True) as z:
        e = np.asarray(z["edges"], np.float64).reshape(-1)
        if "p_joint_scaled" in z.files:
            J = np.asarray(z["p_joint_scaled"], np.float64)               # (q, K, K): axis 1 = y0, axis 2 = y1
            K = J.shape[-1]
            i0 = np.searchsorted(e, y0, side="right") - 1; i1 = np.searchsorted(e, y1, side="right") - 1
            inside = (i0 >= 0) & (i0 < K) & (i1 >= 0) & (i1 < K)
            cov = np.zeros(J.shape[0])
            for q in range(J.shape[0]):
                if not inside[q]:
                    continue
                flat = J[q].ravel(); order = np.argsort(-flat)
                k = int(np.searchsorted(np.cumsum(flat[order]), 1 - A, side="left")) + 1
                cov[q] = float(i0[q] * K + i1[q] in set(order[:min(k, flat.size)].tolist()))
            res[i] = dict(cov_joint=cov.mean(), outside=(~inside).mean())
        else:
            p0, p1 = np.asarray(z["p_y0_scaled"], np.float64), np.asarray(z["p_y1_scaled"], np.float64)
            c0, c1 = [], []
            for q in range(p0.shape[0]):
                L, U = interval(p0[q], e); c0.append(L <= y0[q] <= U)
                L, U = interval(p1[q], e); c1.append(L <= y1[q] <= U)
            res[i] = dict(cov_y0=np.mean(c0), cov_y1=np.mean(c1))
keys = sorted(res)
np.savez(out, idx=np.array(keys), **{k: np.array([res[i][k] for i in keys]) for k in res[keys[0]]})
print(f"{out}: {len(keys)} datasets")
