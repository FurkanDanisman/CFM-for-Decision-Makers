"""Predicted cross-world correlation of a 2D head, per query, from one shard of harness dumps.
python pred_rho_shard.py DUMP_DIR OUT_NPZ

For each query, the head's predicted joint over the interior K x K grid (p_joint_scaled, (n_q, K, K),
rows = Y(0) bin, columns = Y(1) bin; the 8 tail regions are not in the dump, as for the paper's
intervals) is read as a piecewise-uniform density: within a cell the two outcomes are uniform and
independent. Then
    E[a] = sum_i p_i. c_i,   Var(a) = sum_i p_i. c_i^2 - E[a]^2 + w^2/12   (same for b),
    Cov(a, b) = sum_ij p_ij c_i c_j - E[a] E[b],     rho_pred = Cov / sqrt(Var(a) Var(b)),
with c the cell centres and w the cell width from the dumped `edges` (uniform interior grid).
The correlation is unchanged by the harness's affine y-scaling, so the scaled grid is used as is.
Saved per dataset (index as the harness numbers it): rho (n_q,), and the dump's true_cate for alignment.
"""
import glob
import os
import re
import sys

import numpy as np


def rho_from_joint(P, centres, w):
    P = np.asarray(P, dtype=np.float64)
    P = P / P.sum(axis=(-2, -1), keepdims=True)
    pa, pb = P.sum(axis=-1), P.sum(axis=-2)                     # marginals of Y(0), Y(1)
    ma, mb = pa @ centres, pb @ centres
    va = pa @ centres ** 2 - ma ** 2 + w * w / 12.0
    vb = pb @ centres ** 2 - mb ** 2 + w * w / 12.0
    cov = np.einsum("...ij,i,j->...", P, centres, centres) - ma * mb
    return cov / np.sqrt(va * vb)


def main(dump_dir, out):
    idx, rho, y = [], [], []
    for f in sorted(glob.glob(os.path.join(dump_dir, "*r[0-9][0-9][0-9].npz"))):
        z = np.load(f, allow_pickle=True)
        if "p_joint_scaled" not in z.files:
            continue
        P = np.asarray(z["p_joint_scaled"], dtype=np.float64)
        if P.ndim != 3:
            continue
        K = P.shape[-1]
        e = np.asarray(z["edges"], dtype=np.float64).reshape(-1) if "edges" in z.files else None
        if e is not None and e.size == K + 1:
            w = float(np.mean(np.diff(e)))
            assert np.allclose(np.diff(e), w, rtol=1e-3), f"{f}: non-uniform interior grid"
            centres = 0.5 * (e[:-1] + e[1:])
        else:                                                   # the scorer's [-1, 1] convention
            w = 2.0 / K
            centres = -1.0 + w * (np.arange(K) + 0.5)
        idx.append(int(re.search(r"r(\d+)\.npz$", f).group(1)))
        rho.append(rho_from_joint(P, centres, w).astype(np.float32))
        y.append(np.asarray(z["true_cate_per_query"] if "true_cate_per_query" in z.files
                            else np.full(P.shape[0], np.nan), dtype=np.float32).reshape(-1))
    np.savez(out, idx=np.array(idx), rho=np.array(rho, dtype=object), y_dump=np.array(y, dtype=object))
    print(f"{out}: {len(idx)} datasets", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
