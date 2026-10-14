"""Which ComplexMech realizations satisfy Bodik, Huang & Yu (2026)'s assumption Y(T) = f_Y(T, X, eps), eps ⟂ T, X?

A realization fails when some covariate is a consequence of the treatment or the outcome:
  - a MEDIATOR: a descendant of T that is an ancestor of Y (T -> ... -> X_k -> ... -> Y);
  - a DESCENDANT OF Y (Y -> ... -> X_k).
Covariates upstream of T and Y, or descendants of T that do not lead to Y, are fine.

Graph: each r*.npz stores adj_matrix in the order [T, Y, kept covariates], A[i, j] = 1 for an edge i -> j
(UWYK SCM.get_adjacency_matrix; UWYK_Fig3_4/generate_pehe_benchmark.py _finalize). A realization whose SCM
has nodes outside that order (n_nodes > 2 + n_real_features) is reported as `incomplete`, since a path
through a missing node cannot be seen.

    python count_pretreatment.py $SCRATCH/cmech_data_v2          (numpy only, seconds)
"""
import glob
import os
import sys

import numpy as np


def reach(A):
    """R[i, j] = 1 iff there is a directed path i -> ... -> j of length >= 1."""
    n = A.shape[0]
    R = (A != 0).astype(np.int64)
    for _ in range(n):
        R2 = ((R + R @ R) > 0).astype(np.int64)
        if (R2 == R).all():
            break
        R = R2
    return R.astype(bool)


def classify(f):
    z = np.load(f, allow_pickle=True)
    A = np.asarray(z["adj_matrix"], float)
    n_feat = int(z["n_real_features"])
    complete = int(z["n_nodes"]) == 2 + n_feat
    A = A[:2 + n_feat, :2 + n_feat]
    R = reach(A)
    T, Y = 0, 1
    ks = range(2, 2 + n_feat)
    mediators = [k for k in ks if R[T, k] and R[k, Y]]
    y_desc = [k for k in ks if R[Y, k]]
    return complete, len(mediators), len(y_desc)


def main(root):
    for nd in sorted(glob.glob(os.path.join(root, "complexmech", "*node", "path_TY", "hide_0.0")),
                     key=lambda p: int(os.path.basename(os.path.dirname(os.path.dirname(p))).replace("node", ""))):
        files = sorted(glob.glob(os.path.join(nd, "r*.npz")))
        ok = med = ydesc = both = incomplete = 0
        for f in files:
            complete, m, y = classify(f)
            incomplete += not complete
            if m and y:
                both += 1
            elif m:
                med += 1
            elif y:
                ydesc += 1
            elif complete:
                ok += 1
        name = os.path.basename(os.path.dirname(os.path.dirname(nd)))
        print(f"{name:7s} n={len(files):3d}  satisfies={ok:3d}  mediator only={med:3d}  "
              f"Y-descendant only={ydesc:3d}  both={both:3d}  incomplete graph={incomplete:3d}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
