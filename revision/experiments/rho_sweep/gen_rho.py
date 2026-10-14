"""rho-sweep data: one simple SCM, cross-world noise correlation rho controlled.
python gen_rho.py OUT_ROOT                       (numpy only, CPU, seconds)

    X ~ N(0, I_5)
    T | X ~ Bernoulli(sigmoid(0.5 * X1))
    mu0(x) = x1 + 0.5 x2,   mu1(x) = mu0(x) + 1 + 0.5 x3
    Y(t) = mu_t(X) + eps_t,  (eps0, eps1) bivariate normal, unit variances, corr rho
    observed Y = Y(T)

Per rho: 100 datasets, each N_CTX = 1000 observational context units and N_Q = 100 query units
with both potential outcomes. Written with E2's writer (E2_in_prior/common.write_world), i.e. the
ComplexMech npz format the paper's harnesses read:

    OUT_ROOT/rho<+r.rr>/shard0/complexmech/5node/path_TY/hide_0.0/r###.npz   (harness input)
        X_train (1000,5)  T_train (1000,)  Y_train (1000,)  X_test (100,5)
        true_cate (100,) = realized delta = Y1 - Y0   n_real_features = 5
    OUT_ROOT/rho<+r.rr>/truths/r###.npz
        delta_shared = Y1 - Y0 (realized ITE, the coverage target), cate = mu1 - mu0 (PEHE target),
        y0, y1, t_test (the query units' own treatment draw; not given to any model)

The format carries no graph. The known graph (X1..X5 -> T, X1..X3 -> Y, T -> Y) is supplied by the
UWYK harnesses' v3a matrix (T->Y, every X->T, every X->Y = +1, rest 0), which is the true ancestral
relation here: X4, X5 are ancestors of Y through T.

Seeding: dataset r uses np.random.default_rng([SEED, r]) for every rho, so the same uniforms and
normals are drawn for every rho, so the datasets differ across rho ONLY in eps1's coupling to eps0
(X, T and eps0 are identical across rho for a given r).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "E2_in_prior"))
from common import write_world, done                          # noqa: E402

RHOS = (-0.5, 0.0, 0.5, 0.9, 1.0)
N_DATASETS, N_CTX, N_Q, D = 100, 1000, 100, 5
SEED = 20261013


def rho_dir(rho):
    return f"rho{rho:+.2f}"


def draw(rng, n, rho):
    X = rng.standard_normal((n, D))
    T = (rng.random(n) < 1.0 / (1.0 + np.exp(-0.5 * X[:, 0]))).astype(np.float64)
    z0, z1 = rng.standard_normal(n), rng.standard_normal(n)
    e0, e1 = z0, rho * z0 + np.sqrt(max(1.0 - rho * rho, 0.0)) * z1
    mu0 = X[:, 0] + 0.5 * X[:, 1]
    mu1 = mu0 + 1.0 + 0.5 * X[:, 2]
    return X, T, mu0, mu1, mu0 + e0, mu1 + e1


def main(out_root):
    for rho in RHOS:
        root = os.path.join(out_root, rho_dir(rho))
        for r in range(N_DATASETS):
            if done(root, r):
                continue
            rng = np.random.default_rng([SEED, r])            # same draws for every rho (see docstring)
            X, T, _, _, y0, y1 = draw(rng, N_CTX, rho)
            Xq, Tq, mu0q, mu1q, y0q, y1q = draw(rng, N_Q, rho)
            Y = np.where(T == 1.0, y1, y0)
            write_world(root, r, X, T, Y, Xq,
                        {"delta_shared": y1q - y0q, "cate": mu1q - mu0q,
                         "y0": y0q, "y1": y1q, "t_test": Tq})
        d = np.load(os.path.join(root, "truths", f"r{N_DATASETS - 1:03d}.npz"))
        print(f"{rho_dir(rho)}: {N_DATASETS} datasets in {root}  "
              f"(last: sd(delta - cate) = {np.std(d['delta_shared'] - d['cate']):.3f}, "
              f"theory sqrt(2 - 2 rho) = {np.sqrt(2 - 2 * rho):.3f})", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
