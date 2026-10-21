"""Write Bodik's datasets in the ComplexMech npz format the paper's harnesses read (DATASET=CMECH_n5).
python make_worlds.py OUT_ROOT

Inputs, next to this script (exported by export_bodik.R from his repository, unchanged):
  bodik_ihdp.npz   IHDP covariates X (747 x 25), treatment T, and for d in (10, 1), rho in (1, .75, .5, .25, 0),
                   rep in 1..50: Y0, Y1, mu0, mu1                       -> 500 datasets
  bodik_twins.npz  Twins: X (11984 x 71), T, Y0, Y1 (one-year mortality)  -> one dataset

IHDP, two ways of presenting each dataset to a model (the methods see the first d covariates, as in his script):
  ihdp_a/d{d}/rho{r}  held-out queries: a fixed 90/10 split per dataset, 672 context units and 75 queries
  ihdp_b/d{d}/rho{r}  in-sample, as in his evaluation: all 747 units are the context and all 747 are queried
Twins, his covariate sets d = 71, 10, 1. 11984 units exceed the context the models take, so the context is a
  fixed random sample of 1000 units (treated share kept), and every other unit is queried, in blocks of 1000
  (dataset r = block r, all with the same context):
  twins/d{d}          the first d covariates
  twins/d71_u50       the first 50 covariates, for the UWYK models (which take at most 50)
Truths (truths/r###.npz): delta_shared = Y1 - Y0 (the realized ITE), cate = mu1 - mu0 on IHDP and the realized
  ITE on Twins (no other truth exists there), y0, y1, t_test.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "E2_in_prior"))
from common import write_world, done                       # noqa: E402

SEED = 20261020
N_Q_A = 75                                                  # 747 - 75 = 672 context units (the npci IHDP split size)
TW_CTX, TW_BLOCK = 1000, 1000


def tag(r):
    return f"rho{r:g}"


def ihdp(out):
    z = np.load(os.path.join(HERE, "bodik_ihdp.npz"))
    X, T = z["X"].astype(np.float64), z["T"].astype(np.float64)
    n = X.shape[0]
    for i, d in enumerate(z["d"]):
        for j, rho in enumerate(z["rho"]):
            for mode in ("a", "b"):
                root = os.path.join(out, f"ihdp_{mode}", f"d{int(d)}", tag(float(rho)))
                for k in range(z["Y0"].shape[2]):
                    if done(root, k):
                        continue
                    y0, y1 = z["Y0"][i, j, k].astype(np.float64), z["Y1"][i, j, k].astype(np.float64)
                    mu0, mu1 = z["mu0"][i, j, k].astype(np.float64), z["mu1"][i, j, k].astype(np.float64)
                    yf = np.where(T == 1, y1, y0)
                    if mode == "a":
                        perm = np.random.default_rng([SEED, int(d), int(round(rho * 100)), k]).permutation(n)
                        q, c = perm[:N_Q_A], perm[N_Q_A:]
                    else:
                        q = c = np.arange(n)
                    Xd = X[:, :int(d)]
                    write_world(root, k, Xd[c], T[c], yf[c], Xd[q],
                                {"delta_shared": (y1 - y0)[q], "cate": (mu1 - mu0)[q], "y0": y0[q], "y1": y1[q],
                                 "t_test": T[q]})
                print(f"{root}: {z['Y0'].shape[2]} datasets", flush=True)


def twins(out):
    z = np.load(os.path.join(HERE, "bodik_twins.npz"))
    X, T = z["X"].astype(np.float64), z["T"].astype(np.float64)
    y0, y1 = z["Y0"].astype(np.float64), z["Y1"].astype(np.float64)
    yf = np.where(T == 1, y1, y0)
    rng = np.random.default_rng([SEED, 3])
    it, ic = np.flatnonzero(T == 1), np.flatnonzero(T == 0)
    n_t = int(round(TW_CTX * it.size / T.size))
    ctx = np.concatenate([rng.choice(it, n_t, replace=False), rng.choice(ic, TW_CTX - n_t, replace=False)])
    rng.shuffle(ctx)
    rest = np.setdiff1d(np.arange(T.size), ctx)
    rest = rest[rng.permutation(rest.size)]
    blocks = [rest[s:s + TW_BLOCK] for s in range(0, rest.size, TW_BLOCK)]
    for name, d in (("d71", 71), ("d10", 10), ("d1", 1), ("d71_u50", 50)):
        root = os.path.join(out, "twins", name)
        Xd = X[:, :d]
        for k, q in enumerate(blocks):
            if done(root, k):
                continue
            delta = (y1 - y0)[q]
            write_world(root, k, Xd[ctx], T[ctx], yf[ctx], Xd[q],
                        {"delta_shared": delta, "cate": delta, "y0": y0[q], "y1": y1[q], "t_test": T[q]})
        print(f"{root}: {len(blocks)} query blocks, context {ctx.size} ({n_t} treated), {rest.size} queries",
              flush=True)


def twins_bodik(out, d=10, n_sub=10, block=2000):
    """Twins in Bodik's setup: every family queried (11,984), d = 10.
      twins_full/d10   context = ALL families (the full data, as he fits on); queries in blocks of `block`
      twins_sub/d10    context = a random 1000 families (treated share kept), n_sub draws; dataset r = draw * n_blocks
                       + block, every draw querying every family
    """
    z = np.load(os.path.join(HERE, "bodik_twins.npz"))
    X, T = z["X"].astype(np.float64)[:, :d], z["T"].astype(np.float64)
    y0, y1 = z["Y0"].astype(np.float64), z["Y1"].astype(np.float64)
    yf = np.where(T == 1, y1, y0)
    allq = np.arange(T.size)
    blocks = [allq[s:s + block] for s in range(0, T.size, block)]
    tr = lambda q: {"delta_shared": (y1 - y0)[q], "cate": (y1 - y0)[q], "y0": y0[q], "y1": y1[q], "t_test": T[q]}
    root = os.path.join(out, "twins_full", f"d{d}")
    for k, q in enumerate(blocks):
        if not done(root, k):
            write_world(root, k, X, T, yf, X[q], tr(q))
    print(f"{root}: context {T.size} (all), {len(blocks)} query blocks", flush=True)
    root = os.path.join(out, "twins_sub", f"d{d}")
    it, ic = np.flatnonzero(T == 1), np.flatnonzero(T == 0)
    n_t = int(round(TW_CTX * it.size / T.size))
    for s_ in range(n_sub):
        rng = np.random.default_rng([SEED, 4, s_])
        ctx = np.concatenate([rng.choice(it, n_t, replace=False), rng.choice(ic, TW_CTX - n_t, replace=False)])
        rng.shuffle(ctx)
        for b, q in enumerate(blocks):
            k = s_ * len(blocks) + b
            if not done(root, k):
                write_world(root, k, X[ctx], T[ctx], yf[ctx], X[q], tr(q))
    print(f"{root}: {n_sub} context draws of {TW_CTX} x {len(blocks)} query blocks", flush=True)


if __name__ == "__main__":
    out = sys.argv[1]
    if "--twins-bodik" in sys.argv:
        for d_ in (10, 1):
            twins_bodik(out, d=d_)
    else:
        ihdp(out)
        twins(out)
