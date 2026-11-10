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


DEQ_W = (0.1, 0.25, 0.5)


def _write_bodik(out, kind, tag, X, T, y0, y1, y_ctx, extra, n_sub=10, block=2000):
    """twins_bodik's two layouts: <kind>_full_<tag> (context = all families) and <kind>_sub_<tag> (1000 families,
    n_sub draws); every family queried; truths y0, y1 (the outcomes the models are scored on) plus `extra`
    (per-family arrays, e.g. the binary outcomes); the context carries y_ctx."""
    allq = np.arange(T.size)
    blocks = [allq[s:s + block] for s in range(0, T.size, block)]
    tr = lambda q: {"delta_shared": (y1 - y0)[q], "cate": (y1 - y0)[q], "y0": y0[q], "y1": y1[q], "t_test": T[q],
                    **{k: v[q] for k, v in extra.items()}}
    d = X.shape[1]
    root = os.path.join(out, f"{kind}_full_{tag}", f"d{d}")
    for k, q in enumerate(blocks):
        if not done(root, k):
            write_world(root, k, X, T, y_ctx, X[q], tr(q))
    root = os.path.join(out, f"{kind}_sub_{tag}", f"d{d}")
    it, ic = np.flatnonzero(T == 1), np.flatnonzero(T == 0)
    n_t = int(round(TW_CTX * it.size / T.size))
    for s_ in range(n_sub):
        rng = np.random.default_rng([SEED, 4, s_])                    # the same context draws as twins_bodik
        ctx = np.concatenate([rng.choice(it, n_t, replace=False), rng.choice(ic, TW_CTX - n_t, replace=False)])
        rng.shuffle(ctx)
        for b, q in enumerate(blocks):
            k = s_ * len(blocks) + b
            if not done(root, k):
                write_world(root, k, X[ctx], T[ctx], y_ctx[ctx], X[q], tr(q))
    print(f"{kind} {tag} d{d}: full + {n_sub} draws of {TW_CTX}, {len(blocks)} query blocks", flush=True)


def _corr_jitter(X, a0, a1, w, seed):
    """Uniform(-w, w) jitter for both twins of a family, correlated (Gaussian copula) so that the cross-world
    correlation of the jittered pair, estimated as in tab:rho-all (benchmarks/rho/rho_cf.py: residuals of a 5-fold
    cross-fitted gradient boosting on X), equals that of the binary pair. Returns (u0, u1, copula corr, target rho)."""
    from scipy.optimize import brentq
    from scipy.stats import norm
    sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "benchmarks", "rho"))
    from rho_cf import _oof
    r0, r1 = a0 - _oof(X, a0, 5, 0), a1 - _oof(X, a1, 5, 0)
    target = np.corrcoef(r0, r1)[0, 1]
    g = np.random.default_rng(seed)
    e0, e1 = g.standard_normal(a0.size), g.standard_normal(a0.size)

    def jit(c):
        return w * (2 * norm.cdf(e0) - 1), w * (2 * norm.cdf(c * e0 + np.sqrt(1 - c * c) * e1) - 1)

    def gap(c):
        u0, u1 = jit(c)
        return np.corrcoef(r0 + u0, r1 + u1)[0, 1] - target
    c = brentq(gap, -0.999, 0.999) if gap(-0.999) < 0 < gap(0.999) else 0.0
    u0, u1 = jit(c)
    return u0, u1, c, target


def twins_deq(out, d):
    """Dequantized Twins: both twins' binary outcomes get a Uniform(-w, w) jitter (w <= 1/2, so the threshold 1/2
    recovers the binary outcome), correlated across the twins so that the cross-world correlation given the d
    covariates stays that of the binary data (_corr_jitter). The context carries the factual twin's jittered
    outcome; the truths are the jittered pair (y0, y1), with the binary pair kept as y0_bin, y1_bin.
      twins_{full,sub}_deq<w>     Twins (Bodik's setup), w in DEQ_W
      twinsind_{full,sub}_deq<w>  control with NO cross-world dependence: the same covariates and treatment, each
                       twin's outcome redrawn independently from P(Y_t = 1 | x) (a logistic fit of that twin's outcome
                       on the 10 covariates); its twins are independent by construction, so their jitter is too;
                       w in (0,) + DEQ_W (w = 0: binary)"""
    from sklearn.linear_model import LogisticRegression
    z = np.load(os.path.join(HERE, "bodik_twins.npz"))
    Xa, T = z["X"].astype(np.float64), z["T"].astype(np.float64)
    y0, y1 = z["Y0"].astype(np.float64), z["Y1"].astype(np.float64)
    X10 = Xa[:, :10]
    rng = np.random.default_rng([SEED, 7])
    p = {t: LogisticRegression(max_iter=2000).fit(X10, yt).predict_proba(X10)[:, 1] for t, yt in ((0, y0), (1, y1))}
    i0, i1 = (rng.random(T.size) < p[0]).astype(np.float64), (rng.random(T.size) < p[1]).astype(np.float64)
    X = Xa[:, :d]
    for kind, (a0, a1), widths in (("twins", (y0, y1), DEQ_W), ("twinsind", (i0, i1), (0.0,) + DEQ_W)):
        for w in widths:
            if w > 0 and kind == "twins":
                u0, u1, c, tgt = _corr_jitter(X, a0, a1, w, [SEED, 9, d, int(round(w * 1000))])
                print(f"{kind} d={d} w={w:g}: target rho {tgt:.3f}, jitter copula corr {c:.3f}", flush=True)
            elif w > 0:                                          # control: the twins are independent by construction,
                g = np.random.default_rng([SEED, 9, d, int(round(w * 1000))])   # so the jitter is independent too
                u0, u1 = g.uniform(-w, w, T.size), g.uniform(-w, w, T.size)
            else:
                u0 = u1 = np.zeros(T.size)
            j0, j1 = a0 + u0, a1 + u1
            _write_bodik(out, kind, f"deq{w:g}", X, T, j0, j1, np.where(T == 1, j1, j0),
                         {"y0_bin": a0, "y1_bin": a1})
    print(f"control: corr(Y0, Y1) {np.corrcoef(i0, i1)[0, 1]:.3f} (Twins {np.corrcoef(y0, y1)[0, 1]:.3f}); "
          f"mortality {i0.mean():.3f}/{i1.mean():.3f} (Twins {y0.mean():.3f}/{y1.mean():.3f})", flush=True)


if __name__ == "__main__":
    out = sys.argv[1]
    if "--twins-deq" in sys.argv:
        for d_ in (10, 1):
            twins_deq(out, d_)
    elif "--twins-bodik" in sys.argv:
        for d_ in (10, 1):
            twins_bodik(out, d=d_)
    else:
        ihdp(out)
        twins(out)
