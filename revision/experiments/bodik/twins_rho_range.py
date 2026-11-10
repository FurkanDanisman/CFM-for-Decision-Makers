"""Per-family correlation bounds of the jittered Twins outcomes given the first d covariates (one twin observed):
comonotone / countermonotone coupling of the two twins' marginals (1-p) U(-w, w) + p (1 + U(-w, w)).
P(Y_t = 1 | x) fitted (5-fold cross-fitted gradient boosting) either on ALL families' Y0 / Y1 ("all"), or only on the
families whose twin t is the factual one ("factual", what an analyst sees)."""
import sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import make_worlds as MW

z = np.load(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "bodik_twins.npz"))
y0, y1, T = z["Y0"].astype(float), z["Y1"].astype(float), z["T"].astype(float)
q = (np.arange(400) + 0.5) / 400


def fit(X, y, rows):
    """P(y = 1 | x) for every family, cross-fitted, trained only on `rows`"""
    p = np.empty(y.size)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        tr = tr[rows[tr]]
        p[te] = HistGradientBoostingClassifier(random_state=0).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return np.clip(p, 1e-6, 1 - 1e-6)


def rowcorr(A, B):
    A = A - A.mean(1, keepdims=True); B = B - B.mean(1, keepdims=True)
    return (A * B).sum(1) / np.sqrt((A * A).sum(1) * (B * B).sum(1))


def qf(p, w):
    P = p[:, None]
    in0 = q[None, :] < 1 - P
    a0 = np.clip(q[None, :] / (1 - P), 0, 1); a1 = np.clip((q[None, :] - (1 - P)) / P, 0, 1)
    return np.where(in0, w * (2 * a0 - 1), 1 + w * (2 * a1 - 1))


for d in (10, 1):
    X = z["X"][:, :d].astype(float)
    for src, r0, r1 in (("all", np.ones(T.size, bool), np.ones(T.size, bool)), ("factual", T == 0, T == 1)):
        p0, p1 = fit(X, y0, r0), fit(X, y1, r1)
        for w in MW.DEQ_W:
            Q0, Q1 = qf(p0, w), qf(p1, w)
            lo, hi = rowcorr(Q0, Q1[:, ::-1]), rowcorr(Q0, Q1)
            print(f"d = {d:2d}  {src:7s}  w = {w:<4}: lower bound median {np.median(lo):+.2f} "
                  f"(10-90%: {np.quantile(lo, .1):+.2f} to {np.quantile(lo, .9):+.2f}); upper bound median "
                  f"{np.median(hi):.2f} (10-90%: {np.quantile(hi, .1):.2f} to {np.quantile(hi, .9):.2f})")
