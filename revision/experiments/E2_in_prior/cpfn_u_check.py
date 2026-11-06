"""Does u dominate in CausalPFN-C's 2D prior? On the R = 1000 in-prior datasets of the cpfn_2d prior (500 query units
each), the shared-noise truths give y_t = m_t + eta (eta shared by the two arms, eta1 := eta0), with m_t = E_y_t.
  eta  : exact, eta = y0 - m0 (= y1 - m1)
  u_t  : the part of m_t that the covariates x do not explain, u_t = m_t - f_t(x), with f_t a gradient-boosting fit of
         m_t on x, 5-fold cross-fitted (each unit's residual from a model that did not see it)
Per dataset: Var(u0), Var(u1), Var(eta), corr(u0, u1), the share of eta in the noise Var(eta) / (Var(u) + Var(eta))
(Var(u) the mean of the two arms), and the realised correlation of the outcomes given x,
corr(Y0, Y1 | x) = corr(u0 + eta, u1 + eta).
python cpfn_u_check.py DATA OUT_TXT   (DATA = $SCRATCH/e2/new/data)
"""
import os, sys
import numpy as np
from multiprocessing import Pool
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SHARD

DATA, OUT = sys.argv[1:3]
PRIOR, NW = "cpfn_2d", int(os.environ.get("CHECK_NW", 1000))   # CHECK_NW: fewer datasets for a local test


def crossfit(X, m):
    res = np.empty_like(m)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        f = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(X[tr], m[tr])
        res[te] = m[te] - f.predict(X[te])
    return res


def one(r):
    t = np.load(os.path.join(DATA, PRIOR, "truths_y", f"r{r:03d}.npz"))
    X = np.load(os.path.join(DATA, PRIOR, f"shard{r // SHARD}", "complexmech", "5node", "path_TY", "hide_0.0",
                             f"r{r:03d}.npz"))["X_test"].astype(np.float64)
    y0, y1, m0, m1 = (t[k].astype(np.float64) for k in ("y0", "y1", "m0", "m1"))
    eta0, eta1 = y0 - m0, y1 - m1
    u0, u1 = crossfit(X, m0), crossfit(X, m1)
    vu, ve = (u0.var() + u1.var()) / 2, eta0.var()
    c = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan
    return dict(r=r, eta_shared_maxdiff=float(np.abs(eta1 - eta0).max() / (eta0.std() + 1e-12)),
                var_u0=u0.var(), var_u1=u1.var(), var_eta=ve, corr_u=c(u0, u1),
                share_eta=ve / (vu + ve) if vu + ve > 0 else np.nan, corr_y_given_x=c(u0 + eta0, u1 + eta1))


if __name__ == "__main__":
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 4))) as pool:
        rows = pool.map(one, range(NW))
    lines = ["CausalPFN-C 2D prior (cpfn_2d), 1000 in-prior datasets x 500 query units: u = part of m_t not explained by x",
             "(gradient boosting, 5-fold cross-fitted), eta = y - m (exact, shared by the arms)", ""]
    q = lambda k: np.array([d[k] for d in rows], float)
    lines.append(f"check eta shared: max |eta1 - eta0| / sd(eta) over datasets {np.nanmax(q('eta_shared_maxdiff')):.2g}")
    for k, name in (("share_eta", "share of eta in the noise, Var(eta) / (Var(u) + Var(eta))"),
                    ("corr_u", "corr(u0, u1)"), ("corr_y_given_x", "realised corr(Y0, Y1 | x)")):
        v = q(k); v = v[np.isfinite(v)]
        lines.append(f"{name}: n {len(v)}, median {np.median(v):.3f}, IQR {np.percentile(v, 25):.3f} to {np.percentile(v, 75):.3f}, "
                     f"5-95% {np.percentile(v, 5):.3f} to {np.percentile(v, 95):.3f}, mean {v.mean():.3f}")
    v = q("share_eta"); v = v[np.isfinite(v)]
    lines.append(f"datasets where u dominates (share of eta < 0.5): {np.mean(v < 0.5):.2f}")
    v = q("corr_y_given_x"); v = v[np.isfinite(v)]
    lines += [f"datasets with realised corr(Y0, Y1 | x) < 0.3: {np.mean(v < 0.3):.2f}; < 0.5: {np.mean(v < 0.5):.2f}; "
              f">= 0.8: {np.mean(v >= 0.8):.2f}"]
    open(OUT, "w").write("\n".join(lines) + "\n")
    np.savez(OUT.replace(".txt", ".npz"), **{k: q(k) for k in rows[0]})
    print("\n".join(lines))
