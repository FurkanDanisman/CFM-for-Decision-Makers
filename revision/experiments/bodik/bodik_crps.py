"""Twins, Bodik's task (predict the other twin from x and the known twin): MSE and CRPS, with the 1D models also
under fixed couplings.
python bodik_crps.py DATA_ROOT OUT_PREFIX [d]

For every query the predicted CDF G of the other twin's outcome (raw units) is built on a fine per-query grid:
  2D models: the learned joint, conditioned on the known twin's outcome (its bins with outcome >= 1/2 if the known
             twin died, < 1/2 if it survived; the bin containing 1/2 split linearly); uniform within bins.
  1D models: the two predicted marginals F_f (known twin's arm) and F_cf (other arm), joined by a copula C, then
             conditioned on the same event:  G(x) = C(F_cf(x), v) / v            if the known twin survived,
                                             G(x) = (F_cf(x) - C(F_cf(x), v)) / (1 - v)   if it died,
             with v = F_f(1/2). C: independence (rho = 0, the main tables), Gaussian copula rho = -0.5 and 0.5,
             comonotone (rho = 1). Under independence G = F_cf: the known twin is not used.
Bins are uniform within; UWYK's half-normal tails are exact.
  MSE  = mean (E[Y_cf] - Y_cf)^2,     E[Y_cf] = x_min + integral (1 - G)
  CRPS = mean integral (G(x) - 1{x >= Y_cf})^2 dx
"""
import csv
import glob
import os
import re
import sys

import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bodik_cf import _map, _above                               # noqa: E402

MODELS_2D = [("Do-PFN 2D", "dopfn2d", None), ("UWYK 2D No-Anc", "graph2d", "noanc"), ("UWYK 2D Anc", "graph2d", "v3a"),
             ("CausalPFN-C 2D", "cpfn2d", None)]
MODELS_1D = [("Do-PFN", "dopfn1d", None), ("Do-PFN (matched resolution)", "dopfn1d_mr", None),
             ("UWYK No-Anc", "uwyk1d", "noanc"), ("UWYK Anc", "uwyk1d", "v3a"),
             ("UWYK No-Anc (matched resolution)", "uwyk1d_mr", "noanc"), ("UWYK Anc (matched resolution)", "uwyk1d_mr", "v3a"),
             ("CausalPFN-C", "cpfn1d", None), ("CausalPFN-C (matched resolution)", "cpfn1d_mr", None)]
COUPLINGS = (("ρ = 0 (independent)", 0.0), ("ρ = −0.5", -0.5), ("ρ = 0.5", 0.5), ("ρ = 1 (comonotone)", 1.0))
SUB = 4                                                          # points per bin (copula makes G non-linear)
TAILQ = np.concatenate([np.geomspace(1e-5, 0.5, 20), 1 - np.geomspace(0.5, 1e-5, 20)[1:]])   # half-normal levels
GL_X, GL_W = np.polynomial.legendre.leggauss(48)


def arm_cdf(P, e, tails, X):
    """CDF of a bin pmf P (n, K) on raw edges e (K+1), uniform within bins, plus half-normal tails
    (pL, sL, pR, sR) per query, evaluated at X (n, M)."""
    pL, sL, pR, sR = tails
    Fe = pL[:, None] + np.concatenate([np.zeros((P.shape[0], 1)), np.cumsum(P, 1)], 1)    # at the edges
    F = np.empty_like(X)
    for i in range(X.shape[0]):
        F[i] = np.interp(X[i], e, Fe[i])
    left, right = X < e[0], X > e[-1]
    F = np.where(left, 2 * pL[:, None] * norm.cdf((X - e[0]) / sL[:, None]), F)
    F = np.where(right, Fe[:, -1:] + pR[:, None] * (2 * norm.cdf((X - e[-1]) / sR[:, None]) - 1), F)
    return np.clip(F, 0.0, 1.0)


def grid(e, tails, n):
    """per-query evaluation points: every edge subdivided SUB times, 0 and 1, and the tails' quantiles."""
    pL, sL, pR, sR = tails
    base = np.concatenate([np.linspace(e[j], e[j + 1], SUB, endpoint=False) for j in range(e.size - 1)] + [[e[-1]], [0.0, 1.0]])
    lt = e[0] - sL[:, None] * norm.ppf(1 - TAILQ[None, :] / 2) * (pL[:, None] > 0)
    rt = e[-1] + sR[:, None] * norm.ppf(1 - TAILQ[None, :] / 2) * (pR[:, None] > 0)
    X = np.concatenate([np.repeat(base[None, :], n, 0), lt, rt], 1)
    return np.sort(X, 1)


def gauss_copula(u, v, rho):
    """C(u, v) = Phi_2(Phi^-1(u), Phi^-1(v); rho); u (n, M), v (n,)."""
    a = norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))
    b = norm.ppf(np.clip(v, 1e-12, 1 - 1e-12))
    lo = -8.0
    z = lo + (b[:, None] - lo) * (GL_X[None, :] + 1) / 2                    # (n, 48) nodes on [lo, b]
    w = (b[:, None] - lo) / 2 * GL_W[None, :] * norm.pdf(z)
    s = np.sqrt(1 - rho * rho)
    out = np.empty_like(u)
    for i0 in range(0, u.shape[0], 200):
        sl = slice(i0, i0 + 200)
        out[sl] = np.einsum("nmk,nk->nm", norm.cdf((a[sl, :, None] - rho * z[sl, None, :]) / s), w[sl])
    return np.clip(out, 0.0, np.minimum(u, v[:, None]))


def conditional(Fcf, v, died, rho):
    if rho == 0.0:
        C = Fcf * v[:, None]
    elif rho == 1.0:
        C = np.minimum(Fcf, v[:, None])
    else:
        C = gauss_copula(Fcf, v, rho)
    G = np.where(died[:, None], (Fcf - C) / np.maximum(1 - v[:, None], 1e-12), C / np.maximum(v[:, None], 1e-12))
    return np.clip(np.maximum.accumulate(G, 1), 0.0, 1.0)


def scores(X, G, y):
    """mean and CRPS from the CDF G at sorted points X, G linear between points, against y in {0, 1} (on the grid)."""
    dx = np.diff(X, axis=1)
    g1, g2 = G[:, :-1], G[:, 1:]
    mean = X[:, 0] + (dx * (1 - (g1 + g2) / 2)).sum(1)
    h = (X[:, :-1] >= y[:, None] - 1e-12).astype(float)                    # step at y (y is a grid point)
    crps = (dx * ((g1 - h) ** 2 + (g1 - h) * (g2 - h) + (g2 - h) ** 2) / 3).sum(1)
    return mean, crps


def load(f, tag):
    z = np.load(f, allow_pickle=True)
    g = lambda b: (np.asarray(z[f"{b}_{tag}" if tag else b], np.float64) if (f"{b}_{tag}" if tag else b) in z.files else None)
    return z, g


def run_file(f, tag, t, yf, ycf, mode, rho=0.0, ld_off=False):
    z, g = load(f, tag)
    es = np.asarray(z["edges"], np.float64).reshape(-1)
    s0, m0 = _map(z, 0); s1, m1 = _map(z, 1)
    e = {0: es * s0 + m0, 1: es * s1 + m1}; s = {0: s0, 1: s1}
    n = t.size
    mean, crps = np.empty(n), np.empty(n)
    joint = g("p_joint_scaled")
    for arm_f in (0, 1):
        sel = np.flatnonzero(t == arm_f)
        if sel.size == 0:
            continue
        arm_cf = 1 - arm_f
        died = yf[sel] >= 0.5
        if mode == "2d":
            J = joint[sel]; J = J / J.sum((1, 2), keepdims=True)
            a = _above(e[arm_f])
            wf = np.where(died[:, None], a[None, :], 1 - a[None, :])
            if ld_off:
                wf = np.ones_like(wf)
            M = J if arm_f == 0 else np.transpose(J, (0, 2, 1))
            W = np.einsum("qi,qij->qj", wf, M); W = W / W.sum(1, keepdims=True)
            zt = (np.zeros(sel.size), np.ones(sel.size), np.zeros(sel.size), np.ones(sel.size))
            X = grid(e[arm_cf], zt, sel.size)
            G = arm_cdf(W, e[arm_cf], zt, X)
        else:
            def marg(a_):
                p = g(f"p_y{a_}_scaled")[sel]; p = p / p.sum(1, keepdims=True)
                if p.shape[1] == es.size + 1:                    # UWYK tails
                    sfx = f"_{tag}" if tag and f"sL_raw_{a_}_{tag}" in z.files else ""
                    soft = lambda x: np.logaddexp(0.0, x)
                    fl = float(z["scale_floor"])
                    sL = float(z["base_s_left"]) * (soft(np.asarray(z[f"sL_raw_{a_}{sfx}"], np.float64)[sel]) + fl) * s[a_]
                    sR = float(z["base_s_right"]) * (soft(np.asarray(z[f"sR_raw_{a_}{sfx}"], np.float64)[sel]) + fl) * s[a_]
                    return p[:, 1:-1], (p[:, 0], sL, p[:, -1], sR)
                return p, (np.zeros(sel.size), np.ones(sel.size), np.zeros(sel.size), np.ones(sel.size))
            Pf, tf = marg(arm_f); Pc, tc = marg(arm_cf)
            v = arm_cdf(Pf, e[arm_f], tf, np.full((sel.size, 1), 0.5))[:, 0]   # P(known twin's outcome < 1/2)
            X = grid(e[arm_cf], tc, sel.size)
            G = conditional(arm_cdf(Pc, e[arm_cf], tc, X), v, died, rho)
        mean[sel], crps[sel] = scores(X, G, ycf[sel])
    z.close()
    return mean, crps


def main(D, out, d=10):
    recs = []
    jobs = [(lab, md, tag, "2d", None, None) for lab, md, tag in MODELS_2D] + \
           [(lab, md, tag, "1d", cl, r) for lab, md, tag in MODELS_1D for cl, r in COUPLINGS]
    for label, mdir, tag, mode, cl, rho in jobs:
        files = sorted(glob.glob(os.path.join(D, "dumps", mdir, "twins", f"d{d}", "*r[0-9]*.npz")))
        if not files:
            continue
        M, C, Y = [], [], []
        for f in files:
            k = int(re.search(r"r(\d+)\.npz$", f).group(1))
            tz = np.load(os.path.join(D, "twins", f"d{d}", "truths", f"r{k:03d}.npz"))
            t = np.rint(tz["t_test"]).astype(int)
            y0, y1 = tz["y0"].astype(float), tz["y1"].astype(float)
            yf, ycf = np.where(t == 1, y1, y0), np.where(t == 1, y0, y1)
            m_, c_ = run_file(f, tag, t, yf, ycf, mode, rho or 0.0)
            M.append(m_); C.append(c_); Y.append(ycf)
        M, C, Y = map(np.concatenate, (M, C, Y))
        rec = dict(setting=f"d={d}", model=label, coupling=cl or "learned joint", n=Y.size,
                   mse=float(np.mean((M - Y) ** 2)), crps=float(C.mean()), mean_pred=float(M.mean()), mean_obs=float(Y.mean()))
        recs.append(rec)
        print(f"{label:34s} {rec['coupling']:22s} MSE {rec['mse']:.4f}  CRPS {rec['crps']:.4f}", flush=True)
    with open(out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)
    md = [f"# Twins (d = {d}), Bodik's task: MSE and CRPS", "",
          "Predict the other twin from x and the known twin. 2D: learned joint conditioned on the known twin; 1D: its two "
          "marginals joined by the stated coupling, then conditioned on the known twin (independence ignores it). "
          "Lower is better.", "",
          "| model | coupling | MSE | CRPS |", "|---|---|---|---|"]
    for r in recs:
        md.append(f"| {r['model']} | {r['coupling']} | {r['mse']:.4f} | {r['crps']:.4f} |")
    open(out + ".md", "w").write("\n".join(md) + "\n")
    print(f"wrote {out}.csv / .md")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 10)
