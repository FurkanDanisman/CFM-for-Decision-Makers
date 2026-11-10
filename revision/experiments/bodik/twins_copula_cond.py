"""Twins (Bodik's task, reviewer request): the single-outcome baselines conditioned on the known twin through a Gaussian
copula with an assumed rho, scored exactly like Table 2.
python twins_copula_cond.py DATA_ROOT OUT_PREFIX [PART]      (PART: bodik = Table 2 setup (default), twins = old setup)

Per query, a 1D baseline gives the marginals of both arms (raw units, UWYK keeps its half-normal tails). They are coupled
by a Gaussian copula with correlation rho on the latent normals (Z_f, Z_c) and conditioned on the known twin's event,
died (y_f >= 1/2) or survived (y_f < 1/2), as the 2D read-out does (bodik_cf_readouts.cond_hist: the bin holding 1/2
split linearly). With a = Phi^{-1}(F_f(1/2)), the conditional CDF of Z_c is
    died:     H(b) = int_{-inf}^b phi(z) Phi((rho z - a)/s) dz / (1 - Phi(a)),   s = sqrt(1 - rho^2)
    survived: H(b) = int_{-inf}^b phi(z) Phi((a - rho z)/s) dz / Phi(a)
(rho = 1: the normal truncated to z > a or z <= a; rho = 0: H = Phi, the marginal, i.e. Table 2's 1D rows).
The other twin's conditional histogram has bin masses H(b_j) - H(b_{j-1}), b_j = Phi^{-1}(F_c(e_j)), and tail masses
H(b_0) and 1 - H(b_K) (the half-normal tail shapes kept). Prediction = that histogram's mean (bodik_cf_readouts.mean_hist,
the read-out of Table 2); MSE against the other twin's outcome, aggregated as score_twins_bodik.py (all families: SE over
families; UWYK: per context draw, SE over draws).
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
import bodik_cf_readouts as R                                    # noqa: E402
from bodik_cf import _map, _above                                # noqa: E402

RHOS = (-0.5, 0.0, 0.5, 1.0)
NB = 6                                                           # query blocks per context draw (score_twins_bodik)
MODELS = [("Do-PFN", "dopfn1d", None, "full"), ("Do-PFN (matched resolution)", "dopfn1d_mr", None, "full"),
          ("CausalPFN-C", "cpfn1d", None, "full"), ("CausalPFN-C (matched resolution)", "cpfn1d_mr", None, "full"),
          ("UWYK No-Anc", "uwyk1d", "noanc", "sub"), ("UWYK Anc", "uwyk1d", "v3a", "sub"),
          ("UWYK No-Anc (matched resolution)", "uwyk1d_mr", "noanc", "sub"),
          ("UWYK Anc (matched resolution)", "uwyk1d_mr", "v3a", "sub")]
ZG = np.linspace(-9.0, 9.0, 18001)                               # latent grid for the conditional CDF H


def marginal(z, tag, arm):
    """Edges E (K+1,), masses W (n, K), tails T (n, 4) of one arm, raw units (the 1D branch of cond_hist)."""
    key = lambda b: f"{b}_{tag}" if tag and f"{b}_{tag}" in z.files else b
    p = np.asarray(z[key(f"p_y{arm}_scaled")], np.float64)
    es = np.asarray(z["edges"], np.float64).reshape(-1)
    s, m = _map(z, arm)
    E = es * s + m
    P = p / p.sum(1, keepdims=True)
    n = P.shape[0]
    if P.shape[1] == es.size + 1:                                 # UWYK: [left tail, K bars, right tail]
        soft = lambda x: np.logaddexp(0.0, x)
        fl = float(z["scale_floor"])
        sL = float(z["base_s_left"]) * (soft(np.asarray(z[key(f"sL_raw_{arm}")], np.float64)) + fl) * s
        sR = float(z["base_s_right"]) * (soft(np.asarray(z[key(f"sR_raw_{arm}")], np.float64)) + fl) * s
        return E, P[:, 1:-1], np.stack([P[:, 0], sL, P[:, -1], sR], 1)
    return E, P, np.zeros((n, 4)) + [0, 1, 0, 1]


def cdf_at_half(E, W, T):
    """F(1/2) per query: left tail + bins below 1/2 + the linear share of the bin holding 1/2."""
    below = 1.0 - _above(E)
    return np.clip(T[:, 0] + (W * below[None, :]).sum(1), 1e-12, 1 - 1e-12)


def cond_cdf(rho, a, died):
    """H on ZG for each query (n, G): conditional CDF of Z_c given the known twin's event."""
    pz = norm.pdf(ZG)[None, :]
    if rho >= 1.0 - 1e-12:
        if died:
            return np.clip((norm.cdf(ZG)[None, :] - norm.cdf(a)[:, None]) / norm.sf(a)[:, None], 0, 1)
        return np.minimum(norm.cdf(ZG)[None, :] / norm.cdf(a)[:, None], 1.0)
    if abs(rho) < 1e-12:
        return np.broadcast_to(norm.cdf(ZG)[None, :], (a.size, ZG.size))
    s = np.sqrt(1 - rho ** 2)
    g = norm.cdf((rho * ZG[None, :] - a[:, None]) / s) if died else norm.cdf((a[:, None] - rho * ZG[None, :]) / s)
    dens = pz * g
    H = np.concatenate([np.zeros((a.size, 1)), np.cumsum(0.5 * (dens[:, 1:] + dens[:, :-1]) * np.diff(ZG), 1)], 1)
    return np.clip(H / H[:, -1:], 0, 1)


def conditional_mean(Ec, Wc, Tc, a, died, rho):
    """Mean of the other twin's conditional histogram (n,) for queries sharing one event (died / survived)."""
    Fe = Tc[:, :1] + np.concatenate([np.zeros((Wc.shape[0], 1)), np.cumsum(Wc, 1)], 1)   # F_c at the edges
    b = norm.ppf(np.clip(Fe, 1e-15, 1 - 1e-15))
    H = cond_cdf(rho, a, died)
    He = np.stack([np.interp(b[i], ZG, H[i]) for i in range(b.shape[0])])
    W = np.diff(He, axis=1)
    T = np.stack([He[:, 0], Tc[:, 1], 1 - He[:, -1], Tc[:, 3]], 1)
    return R.mean_hist(np.broadcast_to(Ec, (W.shape[0], Ec.size)), W, T)


def per_file(f, truth, tag, rho):
    z = np.load(f, allow_pickle=True)
    tz = np.load(truth)
    t = np.rint(tz["t_test"]).astype(int)
    y0, y1 = tz["y0"].astype(float), tz["y1"].astype(float)
    yf, y = np.where(t == 1, y1, y0), np.where(t == 1, y0, y1)
    M = {arm: marginal(z, tag, arm) for arm in (0, 1)}
    pred = np.empty(t.size)
    for arm_f in (0, 1):
        Ef, Wf, Tf = M[arm_f]
        Ec, Wc, Tc = M[1 - arm_f]
        for died in (True, False):
            sel = np.flatnonzero((t == arm_f) & ((yf >= 0.5) == died))
            if sel.size == 0:
                continue
            a = norm.ppf(cdf_at_half(Ef, Wf[sel], Tf[sel]))
            pred[sel] = conditional_mean(Ec, Wc[sel], Tc[sel], a, died, rho)
    return (pred - y) ** 2


def main(D, out, part="bodik"):
    recs = []
    for d in (10, 1):
        for label, mdir, tag, sub in MODELS:
            sub_dir = f"twins_{sub}" if part == "bodik" else "twins"
            files = sorted(glob.glob(os.path.join(D, "dumps", mdir, sub_dir, f"d{d}", "*r[0-9]*.npz")))
            if not files:
                print("missing", label, d, flush=True); continue
            for rho in RHOS:
                got = []
                for f in files:
                    k = int(re.search(r"r(\d+)\.npz$", f).group(1))
                    got.append((k, per_file(f, os.path.join(D, sub_dir, f"d{d}", "truths", f"r{k:03d}.npz"), tag, rho)))
                if sub == "full" or part != "bodik":
                    V = np.concatenate([v for _, v in got])
                    m, se = V.mean(), V.std(ddof=1) / np.sqrt(V.size)
                else:
                    draws = {}
                    for k, v in got:
                        draws.setdefault(k // NB, []).append(v)
                    Mv = np.array([np.concatenate(v).mean() for v in draws.values()])
                    m, se = Mv.mean(), Mv.std(ddof=1) / np.sqrt(Mv.size)
                recs.append(dict(setting=f"d={d}", model=label, rho=rho, mse=float(m), mse_se=float(se)))
                print(f"d={d:2d} {label:34s} rho={rho:+.1f}  MSE {m:.4f} ± {se:.4f}", flush=True)
    with open(out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)
    print(f"wrote {out}.csv", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(sys.argv[3:4]))
