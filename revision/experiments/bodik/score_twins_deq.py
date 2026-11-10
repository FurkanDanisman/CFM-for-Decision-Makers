"""Dequantized Twins (both twins jittered, correlated so the cross-world correlation is kept; make_worlds.twins_deq)
and the no-dependence control, scored like Table 2 on the jittered (continuous) outcomes.
python score_twins_deq.py DATA_ROOT OUT_PREFIX [MODELS: dump dirs, e.g. dopfn1d,dopfn2d | all] [VARIANTS: e.g. twins0.1,twinsind | all]
    [grid | nogrid: the 1D copula at rho = 0, 0.1, ..., 1 instead of -0.5, 0.5, 1, est] [D: covariate sets, e.g. 10]

Per variant (twins: w = 0.1, 0.25, 0.5; twinsind: w = 0, 0.1, 0.25, 0.5), covariate set (d = 10, 1) and model:
  mse          (predicted mean - the other twin's jittered outcome)^2. 2D: the joint conditioned on the known twin's
               jittered outcome (the bin of the known arm that holds it; outside the grid, the end bin); 1D and
               2D with the dependence off (rho = 0): the other arm's marginal (bodik_cf_readouts.cond_hist). The
               prediction is the histogram's mean (bodik_cf_readouts.mean_hist), the Table 2 read-out.
  IS           interval score at alpha = 0.05 of the same histogram's 95% interval (bodik_cf_readouts.hist_interval:
               2.5% / 97.5% points on bin edges, the paper's read-out) against the other twin's jittered outcome
  breakdown    printed under each row: IS = width + miss penalty, coverage, the IS with the interval points interpolated
               inside the bins (no edge snapping), and the miss rate when the known twin died / survived
  rho_pred     2D only: correlation of the predicted joint (raw bin centres), median over queries
  rho_true     correlation of the two twins' binary outcomes in the variant's truths
Aggregation as score_twins_bodik (all families: SE over families; UWYK: per context draw, SE over draws).
"""
import csv
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score_twins_bodik as S                                   # noqa: E402
import bodik_cf_readouts as R                                   # noqa: E402
from twins_copula_cond import marginal                          # noqa: E402  (both arms' marginals, raw units)
from scipy.stats import norm                                    # noqa: E402
from bodik_cf import _map                                       # noqa: E402

MODELS = [("Do-PFN", "dopfn1d", None, "full"), ("Do-PFN 2D", "dopfn2d", None, "full"),
          ("CausalPFN-C", "cpfn1d", None, "full"), ("CausalPFN-C 2D", "cpfn2d", None, "full"),
          ("UWYK No-Anc", "uwyk1d", "noanc", "sub"), ("UWYK Anc", "uwyk1d", "v3a", "sub"),
          ("UWYK 2D No-Anc", "graph2d", "noanc", "sub"), ("UWYK 2D Anc", "graph2d", "v3a", "sub"),
          ("Do-PFN (matched resolution)", "dopfn1d_mr", None, "full"),
          ("CausalPFN-C (matched resolution)", "cpfn1d_mr", None, "full"),
          ("UWYK No-Anc (matched resolution)", "uwyk1d_mr", "noanc", "sub"),
          ("UWYK Anc (matched resolution)", "uwyk1d_mr", "v3a", "sub"),
          ("Do-PFN (matched label)", "dopfn1d_ml", None, "full"),
          ("CausalPFN-C (matched label)", "cpfn1d_ml", None, "full"),
          ("UWYK No-Anc (matched label)", "uwyk1d_ml", "noanc", "sub"),
          ("UWYK Anc (matched label)", "uwyk1d_ml", "v3a", "sub")]
ONLY = [m for m in (sys.argv[3] if len(sys.argv) > 3 else "all").split(",") if m not in ("", "all")]  # dump dirs
VONLY = [v for v in (sys.argv[4] if len(sys.argv) > 4 else "all").split(",") if v not in ("", "all")]  # twins0.1, twinsind
if ONLY:                                                        # a token is a dump dir, or dir:tag (uwyk1d:noanc)
    MODELS = [m for m in MODELS if m[1] in ONLY or f"{m[1]}:{m[2]}" in ONLY]
DS = tuple(int(x) for x in sys.argv[6].split(",")) if len(sys.argv) > 6 else (10, 1)   # covariate sets
COP_RHOS = (-0.5, 0.5, 1.0, "est")                              # 1D conditioned through a Gaussian copula
if len(sys.argv) > 5 and sys.argv[5] == "grid":                 # (argv[5] = nogrid: the default rhos)                 # the analyst's range rho in [0, 1], step 0.1
    COP_RHOS = tuple(k / 10 for k in range(11))
ESTIMATED = {1: 0.649, 10: 0.636}                               # tab:rho-all (the jittered pairs keep it)
VARIANTS = [("twins", w) for w in ("0.1", "0.25", "0.5")] + [("twinsind", w) for w in ("0", "0.1", "0.25", "0.5")]
if VONLY:
    VARIANTS = [(k, w) for k, w in VARIANTS if k in VONLY or f"{k}{w}" in VONLY]


def score(E, W, T, y, yf):
    """per query: squared error of the mean, interval score of the 95% interval; then the breakdown of the IS:
    width, miss penalty, covered, IS with the 2.5% / 97.5% points interpolated inside the bins (no edge snapping),
    and the miss and its group indicator for known twin died / survived (y_f >= 1/2 / < 1/2)"""
    m = R.mean_hist(E, W, T)
    L, U = R.hist_interval(E, W, T)
    pen = (2 / R.A) * np.maximum(0, L - y) + (2 / R.A) * np.maximum(0, y - U)
    F = np.concatenate([T[:, :1], T[:, :1] + np.cumsum(W, 1)], 1)
    Li = np.array([np.interp(R.A / 2, F[i], E[i]) for i in range(y.size)])
    Ui = np.array([np.interp(1 - R.A / 2, F[i], E[i]) for i in range(y.size)])
    ISi = (Ui - Li) + (2 / R.A) * np.maximum(0, Li - y) + (2 / R.A) * np.maximum(0, y - Ui)
    miss, died = (y < L - 1e-9) | (y > U + 1e-9), yf >= 0.5
    return np.stack([(m - y) ** 2, U - L + pen, U - L, pen, ~miss, ISi, miss & died, died, miss & ~died, ~died],
                    1).astype(float)


def marginals_2d(z, tag):
    """the 2D model's own marginals of both arms from its joint (raw units, no tails)"""
    key = f"p_joint_scaled_{tag}" if tag and f"p_joint_scaled_{tag}" in z.files else "p_joint_scaled"
    J = np.asarray(z[key], np.float64)
    J = J / J.sum((1, 2), keepdims=True)
    es = np.asarray(z["edges"], np.float64).reshape(-1)
    n = J.shape[0]
    T = np.zeros((n, 4)) + [0, 1, 0, 1]
    return {a: (es * _map(z, a)[0] + _map(z, a)[1], J.sum(2 if a == 0 else 1), T) for a in (0, 1)}


def copula_1d(f, tag, t, yf, y, rho, from_2d=False):
    """1D baseline conditioned on the known twin's VALUE through a Gaussian copula with correlation rho:
    z_f = Phi^{-1}(F_f(y_f)) under the baseline's marginal of the known twin (within-bin linear CDF); the other
    twin's latent is N(rho z_f, 1 - rho^2), mapped through its marginal: bin masses H(b_j) - H(b_{j-1}),
    b_j = Phi^{-1}(F_c(e_j)), H(b) = Phi((b - rho z_f) / s); tails H(b_0), 1 - H(b_K) (half-normal shapes kept)."""
    z = np.load(f, allow_pickle=True)
    M = marginals_2d(z, tag) if from_2d else {a: marginal(z, tag, a) for a in (0, 1)}
    K = M[0][1].shape[1]
    E, W, T = np.empty((t.size, K + 1)), np.empty((t.size, K)), np.empty((t.size, 4))
    s = np.sqrt(max(1 - rho ** 2, 1e-6))
    for arm_f in (0, 1):
        sel = np.flatnonzero(t == arm_f)
        if sel.size == 0:
            continue
        Ef, Wf, Tf = M[arm_f][0], M[arm_f][1][sel], M[arm_f][2][sel]
        Ec, Wc, Tc = M[1 - arm_f][0], M[1 - arm_f][1][sel], M[1 - arm_f][2][sel]
        Ff = Tf[:, :1] + np.concatenate([np.zeros((sel.size, 1)), np.cumsum(Wf, 1)], 1)
        v = yf[sel]
        i = np.clip(np.searchsorted(Ef, v, side="right") - 1, 0, K - 1)
        fr = np.clip((v - Ef[i]) / (Ef[i + 1] - Ef[i]), 0, 1)
        r_ = np.arange(sel.size)
        u = Ff[r_, i] + Wf[r_, i] * fr
        u = np.where(v < Ef[0], Tf[:, 0] / 2, np.where(v > Ef[-1], 1 - Tf[:, 2] / 2, u))
        zf = norm.ppf(np.clip(u, 1e-9, 1 - 1e-9))
        Fc = Tc[:, :1] + np.concatenate([np.zeros((sel.size, 1)), np.cumsum(Wc, 1)], 1)
        b = norm.ppf(np.clip(Fc, 1e-12, 1 - 1e-12))
        H = norm.cdf((b - rho * zf[:, None]) / s)
        E[sel], W[sel] = Ec, np.diff(H, axis=1)
        T[sel] = np.stack([H[:, 0], Tc[:, 1], 1 - H[:, -1], Tc[:, 3]], 1)
    return score(E, W, T, y, yf)


def per_query(D, part, f, tag, is2d, d, ld_off=False, rho=None):
    """rho: condition through a Gaussian copula with this rho on the model's marginals (2D: its joint's marginals,
    i.e. its learned dependence replaced by rho)"""
    k = int(re.search(r"r(\d+)\.npz$", f).group(1))
    z = np.load(os.path.join(D, part, f"d{d}", "truths", f"r{k:03d}.npz"))
    t = np.rint(z["t_test"]).astype(int)
    y0, y1 = z["y0"].astype(float), z["y1"].astype(float)
    yf, y = np.where(t == 1, y1, y0), np.where(t == 1, y0, y1)
    if rho is not None:
        return k, copula_1d(f, tag, t, yf, y, rho, from_2d=is2d)
    if not is2d or ld_off:
        E, W, T, _ = R.cond_hist(f, tag, t, yf, ld_off=ld_off)
        return k, score(E, W, T, y, yf)
    with np.load(f, allow_pickle=True) as zz:
        key = f"p_joint_scaled_{tag}" if tag and f"p_joint_scaled_{tag}" in zz.files else "p_joint_scaled"
        J = np.asarray(zz[key], np.float64)
        es = np.asarray(zz["edges"], np.float64).reshape(-1)
        e = {a: es * _map(zz, a)[0] + _map(zz, a)[1] for a in (0, 1)}
    J = J / J.sum((1, 2), keepdims=True)
    K = J.shape[-1]
    E, W = np.empty((t.size, K + 1)), np.empty((t.size, K))
    for arm_f in (0, 1):
        sel = np.flatnonzero(t == arm_f)
        if sel.size == 0:
            continue
        kb = np.clip(np.searchsorted(e[arm_f], yf[sel], side="right") - 1, 0, K - 1)
        M = J[sel] if arm_f == 0 else np.transpose(J[sel], (0, 2, 1))       # [q, known bin, other bin]
        w_ = M[np.arange(sel.size), kb, :]
        W[sel] = w_ / np.maximum(w_.sum(1, keepdims=True), 1e-300)
        E[sel] = e[1 - arm_f]
    return k, score(E, W, np.zeros((t.size, 4)) + [0, 1, 0, 1], y, yf)


def rho_pred(f, tag):
    with np.load(f, allow_pickle=True) as z:
        k = f"p_joint_scaled_{tag}" if tag and f"p_joint_scaled_{tag}" in z.files else "p_joint_scaled"
        J = np.asarray(z[k], np.float64)
        J = J / J.sum((1, 2), keepdims=True)
        es = np.asarray(z["edges"], np.float64).reshape(-1)
        (s0, m0), (s1, m1) = _map(z, 0), _map(z, 1)
    e0, e1 = es * s0 + m0, es * s1 + m1
    c0, c1 = 0.5 * (e0[:-1] + e0[1:]), 0.5 * (e1[:-1] + e1[1:])
    p0, p1 = J.sum(2), J.sum(1)
    mu0, mu1 = p0 @ c0, p1 @ c1
    v0, v1 = p0 @ c0 ** 2 - mu0 ** 2, p1 @ c1 ** 2 - mu1 ** 2
    cov = np.einsum("qij,i,j->q", J, c0, c1) - mu0 * mu1
    return cov / np.sqrt(np.maximum(v0 * v1, 1e-12))


def agg(got, sub):
    """mean and SE per column (MSE, IS)"""
    if not sub:
        V = np.concatenate([v for _, v in got])
        return V.mean(0), V.std(0, ddof=1) / np.sqrt(V.shape[0])
    draws = {}
    for k, v in got:
        draws.setdefault(k // S.NB, []).append(v)
    M = np.array([np.concatenate(v).mean(0) for v in draws.values()])
    return M.mean(0), M.std(0, ddof=1) / np.sqrt(M.shape[0])


def main(D, out):
    recs = []
    for kind, w in VARIANTS:
        for d in DS:
            for label, mdir, tag, part in MODELS:
                sub = part == "sub"
                pname = f"{kind}_{part}_deq{w}"
                files = sorted(glob.glob(os.path.join(D, "dumps", mdir, pname, f"d{d}", "*r[0-9]*.npz")))
                if not files:
                    print("missing", pname, label, d, flush=True); continue
                is2d = "2D" in label
                rows = [(label, False, None)] + ([(label + " (ρ = 0)", True, None),
                                                   (label + " copula ρ=est", False, ESTIMATED[d])] if is2d else
                         [(label + f" copula ρ={'est' if r == 'est' else r}", False, ESTIMATED[d] if r == "est" else r)
                          for r in COP_RHOS])
                for lab, ld_off, rho in rows:
                    got = []
                    for f in files:
                        got.append(per_query(D, pname, f, tag, is2d, d, ld_off=ld_off, rho=rho))
                    m, se = agg(got, sub)
                    r = dict(variant=kind, w=w, setting=f"d={d}", model=lab, mse=float(m[0]), mse_se=float(se[0]),
                             IS=float(m[1]), IS_se=float(se[1]))
                    if is2d and not ld_off and rho is None:
                        r["rho_pred"] = float(np.median(np.concatenate([rho_pred(f, tag) for f in files])))
                    t = np.load(os.path.join(D, pname, f"d{d}", "truths", "r000.npz"))
                    r["rho_true"] = float(np.corrcoef(t["y0"], t["y1"])[0, 1])
                    recs.append(r)
                    print(f"{kind:8s} w={w:4s} d={d:2d} {lab:26s} MSE {m[0]:.4f} ± {se[0]:.4f}  IS {m[1]:.3f} ± {se[1]:.3f}"
                          + (f"  rho_pred {r['rho_pred']:.3f}" if "rho_pred" in r else "")
                          + f"  rho_true {r['rho_true']:.3f}", flush=True)
                    print(f"{'':44s}IS = width {m[2]:.3f} + miss penalty {m[3]:.3f}; coverage {m[4]:.3f};"
                          f" IS interpolated {m[5]:.3f}; miss rate | known died {m[6] / m[7]:.3f},"
                          f" | known survived {m[8] / m[9]:.3f}", flush=True)
    keys = ["variant", "w", "setting", "model", "mse", "mse_se", "IS", "IS_se", "rho_pred", "rho_true"]
    with open(out + ".csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys); wr.writeheader(); wr.writerows(recs)
    print(f"wrote {out}.csv", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
