"""Bodik's (CLeaR 2026) evaluation on Twins, for the paper's models: retrospective counterfactual prediction.
python bodik_cf.py DATA_ROOT OUT_PREFIX          DATA_ROOT = $SCRATCH/bodik  (twins/, dumps/)

For every queried pair, given its covariates x and the FACTUAL twin's observed outcome y_f, predict the other
twin's outcome Y_cf. Reported: MSE (as Bodik), and the 95% prediction interval's |coverage - 0.95|, length and
interval score (alpha = 0.05, as in the paper's tables; Bodik uses the interval score at alpha = 0.1):
    MSE = mean (Yhat_cf - Y_cf)^2,      Yhat_cf = E[Y_cf | y_f, x]
    IS  = mean (U - L) + (2/a)(L - Y_cf)+ + (2/a)(Y_cf - U)+,   [L, U] = 2.5% and 97.5% quantiles of Y_cf | y_f, x
The conditional distribution of Y_cf:
    2D models: the learned joint, conditioned on the factual twin's outcome, i.e. on the event Y_f >= 1/2
               (dead) or < 1/2 (alive) on its grid, the bin containing 1/2 split linearly;
    1D models: their predicted marginal of the counterfactual arm (under independence y_f carries no information).
The outcome is 0/1, so each prediction is read as P = P(other twin dies), the probability above 1/2 (the split of
score_bodik.table_2x2, every model's grid in raw units, UWYK's half-normal tails included); Yhat_cf = P and the 95%
interval is {0} if P < 2.5%, {1} if P > 97.5%, else {0, 1}. Twins d = 10 and d = 1, all 11 query blocks pooled.
"""
import csv
import glob
import os
import re
import sys

import numpy as np
from scipy.stats import norm

A = 0.05
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = [  # label, dump dir, tag
    ("Do-PFN", "dopfn1d", None), ("Do-PFN 2D", "dopfn2d", None),
    ("UWYK No-Anc", "uwyk1d", "noanc"), ("UWYK Anc", "uwyk1d", "v3a"),
    ("UWYK 2D No-Anc", "graph2d", "noanc"), ("UWYK 2D Anc", "graph2d", "v3a"),
    ("CausalPFN-C", "cpfn1d", None), ("CausalPFN-C 2D", "cpfn2d", None),
    ("Do-PFN (matched resolution)", "dopfn1d_mr", None), ("Do-PFN (matched label)", "dopfn1d_ml", None),
    ("UWYK No-Anc (matched resolution)", "uwyk1d_mr", "noanc"), ("UWYK Anc (matched resolution)", "uwyk1d_mr", "v3a"),
    ("UWYK No-Anc (matched label)", "uwyk1d_ml", "noanc"), ("UWYK Anc (matched label)", "uwyk1d_ml", "v3a"),
    ("CausalPFN-C (matched resolution)", "cpfn1d_mr", None), ("CausalPFN-C (matched label)", "cpfn1d_ml", None),
    ("CausalPFN (published two-stage)", "cpfn_v0", None),
    ("UWYK-B No-Anc", "uwyk_bin", "noanc"), ("UWYK-B Anc", "uwyk_bin", "v3a")]


def _map(z, arm):
    """(scale, shift) from scaled to raw units for one arm."""
    if f"arm{arm}_shift" in z.files:
        return float(np.asarray(z[f"arm{arm}_scale"]).reshape(-1)[0]), float(np.asarray(z[f"arm{arm}_shift"]).reshape(-1)[0])
    return (float(z["y_scale"]) if "y_scale" in z.files else 1.0), (float(z["y_shift"]) if "y_shift" in z.files else 0.0)


def _above(e, thr=0.5):
    lo, hi = e[:-1], e[1:]
    return np.clip((hi - thr) / (hi - lo), 0.0, 1.0)


def _mean_quant(w, e, tails=None):
    """Mean and the A/2, 1 - A/2 quantiles of distributions given as bin masses w (n, K) on raw edges e (K+1),
    uniform within each bin; tails = (pL, sL, pR, sR) in raw units for half-normal tails beyond e[0] and e[-1]."""
    n = w.shape[0]
    c = 0.5 * (e[:-1] + e[1:])
    if tails is None:
        pL = pR = np.zeros(n); sL = sR = np.ones(n)
    else:
        pL, sL, pR, sR = tails
    tot = w.sum(1) + pL + pR
    w, pL, pR = w / tot[:, None], pL / tot, pR / tot
    k = np.sqrt(2 / np.pi)
    mean = (w * c).sum(1) + pL * (e[0] - k * sL) + pR * (e[-1] + k * sR)
    F = np.concatenate([pL[:, None], pL[:, None] + np.cumsum(w, 1)], 1)       # CDF at the edges

    def quant(p):
        out = np.empty(n)
        for i in range(n):
            if p < F[i, 0]:                                   # in the left tail: pL * 2 Phi(-(e0 - y)/sL) = p
                out[i] = e[0] + sL[i] * norm.ppf(p / (2 * F[i, 0]))
            elif p > F[i, -1]:                                # in the right tail
                out[i] = e[-1] + sR[i] * norm.ppf(1 - (1 - p) / (2 * (1 - F[i, -1])))
            else:
                out[i] = np.interp(p, F[i], e)
        return out
    return mean, quant(A / 2), quant(1 - A / 2)


def cf_distribution(f, tag, t, yf):
    """(mean, L, U) of Y_cf | y_f, x per query; t = treatment of the factual twin (1: the heavier is factual)."""
    with np.load(f, allow_pickle=True) as z:
        def g(base):
            k = f"{base}_{tag}" if tag else base
            return np.asarray(z[k], dtype=np.float64) if k in z.files else None
        joint, p0, p1 = g("p_joint_scaled"), g("p_y0_scaled"), g("p_y1_scaled")
        es = np.asarray(z["edges"], dtype=np.float64).reshape(-1)
        s0, m0 = _map(z, 0); s1, m1 = _map(z, 1)
        e0, e1 = es * s0 + m0, es * s1 + m1
        n = t.size
        mean, lo, hi = np.empty(n), np.empty(n), np.empty(n)
        if joint is not None and joint.ndim == 3:
            joint = joint / joint.sum((1, 2), keepdims=True)
            for arm_f, ef, ecf in ((1, e1, e0), (0, e0, e1)):  # factual arm, its edges, counterfactual edges
                sel = np.flatnonzero(t == arm_f)
                if sel.size == 0:
                    continue
                a = _above(ef)                                 # share of each factual bin with Y_f >= 1/2
                wf = np.where(yf[sel, None] >= 0.5, a[None, :], 1 - a[None, :])
                J = joint[sel] if arm_f == 0 else np.transpose(joint[sel], (0, 2, 1))   # [q, factual bin, cf bin]
                w = np.einsum("qi,qij->qj", wf, J)
                mean[sel], lo[sel], hi[sel] = _mean_quant(w, ecf)
            return mean, lo, hi
        # 1D: the counterfactual arm's own marginal
        for arm_cf, p, ecf, sc in ((0, p0, e0, s0), (1, p1, e1, s1)):
            sel = np.flatnonzero(t == 1 - arm_cf)
            if sel.size == 0:
                continue
            P = p[sel]
            if P.shape[1] == es.size - 1:
                mean[sel], lo[sel], hi[sel] = _mean_quant(P, ecf)
            elif P.shape[1] == es.size + 1:                    # UWYK: [left tail, K bars, right tail]
                sfx = f"_{tag}" if tag and f"sL_raw_{arm_cf}_{tag}" in z.files else ""
                soft = lambda x: np.logaddexp(0.0, x)
                fl = float(z["scale_floor"])
                sL = float(z["base_s_left"]) * (soft(np.asarray(z[f"sL_raw_{arm_cf}{sfx}"], np.float64)[sel]) + fl) * sc
                sR = float(z["base_s_right"]) * (soft(np.asarray(z[f"sR_raw_{arm_cf}{sfx}"], np.float64)[sel]) + fl) * sc
                mean[sel], lo[sel], hi[sel] = _mean_quant(P[:, 1:-1], ecf, (P[:, 0], sL, P[:, -1], sR))
            else:
                raise SystemExit(f"{f}: unrecognised layout {P.shape[1]} probabilities, {es.size} edges")
        return mean, lo, hi


def binary_p(f, tag, t, yf):
    """P(other twin dies | known twin's outcome, x) per query, from the model's 2 x 2 table of the pair (the split at
    1/2 of score_bodik.table_2x2: 2D = learned joint, 1D = product of marginals, so the known twin changes nothing)."""
    sys.path.insert(0, HERE)
    import score_bodik as SB
    P = SB.table_2x2(f, tag)
    P = np.clip(P, 1e-12, 1.0); P = P / P.sum(1, keepdims=True)
    M = P.reshape(-1, 2, 2)                                      # M[q, y0, y1]
    q = np.arange(t.size); kf = np.rint(yf).astype(int)
    # known = heavier (t = 1): condition on y1 = kf, predict y0;  known = lighter: condition on y0 = kf, predict y1
    num = np.where(t == 1, M[q, 1, kf], M[q, kf, 1])
    den = np.where(t == 1, M[q, :, kf].sum(1), M[q, kf, :].sum(1))
    return num / den


def main(D, out):
    recs = []
    for d in (10, 1):
        for label, mdir, tag in MODELS:
            files = sorted(glob.glob(os.path.join(D, "dumps", mdir, "twins", f"d{d}", "*r[0-9]*.npz")))
            if not files:
                continue
            Y, Pd = [], []
            for f in files:
                k = int(re.search(r"r(\d+)\.npz$", f).group(1))
                z = np.load(os.path.join(D, "twins", f"d{d}", "truths", f"r{k:03d}.npz"))
                t = np.rint(z["t_test"]).astype(int)
                y0, y1 = z["y0"].astype(float), z["y1"].astype(float)
                yf, ycf = np.where(t == 1, y1, y0), np.where(t == 1, y0, y1)
                Pd.append(binary_p(f, tag, t, yf)); Y.append(ycf)
            Y, P = np.concatenate(Y), np.clip(np.concatenate(Pd), 0.0, 1.0)
            lo = np.where(P > 1 - A / 2, 1.0, 0.0)               # interval on {0, 1}: {0}, {1} or {0, 1}
            hi = np.where(P < A / 2, 0.0, 1.0)
            IS = (hi - lo) + (2 / A) * np.maximum(0, lo - Y) + (2 / A) * np.maximum(0, Y - hi)
            cov = float(np.mean((lo <= Y) & (Y <= hi)))
            recs.append(dict(setting=f"d={d}", model=label, n=Y.size, mse=float(np.mean((P - Y) ** 2)),
                             abs_cov_err=abs(cov - 0.95), coverage=cov, length=float(np.mean(hi - lo)),
                             IS=float(IS.mean()), share_0=float(np.mean(hi == 0)), share_1=float(np.mean(lo == 1))))
            r = recs[-1]
            print(f"d={d:2d} {label:34s} MSE {r['mse']:.4f}  |c-0.95| {r['abs_cov_err']:.3f}  length {r['length']:.3f}  "
                  f"IS {r['IS']:.3f}", flush=True)
    with open(out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)
    md = ["# Twins: Bodik's counterfactual prediction (predict the other twin given x and the known twin)", "",
          "P = P(other twin dies): the model's probability above 1/2 (2D: learned joint conditioned on the known twin; "
          "1D: the other arm's marginal). MSE = mean (P - outcome)^2. 95% interval on {0, 1}: {0} if P < 2.5%, "
          "{1} if P > 97.5%, else {0, 1}. Lower is better.", "",
          "| setting | model | n | MSE | |ĉ − 0.95| | ĉ | length | IS₀.₀₅ | share {0} | share {1} |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        md.append(f"| {r['setting']} | {r['model']} | {r['n']} | {r['mse']:.4f} | {r['abs_cov_err']:.3f} | "
                  f"{r['coverage']:.3f} | {r['length']:.3f} | {r['IS']:.3f} | {r['share_0']:.3f} | {r['share_1']:.3f} |")
    open(out + ".md", "w").write("\n".join(md) + "\n")
    print(f"wrote {out}.csv / .md")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
