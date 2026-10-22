"""Twins, Bodik's task (predict the other twin from x and the known twin), under two ways of reading a model's
predicted histogram of the other twin's outcome. Same four metrics in every case: MSE, |coverage - 0.95|, length,
IS_0.05 (95% interval), against the other twin's actual outcome (0 = survived, 1 = died).
python bodik_cf_readouts.py DATA_ROOT OUT_PREFIX

The histogram (raw outcome units): 2D models, the learned joint conditioned on the known twin's outcome (its bins with
outcome >= 1/2 if the known twin died, < 1/2 if it survived, the bin containing 1/2 split linearly); 1D models, the
other arm's marginal. UWYK's 1D bar distribution keeps its half-normal tails.

Read-outs (prediction Yhat for the MSE; interval [L, U]):
  A  Bodik's: Yhat = the histogram's mean; [L, U] = its 2.5% and 97.5% quantiles, each read on a bin EDGE (the
     paper's hist read-out: the lower end is the lower edge of the bin holding the 2.5% point, the upper end the
     upper edge of the bin holding the 97.5% point). No split.
  B  nearest outcome: mass below 1/2 counts as 0, above as 1 (P = mass above 1/2); Yhat = P; interval {0} if
     P < 2.5%, {1} if P > 97.5%, else [0, 1].
Also printed: for every 2D model, whether its joint is stored as [lighter, heavier] (its row and column sums against
the per-arm marginals saved in the same dump).
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
from bodik_cf import MODELS, _map, _above                      # noqa: E402

A = 0.05
K2 = np.sqrt(2 / np.pi)


def cond_hist(f, tag, t, yf, ld_off=False):
    """Per query: edges E (n, K+1), masses W (n, K), tails T (n, 4) = (pL, sL, pR, sR) in raw units; and the 2D axis
    check (None for 1D)."""
    with np.load(f, allow_pickle=True) as z:
        def g(base):
            k = f"{base}_{tag}" if tag else base
            return np.asarray(z[k], dtype=np.float64) if k in z.files else None
        joint, p0, p1 = g("p_joint_scaled"), g("p_y0_scaled"), g("p_y1_scaled")
        es = np.asarray(z["edges"], dtype=np.float64).reshape(-1)
        s0, m0 = _map(z, 0); s1, m1 = _map(z, 1)
        e = {0: es * s0 + m0, 1: es * s1 + m1}
        n = t.size
        if joint is not None and joint.ndim == 3:
            K = joint.shape[-1]
            check = None
            if p0 is not None and p0.shape == joint.shape[:2]:
                check = (float(np.abs(joint.sum(2) - p0).max()), float(np.abs(joint.sum(1) - p0).max()))
            J = joint / joint.sum((1, 2), keepdims=True)
            E, W = np.empty((n, K + 1)), np.empty((n, K))
            for arm_f in (0, 1):
                sel = np.flatnonzero(t == arm_f)
                if sel.size == 0:
                    continue
                a = _above(e[arm_f])
                wf = np.where(yf[sel, None] >= 0.5, a[None, :], 1 - a[None, :])
                if ld_off:                                       # LD=Off: the other arm's marginal of the joint
                    wf = np.ones_like(wf)
                M = J[sel] if arm_f == 0 else np.transpose(J[sel], (0, 2, 1))   # [q, known bin, other bin]
                W[sel] = np.einsum("qi,qij->qj", wf, M)
                E[sel] = e[1 - arm_f]
            return E, W, np.zeros((n, 4)) + [0, 1, 0, 1], check
        K = p0.shape[-1] - (2 if p0.shape[-1] == es.size + 1 else 0)
        E, W, T = np.empty((n, K + 1)), np.empty((n, K)), np.zeros((n, 4)) + [0, 1, 0, 1]
        for arm_cf, p in ((0, p0), (1, p1)):
            sel = np.flatnonzero(t == 1 - arm_cf)
            if sel.size == 0:
                continue
            P = p[sel] / p[sel].sum(1, keepdims=True)
            E[sel] = e[arm_cf]
            if P.shape[1] == es.size + 1:                     # UWYK: [left tail, K bars, right tail]
                sfx = f"_{tag}" if tag and f"sL_raw_{arm_cf}_{tag}" in z.files else ""
                soft = lambda x: np.logaddexp(0.0, x)
                fl, sc = float(z["scale_floor"]), (s0 if arm_cf == 0 else s1)
                sL = float(z["base_s_left"]) * (soft(np.asarray(z[f"sL_raw_{arm_cf}{sfx}"], np.float64)[sel]) + fl) * sc
                sR = float(z["base_s_right"]) * (soft(np.asarray(z[f"sR_raw_{arm_cf}{sfx}"], np.float64)[sel]) + fl) * sc
                W[sel] = P[:, 1:-1]
                T[sel] = np.stack([P[:, 0], sL, P[:, -1], sR], 1)
            else:
                W[sel] = P
        return E, W, T, None


def mean_hist(E, W, T):
    c = 0.5 * (E[:, :-1] + E[:, 1:])
    return (W * c).sum(1) + T[:, 0] * (E[:, 0] - K2 * T[:, 1]) + T[:, 2] * (E[:, -1] + K2 * T[:, 3])


def hist_interval(E, W, T):
    """2.5% / 97.5% quantiles on bin edges (lower edge / upper edge of the bin holding the point); in a tail, the
    half-normal quantile."""
    n = E.shape[0]
    F = np.concatenate([T[:, :1], T[:, :1] + np.cumsum(W, 1)], 1)           # CDF at the edges
    L, U = np.empty(n), np.empty(n)
    for i in range(n):
        for p, out, lower in ((A / 2, L, True), (1 - A / 2, U, False)):
            if p < F[i, 0]:
                out[i] = E[i, 0] + T[i, 1] * norm.ppf(p / (2 * F[i, 0]))
            elif p > F[i, -1]:
                out[i] = E[i, -1] + T[i, 3] * norm.ppf(1 - (1 - p) / (2 * (1 - F[i, -1])))
            else:
                b = int(np.clip(np.searchsorted(F[i], p, side="left") - 1, 0, W.shape[1] - 1))
                out[i] = E[i, b] if lower else E[i, b + 1]
    return L, U


def binary_interval(P):
    return np.where(P > 1 - A / 2, 1.0, 0.0), np.where(P < A / 2, 0.0, 1.0)


def readouts(E, W, T):
    """{name: (Yhat, L, U)}"""
    out = {}
    tot = W.sum(1) + T[:, 0] + T[:, 2]
    W, T = W / tot[:, None], T.copy(); T[:, 0] /= tot; T[:, 2] /= tot
    # A
    out["A"] = (mean_hist(E, W, T),) + hist_interval(E, W, T)
    # B
    lo, hi = E[:, :-1], E[:, 1:]
    P = (W * np.clip((hi - 0.5) / (hi - lo), 0, 1)).sum(1) + T[:, 2] * (E[:, -1] >= 0.5) + T[:, 0] * (E[:, 0] > 0.5)
    P = np.clip(P, 0, 1)
    out["B"] = (P,) + binary_interval(P)
    return out


def metrics(Yhat, L, U, Y):
    cov = float(np.mean((L <= Y + 1e-9) & (Y <= U + 1e-9)))
    IS = (U - L) + (2 / A) * np.maximum(0, L - Y) + (2 / A) * np.maximum(0, Y - U)
    return dict(mse=float(np.mean((Yhat - Y) ** 2)), abs_cov_err=abs(cov - 0.95), coverage=cov,
                length=float(np.mean(U - L)), IS=float(IS.mean()))


def main(D, out):
    recs = []
    for d in (10, 1):
        for label, mdir, tag in MODELS:
            files = sorted(glob.glob(os.path.join(D, "dumps", mdir, "twins", f"d{d}", "*r[0-9]*.npz")))
            if not files:
                continue
            acc = {k: [[], [], []] for k in "AB"}; Ys = []; checks = []
            for f in files:
                kk = int(re.search(r"r(\d+)\.npz$", f).group(1))
                z = np.load(os.path.join(D, "twins", f"d{d}", "truths", f"r{kk:03d}.npz"))
                t = np.rint(z["t_test"]).astype(int)
                y0, y1 = z["y0"].astype(float), z["y1"].astype(float)
                yf, ycf = np.where(t == 1, y1, y0), np.where(t == 1, y0, y1)
                E, W, T, ch = cond_hist(f, tag, t, yf)
                if ch is not None:
                    checks.append(ch)
                for k, v in readouts(E, W, T).items():
                    for j in range(3):
                        acc[k][j].append(v[j])
                Ys.append(ycf)
            Y = np.concatenate(Ys)
            if checks:
                c = np.array(checks).max(0)
                print(f"  axis check {label}: |rows - p_y0| = {c[0]:.2e}, |cols - p_y0| = {c[1]:.2e} "
                      f"-> stored as {'[lighter, heavier]' if c[0] < c[1] else '[heavier, lighter]  <-- SWAPPED'}", flush=True)
            for k in "AB":
                r = dict(readout=k, setting=f"d={d}", model=label, n=Y.size,
                         **metrics(*(np.concatenate(acc[k][j]) for j in range(3)), Y))
                recs.append(r)
            print(f"d={d:2d} {label:34s} " + "  ".join(
                f"{r['readout']}: MSE {r['mse']:.4f} IS {r['IS']:.3f}" for r in recs[-2:]), flush=True)
    with open(out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)
    names = {"A": "A. Bodik's: histogram mean; 2.5%/97.5% quantiles on bin edges",
             "B": "B. nearest outcome (split at 1/2)"}
    md = ["# Twins, Bodik's task: two read-outs of the predicted histogram of the other twin", "",
          "Predict the other twin from x and the known twin (2D: joint conditioned on the known twin; 1D: the other "
          "arm's marginal). MSE, |ĉ − 0.95|, length, IS₀.₀₅ of the 95% interval. Lower is better.", ""]
    for k in "AB":
        md += [f"## {names[k]}", "", "| setting | model | MSE | |ĉ − 0.95| | ĉ | length | IS₀.₀₅ |", "|---|---|---|---|---|---|---|"]
        for r in recs:
            if r["readout"] == k:
                md.append(f"| {r['setting']} | {r['model']} | {r['mse']:.4f} | {r['abs_cov_err']:.3f} | {r['coverage']:.3f} | "
                          f"{r['length']:.3f} | {r['IS']:.3f} |")
        md.append("")
    open(out + ".md", "w").write("\n".join(md) + "\n")
    print(f"wrote {out}.csv / .md")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
