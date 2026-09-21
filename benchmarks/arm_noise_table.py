#!/usr/bin/env python
"""One table of the arm-noise correlation across every benchmark, with a test of
rho = 0.

WHAT IS CORRELATED. Not Y(0) against Y(1) -- that correlation is dominated by the
shared conditional mean and is large even when the noise is independent. The
quantity is the NOISE:

    e0 = Y(0) - E[Y(0)|X]        e1 = Y(1) - E[Y(1)|X]

and rho = corr(e0, e1). This is what decides whether a joint head has dependence to
learn: Var(tau) = s0^2 + s1^2 - 2 rho s0 s1, so rho = 0 leaves the convolution of
two marginals correct, and rho = 1 makes tau deterministic given X.

HOW E[Y|X] IS OBTAINED differs by benchmark, and the table says which was used --
they are not equally strong:

  mu          the generator stores the noiseless means (IHDP, ACIC). Exact.
  within-unit RealCause draws each unit K times; the within-unit mean estimates
              E[Y|X] without a model. Exact up to 1/K.
  ols-on-X    only one draw per unit and no stored mean, so E[Y|X] is estimated by
              regressing each arm on X. A PROXY: any nonlinearity it misses stays in
              the residual and inflates |rho|. Treat these rows as upper bounds.

THE TEST. Per realization, Fisher z = atanh(r) with SE = 1/sqrt(n-3), then combined
across realizations by inverse-variance weighting. The reported p is two-sided
against z = 0. With rho near 1 the z transform is numerically extreme, which is
itself the answer -- those rows are reported as rho > 0.999 rather than a p-value
that underflows.

    python benchmarks/arm_noise_table.py --out arm_noise.md
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _ols_resid(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """y - Xb with an intercept, via lstsq. Estimates E[y|X] linearly."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    A = np.hstack([np.ones((X.shape[0], 1)), X])
    beta, *_ = np.linalg.lstsq(A, np.asarray(y, dtype=float).ravel(), rcond=None)
    return np.asarray(y, dtype=float).ravel() - A @ beta


def _rho(e0, e1):
    e0, e1 = np.asarray(e0, float).ravel(), np.asarray(e1, float).ravel()
    ok = np.isfinite(e0) & np.isfinite(e1)
    e0, e1 = e0[ok], e1[ok]
    if e0.size < 5 or e0.std() == 0 or e1.std() == 0:
        return float("nan"), e0.size
    return float(np.corrcoef(e0, e1)[0, 1]), int(e0.size)


def _combine(rs, ns):
    """Fisher-z combine per-realization correlations -> (rho, lo, hi, p, n_real)."""
    zs, ws = [], []
    for r, n in zip(rs, ns):
        if not np.isfinite(r) or n < 5:
            continue
        r = min(max(r, -0.999999), 0.999999)
        zs.append(math.atanh(r)); ws.append(max(n - 3, 1))
    if not zs:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0
    zs, ws = np.asarray(zs), np.asarray(ws, dtype=float)
    z = float(np.sum(ws * zs) / np.sum(ws))
    se = float(1.0 / math.sqrt(np.sum(ws)))
    lo, hi = math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)
    # two-sided normal p for z/se
    zz = abs(z) / se
    p = math.erfc(zz / math.sqrt(2.0))
    return math.tanh(z), lo, hi, p, len(zs)


def _from_npz(paths, k0, k1, xk, max_files=None):
    """Per-file rho from stored both-arm outcomes, residualised on X."""
    rs, ns, used = [], [], "ols-on-X"
    for f in (paths[:max_files] if max_files else paths):
        try:
            z = np.load(f, allow_pickle=True)
        except Exception:
            continue
        keys = set(z.files)
        if not ({k0, k1} <= keys):
            continue
        y0 = np.asarray(z[k0], float).ravel()
        y1 = np.asarray(z[k1], float).ravel()
        # Prefer stored noiseless means if the generator kept them.
        if {"mu0", "mu1"} <= keys:
            e0 = y0 - np.asarray(z["mu0"], float).ravel()
            e1 = y1 - np.asarray(z["mu1"], float).ravel()
            used = "mu"
        elif xk in keys:
            X = np.asarray(z[xk], float)
            if X.shape[0] != y0.size:
                continue
            e0, e1 = _ols_resid(X, y0), _ols_resid(X, y1)
        else:
            continue
        r, n = _rho(e0, e1)
        if np.isfinite(r):
            rs.append(r); ns.append(n)
    return rs, ns, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", default=os.environ.get("SCRATCH", ""))
    ap.add_argument("--causalpfn", default=os.environ.get("CAUSALPFN", ""))
    ap.add_argument("--max-files", type=int, default=25)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    SC = a.scratch
    rows = []

    # ── RealCause: reuse the existing, exact estimators ──────────────────────
    sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "realcause_eval"))
    try:
        import plot_arm_noise_independence as P
    except Exception as e:
        print(f"note: RealCause estimators unavailable ({type(e).__name__}); "
              f"those rows will be skipped", file=sys.stderr)
        P = None
    if P is not None and a.causalpfn:
        for name, fn in (("IHDP", "ihdp_residuals"), ("ACIC", "acic_residuals")):
            f = getattr(P, fn, None)
            if f is None:
                continue
            try:
                got = f(a.causalpfn) if name == "IHDP" else f(a.causalpfn)
                e0, e1 = got[0], got[1]
                r, n = _rho(e0, e1)
                rows.append((f"RealCause / {name}", r, *(_combine([r], [n])[1:4]),
                             1, n, "mu"))
            except Exception as e:
                print(f"note: {name}: {type(e).__name__}: {e}", file=sys.stderr)
        try:
            csv_dir = P.find_csv_dir(a.causalpfn)
            for name, paths in P.discover(csv_dir).items():
                e0, e1, r, n_unit, K = P.residuals(paths, max_files=a.max_files)
                r2, n = _rho(e0, e1)
                rows.append((f"RealCause / {name}", r2,
                             *(_combine([r2], [n])[1:4]), 1, n, "within-unit"))
        except Exception as e:
            print(f"note: CPS/PSID: {type(e).__name__}: {e}", file=sys.stderr)

    # ── ComplexMech: one column, pooled over node counts ────────────────────
    cm = sorted(glob.glob(os.path.join(SC, "cmech_data_v2", "complexmech",
                                       "*node", "*", "*", "r*.npz")))
    if cm:
        rs, ns, used = _from_npz(cm, "Y_do0", "Y_do1", "X_test", a.max_files)
        rho, lo, hi, p, k = _combine(rs, ns)
        rows.append(("ComplexMech (all n, pooled)", rho, lo, hi, p, k,
                     int(np.sum(ns)) if ns else 0, used))

    # ── Case studies: one column, pooled over shift x d x case ──────────────
    cs = sorted(glob.glob(os.path.join(
        os.path.dirname(_HERE), "case_study_data", "d_variation",
        "shift*", "d*", "*", "N1000", "*.npz")))
    if not cs:
        cs = sorted(glob.glob(os.path.join(SC, "case_study_data", "d_variation",
                                           "shift*", "d*", "*", "N1000", "*.npz")))
    if cs:
        for k0, k1 in (("Y_do0", "Y_do1"), ("y0", "y1")):
            rs, ns, used = _from_npz(cs, k0, k1, "X_test", a.max_files)
            if rs:
                rho, lo, hi, p, k = _combine(rs, ns)
                rows.append(("Case study (all cells, pooled)", rho, lo, hi, p, k,
                             int(np.sum(ns)), used))
                break
        else:
            rows.append(("Case study (all cells, pooled)", float("nan"),
                         float("nan"), float("nan"), float("nan"), 0, 0,
                         "both arms not stored"))

    # ── emit ────────────────────────────────────────────────────────────────
    L = ["## Arm-noise correlation  rho = corr(Y(0)-E[Y(0)|X], Y(1)-E[Y(1)|X])", "",
         "| dataset | rho | 95% CI | p (rho=0) | realizations | n | E[Y|X] from |",
         "|---|---|---|---|---|---|---|"]
    for name, rho, lo, hi, p, k, n, used in rows:
        if not np.isfinite(rho):
            L.append(f"| {name} | — | — | — | {k} | {n} | {used} |")
            continue
        if abs(rho) > 0.999:
            rs_, ps_ = f"**> {abs(rho):.4f}**", "&lt; 1e-300"
        else:
            rs_, ps_ = f"{rho:+.4f}", (f"{p:.3g}" if p >= 1e-300 else "&lt; 1e-300")
        L.append(f"| {name} | {rs_} | [{lo:+.4f}, {hi:+.4f}] | {ps_} | "
                 f"{k} | {n} | {used} |")
    L += ["",
          "`mu` / `within-unit` are exact. `ols-on-X` estimates E[Y|X] by linear",
          "regression because only one draw per unit is stored and no noiseless mean",
          "is kept; any nonlinearity it misses remains in the residual and inflates",
          "|rho|, so those rows are UPPER BOUNDS.",
          "",
          "Why it matters: Var(tau) = s0^2 + s1^2 - 2 rho s0 s1. At rho = 0 the",
          "convolution of two marginals is correct and a joint head has no dependence",
          "to learn; at rho = 1 tau is deterministic given X and no 95% interval for",
          "it is well posed."]
    txt = "\n".join(L)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt + "\n")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
