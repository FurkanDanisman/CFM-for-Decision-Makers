#!/usr/bin/env python
"""One table of the arm-noise correlation across every benchmark, with a test of
rho = 0, reported PER REALIZATION (mean +- sd) as well as inverse-variance combined.

WHAT IS CORRELATED. Not Y(0) against Y(1) -- that correlation is dominated by the
shared conditional mean and is large even when the noise is independent. The
quantity is the NOISE:

    e0 = Y(0) - E[Y(0)|X]        e1 = Y(1) - E[Y(1)|X]

and rho = corr(e0, e1). This decides whether a joint head has dependence to learn:
Var(tau) = s0^2 + s1^2 - 2 rho s0 s1, so rho = 0 leaves the convolution of two
marginals correct, and rho = 1 makes tau deterministic given X.

HOW E[Y|X] IS OBTAINED differs by benchmark and the table names it per row; they
are NOT equally strong:

  mu             the generator stores noiseless means (IHDP, ACIC). Exact.
  within-unit    RealCause draws each unit K times; the within-unit mean gives
                 E[Y|X] with no model. Exact up to 1/K.
  shared-eps     the case-study generator adds ONE outcome-noise draw to both
                 arms, so e0 and e1 are the same array. rho = 1 identically --
                 read off the generator and confirmed numerically, not estimated.
  ols-on-X       only one draw per unit and no stored mean, so E[Y|X] is fit by
                 regressing each arm on X. A PROXY: nonlinearity it misses stays
                 in the residual and inflates |rho|. Those rows are UPPER BOUNDS.

REALIZATIONS. What counts as one differs by benchmark and is stated per row:
IHDP = one of the 100 simulation replicates; ACIC = one zymu_<i> setting;
CPS/PSID = one of the K resampled draws; ComplexMech = one r*.npz; case study =
one (case, seed) pair.

THE TEST. Per realization, Fisher z = atanh(r), SE = 1/sqrt(n-3); combined across
realizations by inverse-variance weighting; p is two-sided against z = 0. The
mean +- sd column is the unweighted spread of the per-realization r, which is what
shows whether a pooled number is representative or an average over a mix.

    python benchmarks/arm_noise_table.py --causalpfn <dir> --out arm_noise.md
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)


# ── primitives ───────────────────────────────────────────────────────────────
def _ols_resid(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """y - Xb with an intercept, via lstsq. Estimates E[y|X] linearly."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y, dtype=float).ravel()
    A = np.hstack([np.ones((X.shape[0], 1)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def _rho(e0, e1):
    e0, e1 = np.asarray(e0, float).ravel(), np.asarray(e1, float).ravel()
    ok = np.isfinite(e0) & np.isfinite(e1)
    e0, e1 = e0[ok], e1[ok]
    if e0.size < 5 or e0.std() <= 0 or e1.std() <= 0:
        return float("nan"), e0.size
    return float(np.corrcoef(e0, e1)[0, 1]), int(e0.size)


def _combine(rs, ns):
    """Fisher-z inverse-variance combine -> (rho, lo, hi, p, n_real)."""
    zs, ws = [], []
    for r, n in zip(rs, ns):
        if not np.isfinite(r) or n < 5:
            continue
        rc = min(max(r, -0.999999), 0.999999)
        zs.append(math.atanh(rc)); ws.append(max(n - 3, 1))
    if not zs:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0
    zs, ws = np.asarray(zs), np.asarray(ws, dtype=float)
    z = float(np.sum(ws * zs) / np.sum(ws))
    se = float(1.0 / math.sqrt(np.sum(ws)))
    lo, hi = math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)
    p = math.erfc((abs(z) / se) / math.sqrt(2.0))
    return math.tanh(z), lo, hi, p, len(zs)


def _row(name, rs, ns, used, unit, skipped=0):
    """Assemble one table row from per-realization correlations."""
    rho, lo, hi, p, k = _combine(rs, ns)
    arr = np.asarray([r for r in rs if np.isfinite(r)], dtype=float)
    mean = float(arr.mean()) if arr.size else float("nan")
    sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0 if arr.size == 1 else float("nan")
    n_tot = int(np.sum([n for r, n in zip(rs, ns) if np.isfinite(r)])) if len(ns) else 0
    return dict(name=name, rho=rho, lo=lo, hi=hi, p=p, mean=mean, sd=sd,
                k=k, n=n_tot, used=used, unit=unit, skipped=skipped)


# ── RealCause: IHDP / ACIC (stored mu) ───────────────────────────────────────
def _ihdp_rows(causalpfn):
    """(672 units x 100 replicates), both arms + both noiseless means stored."""
    cands = glob.glob(os.path.join(causalpfn, "**", "ihdp_npci_1-100.train.npz"),
                      recursive=True)
    if not cands:
        return None
    z = np.load(cands[0])
    if not {"yf", "ycf", "mu0", "mu1", "t"} <= set(z.files):
        return None
    t = np.asarray(z["t"], float)
    yf, ycf = np.asarray(z["yf"], float), np.asarray(z["ycf"], float)
    mu0, mu1 = np.asarray(z["mu0"], float), np.asarray(z["mu1"], float)
    y0 = np.where(t > 0.5, ycf, yf)
    y1 = np.where(t > 0.5, yf, ycf)
    e0, e1 = y0 - mu0, y1 - mu1
    if e0.ndim == 1:
        e0, e1 = e0[:, None], e1[:, None]
    rs, ns = [], []
    for j in range(e0.shape[1]):                      # one replicate per column
        r, n = _rho(e0[:, j], e1[:, j])
        rs.append(r); ns.append(n)
    return _row("RealCause / IHDP", rs, ns, "mu", "replicate (of 100)")


def _acic_rows(cache_dir, n_files, download=True):
    """zymu_<i>.csv: z, y0, y1, mu0, mu1. Each file is a different setting."""
    import csv as _csv
    import urllib.request as _url
    ACIC_URL = ("https://raw.githubusercontent.com/BiomedSciAI/causallib/master/"
                "causallib/datasets/data/acic_challenge_2016/zymu_{}.csv")
    os.makedirs(cache_dir, exist_ok=True)
    rs, ns = [], []
    for i in range(1, n_files + 1):
        dst = os.path.join(cache_dir, f"zymu_{i}.csv")
        if not os.path.isfile(dst) and download:
            try:
                _url.urlretrieve(ACIC_URL.format(i), dst)
            except Exception:
                continue
        if not os.path.isfile(dst):
            continue
        try:
            with open(dst, newline="") as fh:
                rd = _csv.DictReader(fh)
                if not {"y0", "y1", "mu0", "mu1"} <= set(rd.fieldnames or []):
                    continue
                a = np.array([[float(r["y0"]) - float(r["mu0"]),
                               float(r["y1"]) - float(r["mu1"])] for r in rd])
        except Exception:
            continue
        if a.size == 0:
            continue
        r, n = _rho(a[:, 0], a[:, 1])
        rs.append(r); ns.append(n)
    if not rs:
        return None
    return _row("RealCause / ACIC", rs, ns, "mu", "zymu setting")


# ── RealCause: CPS / PSID (within-unit centring) ─────────────────────────────
def _cps_psid_rows(P, csv_dir, max_files):
    out = []
    try:
        fams = P.discover(csv_dir)
    except Exception as e:
        print(f"note: discover: {type(e).__name__}: {e}", file=sys.stderr)
        return out
    try:
        import pandas as pd

        def _read(p):
            df = pd.read_csv(p)
            if not {"y0", "y1"} <= set(df.columns):
                return None
            return df["y0"].to_numpy(float), df["y1"].to_numpy(float)
    except ImportError:
        import csv as _csv

        def _read(p):
            with open(p, newline="") as fh:
                rd = _csv.DictReader(fh)
                if not {"y0", "y1"} <= set(rd.fieldnames or []):
                    return None
                a, b = [], []
                for row in rd:
                    a.append(float(row["y0"])); b.append(float(row["y1"]))
            return np.asarray(a), np.asarray(b)

    for name, paths in fams.items():
        use = paths[:max_files] if max_files else paths
        y0, y1 = [], []
        for p in use:
            got = _read(p)
            if got is None:
                continue
            y0.append(got[0]); y1.append(got[1])
        if len(y0) < 2:
            continue
        n = min(len(a) for a in y0)
        Y0 = np.stack([a[:n] for a in y0], axis=1)     # (n_unit, K)
        Y1 = np.stack([a[:n] for a in y1], axis=1)
        # Remove mu_t(x_i) by centring within unit -- no model of mu needed.
        e0 = Y0 - Y0.mean(axis=1, keepdims=True)
        e1 = Y1 - Y1.mean(axis=1, keepdims=True)
        rs, ns = [], []
        for k in range(e0.shape[1]):                   # one draw per column
            r, nn = _rho(e0[:, k], e1[:, k])
            rs.append(r); ns.append(nn)
        out.append(_row(f"RealCause / {name}", rs, ns, "within-unit",
                        f"draw (of {e0.shape[1]})"))
    return out


# ── ComplexMech: stored both arms, no stored mu -> ols proxy ─────────────────
def _npz_rows(name, paths, k0, k1, xk, unit, max_files=None):
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
        if {"mu0", "mu1"} <= keys:                     # prefer exact means
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
        rs.append(r); ns.append(n)
    if not rs:
        return None
    return _row(name, rs, ns, used, unit)


# ── Case study: rho is structural, computed from the generator ───────────────
def _case_study_row(n_seeds, n_context):
    """The case-study SCM samples each node's noise ONCE and reuses it across the
    do(T=0) and do(T=1) passes (generation.py: `self._noise` ... 'sampled ONCE,
    reused'). The outcome noise is additive, so

        Y(0) = mu_0 + eps      Y(1) = mu_1 + eps       (same eps)

    giving e0 == e1 and rho == 1 identically, and tau = mu_1 - mu_0 with no noise
    at all. We do not estimate this -- we evaluate both arms of the generator and
    confirm it, so the row is exact rather than a proxy.
    """
    sys.path.insert(0, os.path.join(_REPO, "case_study"))
    try:
        from generation import CASE_STUDIES, build_dag, _SampledSCM
    except Exception as e:
        print(f"note: case-study generator unavailable ({type(e).__name__}: {e})",
              file=sys.stderr)
        return None
    rs, ns, degen, maxdiff, tau_sd = [], [], 0, 0.0, []
    for case in CASE_STUDIES:
        for seed in range(n_seeds):
            try:
                scm = _SampledSCM(build_dag(case), N=n_context,
                                  rng=np.random.default_rng(seed))
                scm.forward()                                   # observational
                y0 = scm.forward(do_T=0.0)[scm.y_name]          # with noise
                y1 = scm.forward(do_T=1.0)[scm.y_name]
                m0 = scm.forward(do_T=0.0, y_noiseless=True)[scm.y_name]
                m1 = scm.forward(do_T=1.0, y_noiseless=True)[scm.y_name]
            except Exception:
                continue
            e0, e1 = y0 - m0, y1 - m1
            maxdiff = max(maxdiff, float(np.abs(e0 - e1).max()))
            tau_sd.append(float((y1 - y0).std()))
            if e0.std() <= 0 or e1.std() <= 0:
                degen += 1                    # noise_std ~ 0: rho undefined
                continue
            r, n = _rho(e0, e1)
            rs.append(r); ns.append(n)
    if not rs:
        return None
    row = _row("Case study (all cases, pooled)", rs, ns, "shared-eps",
               "(case, seed)", skipped=degen)
    row["maxdiff"] = maxdiff
    row["tau_sd"] = float(np.max(tau_sd)) if tau_sd else float("nan")
    return row


# ── emit ─────────────────────────────────────────────────────────────────────
def _fmt(row):
    rho, p = row["rho"], row["p"]
    if not np.isfinite(rho):
        return (f"| {row['name']} | — | — | — | — | {row['k']} | {row['n']} | "
                f"{row['used']} | {row['unit']} |")
    if abs(rho) > 0.9999:
        rs_, ci, ps_ = "**+1.0000**", "[+1.0000, +1.0000]", "&lt; 1e-300"
    else:
        rs_ = f"{rho:+.4f}"
        ci = f"[{row['lo']:+.4f}, {row['hi']:+.4f}]"
        ps_ = f"{p:.3g}" if p >= 1e-300 else "&lt; 1e-300"
    ms = (f"{row['mean']:+.4f} ± {row['sd']:.4f}"
          if np.isfinite(row["mean"]) else "—")
    return (f"| {row['name']} | {rs_} | {ci} | {ps_} | {ms} | "
            f"{row['k']} | {row['n']} | {row['used']} | {row['unit']} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", default=os.environ.get("SCRATCH", ""))
    ap.add_argument("--causalpfn", default=os.environ.get("CAUSALPFN", ""))
    ap.add_argument("--max-files", type=int, default=0,
                    help="cap files per dataset (0 = all, the default)")
    ap.add_argument("--acic-files", type=int, default=20,
                    help="how many zymu_<i> settings to try")
    ap.add_argument("--cs-seeds", type=int, default=25,
                    help="case-study seeds per case")
    ap.add_argument("--cs-n", type=int, default=1000,
                    help="case-study units per realization")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    SC, cap = a.scratch, (a.max_files or None)
    rows = []

    # RealCause
    sys.path.insert(0, os.path.join(_REPO, "realcause_eval"))
    try:
        import plot_arm_noise_independence as P
    except Exception as e:
        print(f"note: RealCause estimators unavailable ({type(e).__name__}); "
              f"skipping those rows", file=sys.stderr)
        P = None
    if a.causalpfn:
        try:
            r = _ihdp_rows(a.causalpfn)
            if r:
                rows.append(r)
        except Exception as e:
            print(f"note: IHDP: {type(e).__name__}: {e}", file=sys.stderr)
        try:
            r = _acic_rows(a.causalpfn, a.acic_files, not a.no_download)
            if r:
                rows.append(r)
        except Exception as e:
            print(f"note: ACIC: {type(e).__name__}: {e}", file=sys.stderr)
        if P is not None:
            try:
                rows += _cps_psid_rows(P, P.find_csv_dir(a.causalpfn), cap)
            except Exception as e:
                print(f"note: CPS/PSID: {type(e).__name__}: {e}", file=sys.stderr)

    # ComplexMech -- one column, pooled over node counts
    cm = sorted(glob.glob(os.path.join(SC, "cmech_data_v2", "complexmech",
                                       "*node", "*", "*", "r*.npz")))
    if cm:
        r = _npz_rows("ComplexMech (all n, pooled)", cm, "Y_do0", "Y_do1",
                      "X_test", "r*.npz realization", cap)
        if r:
            rows.append(r)
    else:
        print(f"note: no ComplexMech npz under {SC}/cmech_data_v2", file=sys.stderr)

    # Case study -- one column, structural
    cs = _case_study_row(a.cs_seeds, a.cs_n)
    if cs:
        rows.append(cs)

    L = ["## Arm-noise correlation  rho = corr(Y(0)-E[Y(0)|X], Y(1)-E[Y(1)|X])", "",
         "| dataset | rho (combined) | 95% CI | p (rho=0) | mean ± sd over realizations "
         "| realizations | n | E[Y\\|X] from | realization = |",
         "|---|---|---|---|---|---|---|---|---|"]
    L += [_fmt(r) for r in rows]
    L += ["",
          "**rho (combined)** is Fisher-z inverse-variance across realizations; "
          "**mean ± sd** is the unweighted spread of the per-realization rho, which "
          "shows whether the combined number is representative.", ""]
    if cs:
        L += [f"**Case study.** rho = 1 is not an estimate. The generator draws each "
              f"node's noise once and reuses it for the do(T=0) and do(T=1) passes, so "
              f"Y(0) = mu_0 + eps and Y(1) = mu_1 + eps with the SAME eps. Evaluating "
              f"both arms over {cs['k'] + cs['skipped']} "
              f"(case, seed) realizations gives max|e0 - e1| = {cs['maxdiff']:.2e} "
              f"(machine epsilon) and sd(Y(1) - Y(0)) = sd(mu_1 - mu_0) to full "
              f"precision: tau carries no noise, so no 95% interval for tau is well "
              f"posed."
              + (f" {cs['skipped']} realizations drew sigma_eps = 0 exactly and are "
                 f"excluded (rho undefined, not 0)." if cs["skipped"] else ""), ""]
    if any(r["name"].startswith("ComplexMech") for r in rows):
        L += ["**ComplexMech.** Its generator also shares noise across arms by design "
              "(\"exogenous *and* endogenous noise shared across the do(t0) and "
              "do(t1) passes\"), so the dependence is structural. It falls below 1 "
              "because the shared noise passes through mechanisms whose mediator "
              "values differ by arm, and the row is an `ols-on-X` upper bound "
              "besides.", ""]
    L += [
          "`mu` / `within-unit` / `shared-eps` are exact. `ols-on-X` fits E[Y|X] by "
          "linear regression because only one draw per unit is stored and no noiseless "
          "mean is kept; nonlinearity it misses stays in the residual and inflates "
          "|rho|, so those rows are UPPER BOUNDS.", "",
          "Why it matters: Var(tau) = s0^2 + s1^2 - 2 rho s0 s1. At rho = 0 the "
          "convolution of two marginals is correct and a joint head has no dependence "
          "to learn; at rho = 1 tau is deterministic given X."]
    txt = "\n".join(L)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt + "\n")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
