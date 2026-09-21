#!/usr/bin/env python
"""Per-dataset outcome spread sd(Y), the constant that turns an interval length in
raw outcome units into a length a reader can interpret.

WHY. `Len` is in the outcome's own units, so 1.8 on one dataset and 0.4 on another
say nothing about which model is tighter -- the datasets are on different scales.
Dividing by one constant per dataset puts every length in units of outcome spread.

WHY sd(Y) AND NOT sd(tau). The spread of the true ITEs is the tempting choice and
it does not work: on the case studies the generator shares one noise draw across
arms, so tau = mu_1 - mu_0 carries no noise and 42 of 150 realizations have
sd(tau) < 1e-6 (four are exactly 0). That normaliser divides by zero on a quarter
of the table. sd(Y) is never near zero (min 0.0136 over the same realizations).

DEFINITION. Per realization, the SD of the POOLED potential outcomes [Y(0); Y(1)];
then the mean of those per-realization SDs. Pooling both arms is what makes the
constant comparable across benchmarks that store different things, and averaging
per-realization SDs (rather than one SD over everything) keeps between-realization
mean shifts out of the constant -- the lengths it divides are per-realization too.
Where only the factual outcome exists the source column says so.

HOW TO READ THE RESULT. For a calibrated 95% interval Len = 3.92 * sd(tau|X), so
Len/sd(Y) = 3.92 * sd(tau|X)/sd(Y). Independent arms with equal noise give
sd(tau|X) = sqrt(2)*sigma, and sigma/sd(Y) <= 1 always because
Var(Y) = Var(mu(X)) + sigma^2. So 5.54 is a HARD CEILING: no calibrated model can
exceed it on any dataset, and a value above it is over-dispersion.

    python benchmarks/length_normalizers.py --causalpfn <dir> --out norm.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_KIT = os.path.dirname(_REPO)          # deploy root: sibling dirs hold the data


def _sd_pooled(y0, y1=None):
    """SD of the pooled potential outcomes for ONE realization."""
    v = np.asarray(y0, float).ravel() if y1 is None else np.concatenate(
        [np.asarray(y0, float).ravel(), np.asarray(y1, float).ravel()])
    v = v[np.isfinite(v)]
    return float(v.std(ddof=1)) if v.size > 1 else float("nan")


def _agg(sds):
    """Mean of per-realization SDs (+ spread, for the report)."""
    a = np.asarray([s for s in sds if np.isfinite(s) and s > 0], float)
    if not a.size:
        return None
    return dict(sd=float(a.mean()), spread=float(a.std(ddof=1)) if a.size > 1 else 0.0,
                n_real=int(a.size), lo=float(a.min()), hi=float(a.max()))


# ── RealCause ────────────────────────────────────────────────────────────────
def _ihdp(causalpfn):
    c = glob.glob(os.path.join(causalpfn, "**", "ihdp_npci_1-100.train.npz"),
                  recursive=True)
    if not c:
        return None
    z = np.load(c[0])
    if not {"yf", "ycf", "t"} <= set(z.files):
        return None
    t = np.asarray(z["t"], float)
    yf, ycf = np.asarray(z["yf"], float), np.asarray(z["ycf"], float)
    y0 = np.where(t > 0.5, ycf, yf)
    y1 = np.where(t > 0.5, yf, ycf)
    if y0.ndim == 1:
        y0, y1 = y0[:, None], y1[:, None]
    return _agg([_sd_pooled(y0[:, j], y1[:, j]) for j in range(y0.shape[1])])


def _acic(cache_dir, n_files):
    import csv as _csv
    sds = []
    for i in range(1, n_files + 1):
        p = os.path.join(cache_dir, f"zymu_{i}.csv")
        if not os.path.isfile(p):
            continue
        try:
            with open(p, newline="") as fh:
                rd = _csv.DictReader(fh)
                if not {"y0", "y1"} <= set(rd.fieldnames or []):
                    continue
                a = np.array([[float(r["y0"]), float(r["y1"])] for r in rd])
        except Exception:
            continue
        if a.size:
            sds.append(_sd_pooled(a[:, 0], a[:, 1]))
    return _agg(sds)


def _lalonde(P, causalpfn, max_files):
    out = {}
    try:
        fams = P.discover(P.find_csv_dir(causalpfn))
    except Exception as e:
        print(f"note: lalonde discover: {type(e).__name__}: {e}", file=sys.stderr)
        return out
    try:
        import pandas as pd
        rd = lambda p: pd.read_csv(p)[["y0", "y1"]].to_numpy(float)
    except ImportError:
        import csv as _csv

        def rd(p):
            with open(p, newline="") as fh:
                return np.array([[float(r["y0"]), float(r["y1"])]
                                 for r in _csv.DictReader(fh)])
    for fam, paths in fams.items():
        sds = []
        for p in (paths[:max_files] if max_files else paths):
            try:
                a = rd(p)
            except Exception:
                continue
            if a.size:
                sds.append(_sd_pooled(a[:, 0], a[:, 1]))
        g = _agg(sds)
        if not g:
            continue
        key = ("CPS" if "cps" in fam.lower() else
               "PSID" if "psid" in fam.lower() else fam)
        out[key] = g
    return out


# ── ComplexMech: both arms stored ────────────────────────────────────────────
def _cmech(scratch, max_files):
    per = {}
    files = sorted(glob.glob(os.path.join(scratch, "cmech_data_v2", "complexmech",
                                          "*node", "*", "*", "r*.npz")))
    for f in files:
        m = re.search(r"(\d+)node", f)
        if not m:
            continue
        per.setdefault(m.group(1), []).append(f)
    out = {}
    for n, fs in per.items():
        sds = []
        for f in (fs[:max_files] if max_files else fs):
            try:
                z = np.load(f, allow_pickle=True)
            except Exception:
                continue
            if not {"Y_do0", "Y_do1"} <= set(z.files):
                continue
            sds.append(_sd_pooled(z["Y_do0"], z["Y_do1"]))
        g = _agg(sds)
        if g:
            out[n] = g
    return out


# ── Case study: reconstruct both arms from the shared noise ──────────────────
def _find_cs_root(explicit):
    cands = [explicit, os.environ.get("CASE_STUDY_DATA", ""),
             os.path.join(_KIT, "case_study_data", "d_variation"),
             os.path.join(os.environ.get("SCRATCH", ""), "case_study_data",
                          "d_variation"),
             os.path.join(_REPO, "case_study", "data")]
    for c in cands:
        if c and os.path.isdir(c) and glob.glob(
                os.path.join(c, "shift*", "d*", "*", "N*", "*.npz")):
            return c
    return None


def _case_study_npz(root, max_files):
    """Y(0)=mu_0+eps, Y(1)=mu_1+eps with eps recovered from the factual Y. Exact,
    because the generator adds the SAME eps to both arms."""
    per = {}
    for f in sorted(glob.glob(os.path.join(root, "shift*", "d*", "*", "N*",
                                           "*.npz"))):
        case = os.path.basename(os.path.dirname(os.path.dirname(f)))
        per.setdefault(case, []).append(f)
    out = {}
    for case, fs in per.items():
        sds = []
        for f in (fs[:max_files] if max_files else fs):
            try:
                z = np.load(f, allow_pickle=True)
            except Exception:
                continue
            if not {"T", "Y", "mu_0", "mu_1"} <= set(z.files):
                continue
            T = np.asarray(z["T"], float).ravel()
            Y = np.asarray(z["Y"], float).ravel()
            m0 = np.asarray(z["mu_0"], float).ravel()
            m1 = np.asarray(z["mu_1"], float).ravel()
            eps = Y - np.where(T > 0.5, m1, m0)
            sds.append(_sd_pooled(m0 + eps, m1 + eps))
        g = _agg(sds)
        if g:
            out[case] = g
    return out


def _case_study_gen(n_seeds, n_ctx):
    """Fallback when the npz cells are not on this filesystem."""
    sys.path.insert(0, os.path.join(_REPO, "case_study"))
    try:
        from generation import CASE_STUDIES, build_dag, _SampledSCM
    except Exception as e:
        print(f"note: case-study generator unavailable ({type(e).__name__}: {e})",
              file=sys.stderr)
        return {}
    out = {}
    for case in CASE_STUDIES:
        sds = []
        for seed in range(n_seeds):
            try:
                scm = _SampledSCM(build_dag(case), N=n_ctx,
                                  rng=np.random.default_rng(seed))
                scm.forward()
                y0 = scm.forward(do_T=0.0)[scm.y_name]
                y1 = scm.forward(do_T=1.0)[scm.y_name]
            except Exception:
                continue
            sds.append(_sd_pooled(y0, y1))
        g = _agg(sds)
        if g:
            out[case] = g
    return out


# ── Do-PFN semi-real: factual outcome only ───────────────────────────────────
def _semireal(max_files):
    out = {}
    sys.path.insert(0, _HERE)
    try:
        from dopfn_semireal_dataset import DoPFNSemiRealDataset as DS
    except Exception as e:
        print(f"note: semi-real adapter unavailable ({type(e).__name__}: {e})",
              file=sys.stderr)
        return out
    for ds_name in ("sales", "law_race"):
        try:
            ds = DS(f"SEMIREAL_{ds_name}")
            n = ds.n_tables
        except Exception as e:
            print(f"note: semi-real {ds_name}: {type(e).__name__}: {e}",
                  file=sys.stderr)
            continue
        sds = []
        for r in range(min(n, max_files) if max_files else n):
            try:
                cate_ds, _ = ds[r]
                y = np.asarray(getattr(cate_ds, "y_train"), float).ravel()
            except Exception:
                continue
            if y.size:
                sds.append(_sd_pooled(y))
        g = _agg(sds)
        if g:
            out[ds_name] = g
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", default=os.environ.get("SCRATCH", ""))
    ap.add_argument("--causalpfn", default=os.environ.get("CAUSALPFN", ""))
    ap.add_argument("--case-study-root", default="")
    ap.add_argument("--max-files", type=int, default=0, help="0 = all")
    ap.add_argument("--acic-files", type=int, default=20)
    ap.add_argument("--cs-seeds", type=int, default=25)
    ap.add_argument("--cs-n", type=int, default=1000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cap = a.max_files or None
    res = {"realcause": {}, "case_study": {}, "cmech": {}, "semireal": {}}
    src = {}

    if a.causalpfn:
        g = _ihdp(a.causalpfn)
        if g:
            res["realcause"]["IHDP"] = g; src["IHDP"] = "both arms"
        g = _acic(a.causalpfn, a.acic_files)
        if g:
            res["realcause"]["ACIC"] = g; src["ACIC"] = "both arms"
        sys.path.insert(0, os.path.join(_REPO, "realcause_eval"))
        try:
            import plot_arm_noise_independence as P
            for k, g in _lalonde(P, a.causalpfn, cap).items():
                res["realcause"][k] = g; src[k] = "both arms"
        except Exception as e:
            print(f"note: lalonde: {type(e).__name__}: {e}", file=sys.stderr)
    # PSID_bal is a seed-42 balance subsample of the SAME outcomes, so it shares
    # PSID's constant rather than getting one fitted to a different unit mix.
    if "PSID" in res["realcause"]:
        res["realcause"]["PSID_bal"] = dict(res["realcause"]["PSID"])
        src["PSID_bal"] = "both arms (shared with PSID)"

    res["cmech"] = _cmech(a.scratch, cap)
    for k in res["cmech"]:
        src[f"cmech n={k}"] = "both arms"

    csr = _find_cs_root(a.case_study_root)
    if csr:
        res["case_study"] = _case_study_npz(csr, cap)
        for k in res["case_study"]:
            src[k] = "both arms (mu_t + shared eps)"
    if not res["case_study"]:
        print("note: no case-study npz found; using the generator", file=sys.stderr)
        res["case_study"] = _case_study_gen(a.cs_seeds, a.cs_n)
        for k in res["case_study"]:
            src[k] = "both arms (generator)"

    res["semireal"] = _semireal(cap)
    for k in res["semireal"]:
        src[k] = "factual Y only"

    print("## Length normalisers  sd(Y) = mean over realizations of "
          "sd([Y(0); Y(1)])\n")
    print("| group | dataset | sd(Y) | sd spread | min | max | realizations "
          "| outcome source |")
    print("|---|---|---|---|---|---|---|---|")
    for grp, d in res.items():
        for k in sorted(d, key=lambda s: (len(s), s)):
            g = d[k]
            print(f"| {grp} | {k} | {g['sd']:.4f} | {g['spread']:.4f} | "
                  f"{g['lo']:.4f} | {g['hi']:.4f} | {g['n_real']} | "
                  f"{src.get(k, src.get(f'cmech n={k}', '—'))} |")
    print("\nLen/sd(Y) has a HARD CEILING of 5.54 for a calibrated 95% interval: "
          "independent\narms give sd(tau|X) = sqrt(2)*sigma and sigma/sd(Y) <= 1, "
          "since Var(Y) = Var(mu) + sigma^2.\nAnything above it is over-dispersed.")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({k: {kk: vv["sd"] for kk, vv in v.items()}
                       for k, v in res.items()}, fh, indent=2, sort_keys=True)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
