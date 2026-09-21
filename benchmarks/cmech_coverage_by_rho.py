#!/usr/bin/env python
"""Does ComplexMech coverage improve on realizations where the arms are strongly
coupled? One table per model: coverage over all realizations, and restricted to
rho > threshold.

WHY THIS IS THE RIGHT TEST. ComplexMech is the only benchmark whose coverage
target carries noise: generate_pehe_benchmark writes

    true_cate = Y_do1 - Y_do0          (difference of NOISY paired outcomes)

while the case studies and RealCause score against the noiseless mu_1 - mu_0. A
head that predicts the CATE therefore has to cover a NOISY tau on ComplexMech and
is short by whatever noise fails to cancel between the two passes. How much fails
to cancel is exactly what the arm-noise correlation measures: at rho = 1 the noise
cancels and the target collapses onto the noiseless CATE; at rho = 0 none of it
cancels. So if the undercoverage is a target mismatch rather than a miscalibrated
joint head, coverage MUST rise with rho. If coverage is flat in rho, the target is
not the explanation and the head itself is too narrow.

ALIGNMENT. Per-realization coverage is not read from the perreal dumps -- those
store values without file identity, and UWYKFig34Dataset SKIPS realizations whose
subset is empty (`n_skipped`), so position in a raw glob is not the realization
index. Each dump file is scored here directly and mapped to its data file through
the dataset's own `_paths`, which is the only correct index -> file map.

rho is estimated per realization by residualising each arm on X (`ols-on-X`), on
the SAME subset mask the coverage uses. It is an upper bound on |rho|: any
nonlinearity the linear fit misses stays in the residual. That direction is
harmless here -- it can only blur the contrast between buckets, not manufacture one.

    python benchmarks/cmech_coverage_by_rho.py --root $SCRATCH/cmech_dumps/<model> \
        --nodes 5 10 20 --out rho_buckets.md
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "UWYK_Fig3_4"))

from cate_density_metrics import (METHODS, score_file, _resolve_dir)   # noqa: E402
from uwyk_fig34_dataset import UWYKFig34Dataset                        # noqa: E402


def _ols_resid(X, y):
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y, float).ravel()
    A = np.hstack([np.ones((X.shape[0], 1)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def _rho_for(path, mask):
    """corr of the two arms' residuals on the masked rows of one data file."""
    try:
        z = np.load(path, allow_pickle=True)
    except Exception:
        return float("nan")
    keys = set(z.files)
    if not {"Y_do0", "Y_do1", "X_test"} <= keys:
        return float("nan")
    y0 = np.asarray(z["Y_do0"], float).ravel()
    y1 = np.asarray(z["Y_do1"], float).ravel()
    X = np.asarray(z["X_test"], float)
    if mask is not None and mask.shape[0] == y0.size:
        y0, y1, X = y0[mask], y1[mask], X[mask]
    if y0.size < 8 or X.shape[0] != y0.size:
        return float("nan")
    e0, e1 = _ols_resid(X, y0), _ols_resid(X, y1)
    if e0.std() <= 0 or e1.std() <= 0:
        return float("nan")
    return float(np.corrcoef(e0, e1)[0, 1])


def _bucket(recs, lo=None, hi=None):
    """(n, mean coverage, mean length) over records with lo < rho <= hi."""
    sel = [(c, l) for c, l, r in recs
           if np.isfinite(r) and (lo is None or r > lo) and (hi is None or r <= hi)]
    if not sel:
        return 0, float("nan"), float("nan")
    c = np.asarray([s[0] for s in sel]); l = np.asarray([s[1] for s in sel])
    return len(sel), float(c.mean()), float(l.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", nargs="+", required=True,
                    help="model dump root(s), e.g. $SCRATCH/cmech_dumps/<model>")
    ap.add_argument("--nodes", type=int, nargs="+", default=[5, 10, 20, 30, 40, 50])
    ap.add_argument("--subset", default="total", choices=["total", "nonzero", "zero"])
    ap.add_argument("--context", type=int, default=1000)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.8, 0.9])
    ap.add_argument("--data-root", default=os.environ.get("UWYK_FIG34_DATA", ""))
    ap.add_argument("--regime", default="path_TY")
    ap.add_argument("--hide", type=float, default=0.0)
    ap.add_argument("--coupling", default="indep")
    ap.add_argument("--joint-coupling", default="learned")
    ap.add_argument("--max-real", type=int, default=0, help="0 = all")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ths = sorted(a.thresholds)
    L = ["## ComplexMech coverage vs arm-noise correlation", "",
         "| model | n | realizations | Cov (all) | Len (all) "
         + "".join(f"| N(rho>{t}) | Cov (rho>{t}) | Len (rho>{t}) " for t in ths)
         + f"| N(rho<={ths[0]}) | Cov (rho<={ths[0]}) | corr(cov, rho) |",
         "|" + "---|" * (6 + 3 * len(ths) + 2)]
    _ds_cache: dict = {}

    for root in a.root:
        for label, subdir, tag in METHODS:
            if a.methods and label not in a.methods:
                continue
            for n in a.nodes:
                subs = (["nonzero", "zero"] if a.subset == "total" else [a.subset])
                recs = []
                for sub in subs:
                    d = _resolve_dir(os.path.join(root, f"N{a.context}"), subdir,
                                     f"CMECH_n{n}_{sub}")
                    if not d:
                        continue
                    key = (n, sub)
                    if key not in _ds_cache:
                        try:
                            _ds_cache[key] = UWYKFig34Dataset(
                                f"CMECH_n{n}_{sub}",
                                data_root=a.data_root or None,
                                prior="complexmech", regime=a.regime, hide=a.hide)
                        except Exception as e:
                            print(f"note: n={n} {sub}: {type(e).__name__}: {e}",
                                  file=sys.stderr)
                            _ds_cache[key] = None
                    ds = _ds_cache[key]
                    if ds is None:
                        continue
                    files = [f for f in sorted(glob.glob(os.path.join(d, "*.npz")))
                             if os.path.basename(f) != "summary.npz"]
                    if a.max_real:
                        files = files[: a.max_real]
                    for f in files:
                        m = re.search(r"r(\d+)", os.path.basename(f))
                        if not m:
                            continue
                        idx = int(m.group(1))
                        if idx >= len(ds._paths):
                            print(f"note: {label} n={n} {sub}: dump r{idx} has no "
                                  f"data file (dataset has {len(ds._paths)})",
                                  file=sys.stderr)
                            continue
                        got = score_file(f, tag, a.coupling, a.joint_coupling)
                        if not got:
                            continue
                        cov = [g["cover"] for g in got if g.get("cover") is not None]
                        ln = [g["length"] for g in got if g.get("length") is not None]
                        if not cov:
                            continue
                        dpath = ds._paths[idx]
                        with np.load(dpath) as z:
                            mask = ds._mask_for(np.asarray(z["true_cate"]).ravel())
                        recs.append((float(np.mean(cov)),
                                     float(np.mean(ln)) if ln else float("nan"),
                                     _rho_for(dpath, mask)))
                if not recs:
                    continue
                nn, call, lall = _bucket(recs)
                cells = [f"{os.path.basename(root.rstrip('/'))}/{label}", str(n),
                         str(nn), f"{call:.4f}", f"{lall:.4f}"]
                for t in ths:
                    k, c, l = _bucket(recs, lo=t)
                    cells += [str(k), "—" if not k else f"{c:.4f}",
                              "—" if not k else f"{l:.4f}"]
                k0, c0, _ = _bucket(recs, hi=ths[0])
                cells += [str(k0), "—" if not k0 else f"{c0:.4f}"]
                rr = np.asarray([r for _, _, r in recs], float)
                cc = np.asarray([c for c, _, r in recs], float)
                ok = np.isfinite(rr) & np.isfinite(cc)
                cells.append(f"{np.corrcoef(cc[ok], rr[ok])[0,1]:+.3f}"
                             if ok.sum() > 4 and rr[ok].std() > 0 else "—")
                L.append("| " + " | ".join(cells) + " |")

    L += ["",
          "ComplexMech scores coverage against `true_cate = Y_do1 - Y_do0`, a NOISY",
          "tau, while every other benchmark uses the noiseless `mu_1 - mu_0`. Noise",
          "cancels between the paired passes exactly when the arms are coupled, so a",
          "CATE-predicting head should cover BETTER at high rho. Coverage rising with",
          "rho (positive corr, higher Cov in the rho>t buckets) means the gap is the",
          "target definition; coverage flat in rho means the head is simply too narrow.",
          "",
          "rho is `ols-on-X` and so an upper bound on |rho|; that can blur the",
          "contrast between buckets but cannot create one."]
    txt = "\n".join(L)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt + "\n")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
