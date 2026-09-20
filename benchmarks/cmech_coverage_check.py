#!/usr/bin/env python
"""Model-free check that a complexmech dataset can yield non-binary coverage.

Reads true_cate from the generated files and asks what coverage an ORACLE interval
would get per realization. Two oracles, and the contrast between them is the point:

  fixed    one half-width for the whole cell, tuned so the median realization hits
           nominal 95%. This is what a model emitting roughly constant-width
           intervals does, and it is the pessimistic case.
  adaptive half-width tuned per realization. This is what a calibrated conditional
           density should do.

If the fixed oracle is bimodal at 0 and 1 while the adaptive one sits at 0.95, the
binary coverage is a property of the data's spread in tau, not of the metric. If
the fixed oracle is well behaved, the data is fine and binary coverage seen in
practice is the models' doing.

    python benchmarks/cmech_coverage_check.py --cells DIR [DIR ...] --labels A B
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

Z = 1.959964


def load(cell):
    out = []
    for f in sorted(glob.glob(os.path.join(cell, "r*.npz"))):
        try:
            out.append(np.asarray(np.load(f)["true_cate"], dtype=float).ravel())
        except Exception:
            continue
    return out


def report(label, taus):
    if not taus:
        print(f"{label}: no realizations found")
        return
    sds = np.array([t.std() for t in taus])
    # fixed: one h for the cell, from the median realization's spread
    h_fixed = Z * float(np.median(sds))
    cov_fix, cov_ada = [], []
    for t in taus:
        c = t.mean()                       # oracle centre: best constant predictor
        cov_fix.append(float((np.abs(t - c) <= h_fixed).mean()))
        h_i = Z * max(float(t.std()), 1e-12)
        cov_ada.append(float((np.abs(t - c) <= h_i).mean()))
    cf, ca = np.array(cov_fix), np.array(cov_ada)
    print(f"\n### {label}   n_real={len(taus)}")
    print(f"  sd(tau) across realizations: min={sds.min():.4g} "
          f"p50={np.median(sds):.4g} max={sds.max():.4g}  ratio={sds.max()/max(sds.min(),1e-12):.4g}")
    print(f"  exactly-constant tau: {int((sds == 0).sum())}/{len(sds)}")
    for nm, c in (("fixed-width oracle", cf), ("adaptive oracle", ca)):
        print(f"  {nm:20s} mean={c.mean():.3f} sd={c.std(ddof=1):.3f} | "
              f"frac<=0.05 = {float((c <= 0.05).mean()):.3f}   "
              f"frac>=0.95 = {float((c >= 0.95).mean()):.3f}   "
              f"frac in [0.8,0.99] = {float(((c >= 0.8) & (c <= 0.99)).mean()):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    a = ap.parse_args()
    labels = a.labels or [os.path.basename(c.rstrip("/")) for c in a.cells]
    for cell, lbl in zip(a.cells, labels):
        report(lbl, load(cell))
    print("\nBimodal fixed-width coverage (mass at <=0.05 and >=0.95 with little")
    print("between) is the binary behaviour. 'frac in [0.8,0.99]' is the share of")
    print("realizations that can report an informative number at all.")


if __name__ == "__main__":
    main()
