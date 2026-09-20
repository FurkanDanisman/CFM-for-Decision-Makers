#!/usr/bin/env python
"""Choose the complexmech validity band from the prior's own statistics.

WHY A BAND AND NOT A FLOOR. Per-realization coverage is
    cov_i ~= 2*Phi(h / sigma_i) - 1
where sigma_i is tau's spread across units in realization i and h is the model's
interval half-width, which is roughly fixed across realizations. So coverage is
intermediate only when sigma_i is comparable to h:

    sigma_i << h  ->  cov_i ~= 1        (interval swallows every unit)
    sigma_i >> h  ->  cov_i ~= h/sigma_i (interval misses nearly everything)

A prior whose sigma_i spans orders of magnitude therefore produces coverage piled
at 0 and 1 with nothing between -- exactly what this benchmark showed -- and
removing only the sigma_i = 0 cases does not fix it, because the high tail is just
as damaging as the zeros.

This reads the raw per-attempt tallies and reports, for candidate bands on
sigma/sd(y0), the acceptance rate and the coverage spread a well-calibrated model
would produce. Pick the band that keeps coverage tight without rejecting so much
that resampling cannot fill a cell.

    python benchmarks/cmech_band_analysis.py --tally-dir /tmp/cmech_tally
"""
from __future__ import annotations

import argparse
import glob
import math
import os

import numpy as np


def _phi(z):
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def coverage_given(sig, h):
    """Coverage a correctly-centred interval of half-width h gets on spread sig."""
    sig = np.asarray(sig, dtype=float)
    out = np.ones_like(sig)
    nz = sig > 0
    out[nz] = 2.0 * _phi(h / sig[nz]) - 1.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tally-dir", required=True)
    ap.add_argument("--bands", nargs="+",
                    default=["0.0,inf", "0.01,inf", "0.05,inf", "0.10,inf",
                             "0.05,2.0", "0.10,2.0", "0.10,1.0", "0.20,1.0",
                             "0.20,2.0", "0.30,3.0"])
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.tally_dir, "**", "tally_*.npz"),
                             recursive=True))
    if not files:
        print(f"no tally_*.npz under {a.tally_dir}")
        return
    print(f"{len(files)} cell tallies\n")

    per_cell = {}
    for f in files:
        z = np.load(f)
        key = os.path.basename(f)[len("tally_"):-4]
        per_cell[key] = np.asarray(z["tau_rel"], dtype=float)

    allv = np.concatenate(list(per_cell.values()))
    print(f"pooled n={allv.size}   sigma/sd(y0) quantiles")
    for q in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99):
        print(f"    p{int(q*100):<3d} {np.quantile(allv, q):.4f}")
    print(f"    zero-exactly: {int((allv <= 0).sum())}/{allv.size} "
          f"({100.0*(allv <= 0).mean():.1f}%)")
    print()

    hdr = (f"{'band':<14} {'accept':>8} {'sigma ratio':>12} "
           f"{'cov p10':>8} {'cov p50':>8} {'cov p90':>8} {'frac<0.5':>9} {'frac>0.99':>10}")
    print(hdr); print("-" * len(hdr))
    for b in a.bands:
        lo, hi = (float(x) for x in b.split(","))
        keep = allv[(allv >= lo) & (allv <= hi)]
        if keep.size < 5:
            print(f"{b:<14} {keep.size:>8} {'--':>12}")
            continue
        # A calibrated model targets the typical realization: set h so the MEDIAN
        # accepted realization lands at nominal 95%.
        h = 1.959964 * float(np.median(keep))
        cov = coverage_given(keep, h)
        ratio = float(keep.max() / max(keep.min(), 1e-12))
        print(f"{b:<14} {100.0*keep.size/allv.size:7.1f}% {ratio:12.1f} "
              f"{np.quantile(cov,0.10):8.3f} {np.quantile(cov,0.50):8.3f} "
              f"{np.quantile(cov,0.90):8.3f} {float((cov<0.5).mean()):9.3f} "
              f"{float((cov>0.99).mean()):10.3f}")

    print()
    print("cov p10..p90 is the coverage spread a CORRECTLY CALIBRATED model would")
    print("still show, from the benchmark alone. frac<0.5 and frac>0.99 are the")
    print("realizations that would read as near-total miss and near-total cover --")
    print("the binary behaviour. Pick a band that shrinks both while keeping the")
    print("acceptance rate high enough for resampling to fill a cell.")
    print()
    print("per-cell zero fraction:")
    for k in sorted(per_cell):
        v = per_cell[k]
        print(f"    {k:<26} n={v.size:<5} zero={100.0*(v<=0).mean():5.1f}%  "
              f"p50={np.median(v):.4f}")


if __name__ == "__main__":
    main()
