#!/usr/bin/env python
"""Compare two density dumps array by array.

A differing md5 does not prove two dumps hold different PREDICTIONS: the files
also carry scaling constants, edges and provenance, so a harness that silently
fell back to another dataset can still produce a byte-different file with
identical densities and identical truth.

Reports, per shared key, whether the arrays are exactly equal, and summarises the
two that decide the question: the true effect and the predicted densities. If
those match while the files differ, the dump stage fell back; if they differ while
the scored table shows identical numbers, the fault is downstream in the scorer.

    python benchmarks/diff_dumps.py A.npz B.npz
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    a = ap.parse_args()

    za, zb = np.load(a.a, allow_pickle=True), np.load(a.b, allow_pickle=True)
    ka, kb = set(za.files), set(zb.files)
    if ka != kb:
        print(f"KEY SETS DIFFER\n  only in A: {sorted(ka - kb)}\n  only in B: {sorted(kb - ka)}")
    shared = sorted(ka & kb)

    print(f"{'key':28} {'shape':>16}  verdict")
    print("-" * 64)
    same, diff = [], []
    for k in shared:
        x, y = np.asarray(za[k]), np.asarray(zb[k])
        if x.shape != y.shape:
            print(f"{k:28} {str(x.shape):>16}  SHAPE {x.shape} vs {y.shape}")
            diff.append(k); continue
        if x.dtype.kind in "fc":
            eq = np.allclose(x, y, rtol=0, atol=0, equal_nan=True)
            if not eq:
                d = float(np.nanmax(np.abs(x.astype(float) - y.astype(float))))
                print(f"{k:28} {str(x.shape):>16}  differ  max|Δ|={d:.6g}")
                diff.append(k); continue
        else:
            eq = bool(np.array_equal(x, y))
            if not eq:
                print(f"{k:28} {str(x.shape):>16}  differ")
                diff.append(k); continue
        print(f"{k:28} {str(x.shape):>16}  identical")
        same.append(k)

    print(f"\nidentical: {len(same)}   differing: {len(diff)}")
    # The two that settle it.
    truth = [k for k in shared if "true" in k.lower() or "cate" in k.lower()
             or k in ("y_true",)]
    dens = [k for k in shared if k.startswith("p_")]
    print(f"\ntruth keys   {truth or '(none found)'}: "
          f"{'IDENTICAL' if all(k in same for k in truth) and truth else 'differ'}")
    print(f"density keys {dens or '(none found)'}: "
          f"{'IDENTICAL' if all(k in same for k in dens) and dens else 'differ'}")
    print("""
If truth AND densities are identical, the two dumps are the same evaluation under
two names -- a dump-stage fallback. If either differs, the dumps are genuinely
distinct and identical scored numbers come from the scorer, not the dump.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
