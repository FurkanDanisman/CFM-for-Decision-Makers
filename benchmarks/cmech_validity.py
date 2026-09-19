"""Validity manifest for ComplexMech realizations: which ones are usable.

WHY. A large fraction of ComplexMech realizations have a SATURATED outcome. The
prior's pre-activation scale is unbounded (unlike the case-study generator, which
uses fan-in bounded weights), so a tanh-like activation pins the outcome at its
boundary. Measured on 5node/path_TY/hide_0.0: up to 99.2% of units sit exactly at
|Y| = 1, and one control arm has sd(Y_do0) = 0.005 against a claimed ATE of 1.96 --
381x the outcome's own spread.

In such a realization the observational outcome carries no gradient, so the effect
is not estimable by anything and every model correctly returns ~0. Because
saturation is a property of the REALIZATION, all of its queries fail together --
which is why per-realization coverage collapses to 0 or 1 and its mean sits at
0.66-0.86 for every model regardless of architecture.

WHAT THIS DOES. Scans the source data and records, per realization, whether it is
usable. Scoring then skips the unusable ones. Nothing is regenerated and the
published DGP is untouched -- this is a stated validity criterion, not a change
to the generative model.

CRITERIA (both configurable):
    pinned  fraction of |Y| > 1-eps in either arm, above --max-pinned
    ratio   |ATE| / sd(Y_do0) above --max-ratio -- an effect larger than a couple
            of times the outcome's own spread is a saturation artifact, not a
            causal effect a model could recover

Usage:
    python benchmarks/cmech_validity.py --data-root R-PFN/UWYK_Fig3_4/data \\
        --out $SCRATCH/cmech_validity.json
    python benchmarks/cmech_validity.py --data-root ... --max-pinned 0.05 --max-ratio 5
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np


def scan(cell_dir, max_pinned, max_ratio, eps=1e-3):
    out, stats = {}, {"n": 0, "pinned": 0, "ratio": 0, "both": 0}
    for f in sorted(glob.glob(os.path.join(cell_dir, "r*.npz")),
                    key=lambda p: int(re.search(r"r(\d+)\.npz$", p).group(1))):
        r = int(re.search(r"r(\d+)\.npz$", f).group(1))
        try:
            z = np.load(f, allow_pickle=True)
            y0 = np.asarray(z["Y_do0"], dtype=np.float64).reshape(-1)
            y1 = np.asarray(z["Y_do1"], dtype=np.float64).reshape(-1)
            tau = np.asarray(z["true_cate"], dtype=np.float64).reshape(-1)
        except Exception:
            continue
        sd0 = float(y0.std())
        pinned = max(float((np.abs(y0) > 1.0 - eps).mean()),
                     float((np.abs(y1) > 1.0 - eps).mean()))
        ratio = abs(float(tau.mean())) / max(sd0, 1e-12)
        bad_p, bad_r = pinned > max_pinned, ratio > max_ratio
        stats["n"] += 1
        stats["pinned"] += bad_p; stats["ratio"] += bad_r
        stats["both"] += (bad_p and bad_r)
        out[r] = dict(valid=not (bad_p or bad_r), pinned=round(pinned, 5),
                      ratio=round(ratio, 4), sd_y0=round(sd0, 6),
                      ate=round(float(tau.mean()), 6))
    return out, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True,
                    help="e.g. R-PFN/UWYK_Fig3_4/data (contains complexmech/)")
    ap.add_argument("--max-pinned", type=float, default=0.01)
    ap.add_argument("--max-ratio", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cells = sorted(glob.glob(os.path.join(a.data_root, "complexmech",
                                          "*node", "*", "hide_*")))
    if not cells:
        raise SystemExit(f"no cells under {a.data_root}/complexmech/*node/*/hide_*")

    man = {"criteria": {"max_pinned": a.max_pinned, "max_ratio": a.max_ratio},
           "cells": {}}
    print(f"{'cell':44s} {'n':>4s} {'valid':>6s} {'%valid':>7s} "
          f"{'pinned':>7s} {'ratio':>6s}")
    print("-" * 82)
    tot = ok = 0
    for c in cells:
        key = os.path.relpath(c, os.path.join(a.data_root, "complexmech"))
        rows, st = scan(c, a.max_pinned, a.max_ratio)
        if not rows:
            continue
        nv = sum(1 for v in rows.values() if v["valid"])
        man["cells"][key] = rows
        tot += st["n"]; ok += nv
        print(f"{key:44s} {st['n']:4d} {nv:6d} {100*nv/max(st['n'],1):6.1f}% "
              f"{st['pinned']:7d} {st['ratio']:6d}")
    print("-" * 82)
    print(f"{'TOTAL':44s} {tot:4d} {ok:6d} {100*ok/max(tot,1):6.1f}%")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(man, fh, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
