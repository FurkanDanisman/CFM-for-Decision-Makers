#!/usr/bin/env python
"""Is a dumped density on the same scale as the truth it is scored against?

Separates two explanations for bad coverage that look identical in a results table:

  MISCALIBRATED MODEL   the density is on the right scale but too wide or too
                        narrow. Coverage is wrong, the numbers are real.
  WRONG SCALE           the density is decoded on a grid that does not match the
                        outcome. Coverage is meaningless, and so is PEHE.

For each realization it reports the tau grid the scorer will use, the predicted
mean and sd implied by the dumped pmf, and the true effect's mean and sd. Then the
two ratios that distinguish the cases:

  grid_span / true_sd   how much room the grid gives the truth. Enormous means the
                        interval cannot help covering; tiny means it cannot reach.
  pred_sd / true_sd     the model's own dispersion against the truth's.

Uses cate_density_metrics' loader, so "the grid the scorer will use" is exactly
that -- not a re-derivation that could differ.

    python benchmarks/diag_dump_scale.py --root $SCRATCH/semireal_dumps/cpfn1d_j32 \
        --dataset SEMIREAL_law_race
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "UWYK_Fig3_4"))
from cate_density_metrics import METHODS, _load_arrays_raw, _resolve_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--max-real", type=int, default=3)
    a = ap.parse_args()

    for label, subdir, tag in METHODS:
        d = _resolve_dir(a.root, subdir, a.dataset)
        if not os.path.isdir(d):
            continue
        fs = [f for f in sorted(glob.glob(os.path.join(d, "*.npz")))
              if os.path.basename(f) != "summary.npz"][: a.max_real]
        if not fs:
            continue
        print(f"\n### {label}   ({len(fs)} realization(s) from {d})")
        print(f"  {'file':<18} {'grid span':>22} {'pred mean':>11} {'pred sd':>10} "
              f"{'true mean':>11} {'true sd':>10} {'span/true_sd':>13} {'pred/true':>10}")
        for f in fs:
            got = _load_arrays_raw(f, tag)
            if got is None:
                print(f"  {os.path.basename(f):<18} (not loadable by the scorer)")
                continue
            atoms, pmfs, y_true, _ = got
            atoms = np.asarray(atoms, dtype=float)
            pmfs = np.atleast_2d(np.asarray(pmfs, dtype=float))
            y_true = np.asarray(y_true, dtype=float).ravel()
            # predicted mean/sd per query from the pmf, then averaged
            pm = pmfs @ atoms
            ps = np.sqrt(np.maximum(pmfs @ (atoms ** 2) - pm ** 2, 0.0))
            t_sd = float(y_true.std()) if y_true.size > 1 else float("nan")
            span = float(atoms.max() - atoms.min())
            print(f"  {os.path.basename(f):<18} "
                  f"[{atoms.min():+9.3g},{atoms.max():+9.3g}] "
                  f"{float(pm.mean()):11.4g} {float(ps.mean()):10.4g} "
                  f"{float(y_true.mean()):11.4g} {t_sd:10.4g} "
                  f"{span / max(t_sd, 1e-12):13.4g} "
                  f"{float(ps.mean()) / max(t_sd, 1e-12):10.4g}")
    print("""
Reading it:
  span/true_sd around 10-100      grid is reasonable
  span/true_sd >> 1000            the grid dwarfs the truth: a 95% interval covers
                                  everything regardless of the model, and coverage
                                  near 1.000 with IS == Len is the signature
  span/true_sd << 10              the grid cannot reach the truth: coverage near 0
  pred/true around 1              the model's dispersion matches the truth
  pred/true >> 1 with a sane span the model is genuinely over-dispersed -- a real
                                  finding about the model, not a scale bug""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
