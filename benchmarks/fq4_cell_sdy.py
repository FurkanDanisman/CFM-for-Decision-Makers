#!/usr/bin/env python
"""sd(Y) for ONE fq4 cell, so interval lengths survive the cell's deletion.

Lengths are in the outcome's units, so they cannot be compared across datasets of
different scale -- a d=50 cell and a ComplexMech cell are not on one ruler. Dividing
by the cell's own sd(Y) fixes that, and sd(Y) is computable ONLY from the generated
data, which the reaper deletes once a cell is scored. So it is recorded here, before
the delete, rather than recovered later (it cannot be).

The recipe matches benchmarks/length_normalizers.py exactly -- the SD of the pooled
potential outcomes per replicate, then the mean over replicates. Reimplementing it
differently would put two definitions of sd(Y) in the same project.

sd(Y), not sd(tau): the case-study generator adds the same eps to both arms, so
tau = mu_1 - mu_0 is noiseless and 42 of 150 realizations have sd(tau) < 1e-6 --
dividing by that produces garbage.

    python benchmarks/fq4_cell_sdy.py --cell <dir of replicate npz>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from length_normalizers import _agg, _sd_pooled            # noqa: E402


def sdy_of_cell(cell, max_files=None):
    fs = sorted(f for f in glob.glob(os.path.join(cell, "*.npz"))
                if os.path.basename(f) != "summary.npz")
    if max_files:
        fs = fs[:max_files]
    sds = []
    for f in fs:
        try:
            z = np.load(f, allow_pickle=True)
        except Exception:
            continue
        keys = set(z.files)
        if {"T", "Y", "mu_0", "mu_1"} <= keys:
            # Case study: eps is recoverable exactly, because the generator adds the
            # SAME eps to both arms.
            T = np.asarray(z["T"], float).ravel()
            Y = np.asarray(z["Y"], float).ravel()
            m0 = np.asarray(z["mu_0"], float).ravel()
            m1 = np.asarray(z["mu_1"], float).ravel()
            eps = Y - np.where(T > 0.5, m1, m0)
            sds.append(_sd_pooled(m0 + eps, m1 + eps))
        elif {"Y_do0", "Y_do1"} <= keys:
            # ComplexMech: the potential outcomes are stored directly.
            sds.append(_sd_pooled(z["Y_do0"], z["Y_do1"]))
        elif "Y_train" in keys:
            sds.append(_sd_pooled(z["Y_train"]))
    return _agg(sds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    g = sdy_of_cell(a.cell, a.max_files)
    if g is None:
        print(f"no usable replicate npz under {a.cell}", file=sys.stderr)
        return 1
    print(json.dumps(g, indent=2))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(g, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
