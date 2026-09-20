#!/usr/bin/env python
"""Assemble the results table from whatever per-realization dumps exist.

Two jobs, both of which the scoring fan-out made necessary:

POOLING. The case-study figure is the pooling of shifts 0, +2 and -2. Averaging
three per-shift means only equals the pooled mean when every shift contributes the
same number of realizations, which is not guaranteed. So the scorer writes
per-realization arrays and this concatenates them before taking any mean.

PARTIAL RESULTS. Every cell writes its own .npz as it finishes, so this reports
whatever has landed and names what is missing, rather than needing a complete set.
A run killed halfway still produces a table.

Layout consumed (written by submit_full_table.sbatch):

    <perreal>/<model_label>/<stage>__<group>__<slice>.npz

  stage  : raw | malc | indep_raw | indep_malc
  group  : the reporting row -- a RealCause dataset, or d<N>_<Case>
  slice  : what gets POOLED into the group -- shift0/shift+2/shift-2, or "-"

    python benchmarks/collect_table.py --perreal $SCRATCH/perreal
    python benchmarks/collect_table.py --perreal $SCRATCH/perreal --bench cs
"""
from __future__ import annotations

import argparse
import glob
import os
from collections import defaultdict

import numpy as np

_KEYS = ("cover", "length", "is05", "crps")
_STAGES = ("raw", "malc", "indep_raw", "indep_malc")


def _load(perreal):
    """-> {(model, stage, group): {key: [arrays, one per slice]}} plus the slices seen."""
    acc = defaultdict(lambda: defaultdict(list))
    slices = defaultdict(set)
    for f in sorted(glob.glob(os.path.join(perreal, "*", "*.npz"))):
        label = os.path.basename(os.path.dirname(f))
        base = os.path.basename(f)[:-4]
        parts = base.split("__")
        if len(parts) != 3:
            continue
        stage, group, sl = parts
        try:
            z = np.load(f)
        except Exception as e:                      # truncated / mid-write
            print(f"  WARN unreadable, skipped: {f} ({type(e).__name__})")
            continue
        for k in z.files:
            if "__" not in k:
                continue
            method, key = k.rsplit("__", 1)
            if key not in _KEYS:
                continue
            arr = np.asarray(z[k], dtype=float).ravel()
            if not arr.size:
                continue
            # Row identity is <root label>/<harness method>, NOT the method alone.
            # Several models share a harness and therefore a subdir name -- the
            # repro joint_2d and the repro 1D heads all dump under 'dopfn_native',
            # as do the released weights in the shared root. Keying on the method
            # alone would concatenate different models' realizations into one row
            # and report the average as if it were a single model.
            model = f"{label}/{method}"
            acc[(model, stage, group)][key].append(arr)
            slices[(model, stage, group)].add(sl)
    return acc, slices


def _stat(arrays):
    """Pool by CONCATENATION, then summarise. Not a mean of means."""
    v = np.concatenate(arrays) if arrays else np.array([])
    if v.size == 0:
        return None
    sd = float(v.std(ddof=1)) if v.size > 1 else float("nan")
    return dict(n=int(v.size), mean=float(v.mean()), sd=sd,
                se=sd / np.sqrt(v.size) if v.size > 1 else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perreal", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-real", type=int, default=1,
                    help="flag rows built from fewer realizations than this")
    a = ap.parse_args()

    acc, slices = _load(a.perreal)
    if not acc:
        print(f"no per-realization dumps under {a.perreal}")
        return

    groups = sorted({g for (_, _, g) in acc})
    methods = sorted({m for (m, _, _) in acc})
    L = []
    for g in groups:
        L += [f"\n### {g}", "",
              "| model | Cov | Len | IS | Cov-MALC | Len-MALC | IS-MALC "
              "| Cov-indep | Len-indep | IS-indep | n | pooled from |",
              "|" + "---|" * 12]
        for m in methods:
            cells, n_seen, pooled = [], 0, set()
            for stage in ("raw", "malc", "indep_raw"):
                st = {k: _stat(acc.get((m, stage, g), {}).get(k, [])) for k in _KEYS}
                if st["cover"] is None:
                    cells += ["—"] * 3
                    continue
                n_seen = max(n_seen, st["cover"]["n"])
                pooled |= slices.get((m, stage, g), set())
                cells += [f"{st['cover']['mean']:.3f} ± {st['cover']['sd']:.3f}",
                          f"{st['length']['mean']:.4f}",
                          f"{st['is05']['mean']:.4f}"]
            if all(c == "—" for c in cells):
                continue
            flag = " ⚠" if 0 < n_seen < a.min_real else ""
            L.append(f"| {m} | " + " | ".join(cells)
                     + f" | {n_seen}{flag} | {','.join(sorted(pooled)) or '—'} |")

    # What is still missing is part of the result, not a footnote.
    L += ["", "### coverage of this table", ""]
    have = {(m, s, g) for (m, s, g) in acc}
    for stage in _STAGES:
        got = sorted({g for (m, s, g) in have if s == stage})
        miss = [g for g in groups if g not in got]
        L.append(f"- **{stage}**: {len(got)}/{len(groups)} groups"
                 + (f" — missing: {', '.join(miss)}" if miss else " — complete"))

    txt = "\n".join(L)
    print(txt)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(txt + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
