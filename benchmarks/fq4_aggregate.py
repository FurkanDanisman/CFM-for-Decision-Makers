#!/usr/bin/env python
"""Average the per-realization fq4 tables into one four-column result.

The hierarchy, unweighted at every level:

  1. per QUERY      coverage over that query's ~100 resampled observational datasets
  2. per REALIZATION mean of its ~10 per-query coverages   <- one _four.json per cell
  3. FINAL           mean over realizations                <- this script

Level 1 and 2 happen in fixedq_ci_coverage / cate_density_metrics, whose per-cell
values these files carry. This does level 3 and nothing else.

Unweighted, not pooled, at each step -- which matters as soon as one cell is short.
A realization with 3 completed replicates counts exactly as much as one with 100,
because the quantity asked for is the average of per-realization coverages, not the
coverage of the pooled replicates. The `cells` and `min_datasets` columns are there so
a mean resting on thin cells is visible rather than implied.

    python benchmarks/fq4_aggregate.py --scores $SCRATCH/fq4_scores \
        --group case,shift,d --out $SCRATCH/fq4_final.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

# TAG is <case>_shift<S>_d<D>_r<R>; the case name itself contains underscores, so
# parse from the right rather than splitting on "_".
_TAG = re.compile(r"^(?P<case>.+)_shift(?P<shift>[+-]?\d+)_d(?P<d>\d+)_r(?P<r>\d+)$")

_COLS = [("cover_vx", "v(x)"), ("len_vx", "len"),
         ("cover_rho1", "v(x) rho=1"), ("len_rho1", "len"),
         ("bayesian", "bayesian"), ("len_bayesian", "len"),
         ("bayesian_malc", "bayesian MALC"), ("len_bayesian_malc", "len")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True,
                    help="directory of <tag>_four.json written by score_fq4.sh")
    ap.add_argument("--group", default="",
                    help="comma-separated subset of case,shift,d to break the table "
                         "down by. Empty = one row per model over everything.")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.scores, "*_four.json")))
    if not files:
        sys.exit(f"no *_four.json under {a.scores} -- run score_fq4.sh per cell first")

    keys = [k for k in a.group.split(",") if k]
    bad = [k for k in keys if k not in ("case", "shift", "d")]
    if bad:
        sys.exit(f"--group accepts case,shift,d; got {bad}")

    # group -> model -> column -> list of per-realization values
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    ncell = defaultdict(lambda: defaultdict(int))
    ndata = defaultdict(lambda: defaultdict(list))
    skipped = []
    for f in files:
        tag = os.path.basename(f)[: -len("_four.json")]
        m = _TAG.match(tag)
        if not m:
            skipped.append(tag)
            continue
        g = tuple(m.group(k) for k in keys)
        with open(f) as fh:
            blob = json.load(fh)
        for model, v in blob.get("models", {}).items():
            got = False
            for col, _ in _COLS:
                x = v.get(col)
                if x is not None and np.isfinite(float(x)):
                    acc[g][model][col].append(float(x))
                    got = True
            if got:
                ncell[g][model] += 1
                if v.get("datasets") is not None:
                    ndata[g][model].append(int(v["datasets"]))

    ttl = f"  ({a.label})" if a.label else ""
    hdr = (["| " + " | ".join(keys + ["model", "cells", "min datasets"]
                              + [n for _, n in _COLS]) + " |"]
           if keys else
           ["| " + " | ".join(["model", "cells", "min datasets"]
                              + [n for _, n in _COLS]) + " |"])
    ncol = hdr[0].count("|") - 1
    L = [f"## fq4 final — average over realizations{ttl}", "",
         f"{len(files)} cell table(s) under {a.scores}", ""] + hdr + \
        ["|" + "---|" * ncol]

    for g in sorted(acc):
        for model in sorted(acc[g]):
            cells = [*g] if keys else []
            nd = ndata[g][model]
            out = cells + [model, str(ncell[g][model]),
                           str(min(nd)) if nd else "—"]
            for col, _ in _COLS:
                vals = acc[g][model][col]
                out.append(f"{np.mean(vals):.4f}" if vals else "—")
            L.append("| " + " | ".join(out) + " |")

    L += ["",
          "Every value is the unweighted mean over REALIZATIONS of that",
          "realization's own average over its queries, each of which is a coverage",
          "over resampled observational datasets. Unweighted at each level, so a",
          "realization with 3 completed replicates counts as much as one with 100 --",
          "read `cells` and `min datasets` before trusting a row.",
          "",
          "v(x) and v(x) rho=1 are normal-approximation intervals from the density's",
          "moments; bayesian is the central 95% of the predictive tau density, raw",
          "and after MALC. A dash means no cell reported that column."]
    if skipped:
        L += ["", f"Unparsed filenames ({len(skipped)}): " + ", ".join(skipped[:8])
              + (" ..." if len(skipped) > 8 else "")]
    txt = "\n".join(L)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt + "\n")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
