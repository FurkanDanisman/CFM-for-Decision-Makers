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
# Case study: <case>_shift<S>_d<D>_r<R>. Case names contain underscores, so the
# shift/d/r suffixes are anchored from the right rather than split on "_".
_TAG = re.compile(r"^(?P<case>.+)_shift(?P<shift>[+-]?\d+)_d(?P<d>\d+)_r(?P<r>\d+)$")
# ComplexMech: CMECH_n<D>_<subset>_r<R>. Mapped onto the same (case, shift, d) keys so
# one aggregator serves both -- d is the node count and shift is fixed, which keeps
# --group working unchanged instead of needing a second code path.
_TAG_CM = re.compile(r"^CMECH_n(?P<d>\d+)_(?P<case>[a-z]+)_r(?P<r>\d+)$")


def _parse_tag(tag):
    m = _TAG.match(tag)
    if m:
        return {"case": m.group("case"), "shift": m.group("shift"),
                "d": m.group("d"), "r": m.group("r")}
    m = _TAG_CM.match(tag)
    if m:
        return {"case": f"CMECH_{m.group('case')}", "shift": "0",
                "d": m.group("d"), "r": m.group("r")}
    return None

_COVER = [("cover_vx", "v(x)"), ("cover_rho1", "v(x) rho=1"),
          ("bayesian", "bayesian"), ("bayesian_malc", "bayesian MALC")]
_LEN = [("len_vx", "v(x)"), ("len_rho1", "v(x) rho=1"),
        ("len_bayesian", "bayesian"), ("len_bayesian_malc", "bayesian MALC")]
_COLS = _COVER + _LEN

# t_{0.975, df}. Hardcoded rather than pulled from scipy, which is not a dependency
# here. With S = 10 SCMs, df = 9 and t = 2.262 -- using 1.96 instead would understate
# the interval by 15%, which matters precisely when S is small.
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
         8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
         14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
         20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
         26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def _t975(df):
    if df <= 0:
        return float("nan")
    return _T975.get(df, 1.960)          # df > 30: the normal value is within 1%


def _summarise(vals):
    """mean, SE across SCMs, t-based 95% CI, and the observed range.

    The SCM is the independent unit, so each is collapsed to ONE coverage and the
    spread of those S numbers carries both real between-SCM variation and the
    leftover Monte Carlo noise from finite R. Within-SCM SEs are not pooled in --
    they are already inside this spread.
    """
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    S = v.size
    if S == 0:
        return None
    mean = float(v.mean())
    if S == 1:
        return {"S": 1, "mean": mean, "se": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "min": mean, "max": mean}
    sd = float(v.std(ddof=1))
    se = sd / np.sqrt(S)
    t = _t975(S - 1)
    return {"S": S, "mean": mean, "se": se, "lo": mean - t * se,
            "hi": mean + t * se, "min": float(v.min()), "max": float(v.max())}


def _fmt(d, prec=3):
    if d is None:
        return "—"
    if d["S"] == 1:
        return f"{d['mean']:.{prec}f} (S=1)"
    return (f"{d['mean']:.{prec}f} ± {d['se']:.{prec}f} "
            f"[{d['lo']:.{prec}f}, {d['hi']:.{prec}f}] "
            f"({d['min']:.{prec}f}–{d['max']:.{prec}f})")


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
    malc_bs = set()
    for f in files:
        tag = os.path.basename(f)[: -len("_four.json")]
        m = _parse_tag(tag)
        if not m:
            skipped.append(tag)
            continue
        g = tuple(m[k] for k in keys)
        with open(f) as fh:
            blob = json.load(fh)
        sdy = (blob.get("sd_Y") or {}).get("sd")
        if blob.get("malc_B") is not None:
            malc_bs.add(int(blob["malc_B"]))
        for model, v in blob.get("models", {}).items():
            got = False
            for col, _ in _COLS:
                x = v.get(col)
                if x is not None and np.isfinite(float(x)):
                    acc[g][model][col].append(float(x))
                    got = True
                    # len/sd(Y): lengths are in the outcome's units, so a d=50 cell and
                    # a d=5 cell are not on one ruler. Dividing by this cell's own
                    # sd(Y) is what lets the length columns be compared or pooled
                    # across d values and benchmarks.
                    if col.startswith("len_") and sdy:
                        acc[g][model][col + "_n"].append(float(x) / float(sdy))
            if sdy:
                acc[g][model]["sd_Y"].append(float(sdy))
            if got:
                ncell[g][model] += 1
                if v.get("datasets") is not None:
                    ndata[g][model].append(int(v["datasets"]))

    ttl = f"  ({a.label})" if a.label else ""
    L = [f"## fq4 final — across SCMs{ttl}", "",
         f"{len(files)} cell table(s) under {a.scores}", "",
         "Each entry is  mean ± SE [95% CI] (min–max across SCMs).", ""]

    def _table(cols, title, prec):
        rows_out = ["### " + title, "",
                    "| " + " | ".join(keys + ["model", "S", "min datasets"]
                                      + [n for _, n in cols]) + " |"]
        rows_out.append("|" + "---|" * (rows_out[-1].count("|") - 1))
        for g in sorted(acc):
            for model in sorted(acc[g]):
                nd = ndata[g][model]
                summ = {c: _summarise(acc[g][model][c]) for c, _ in cols}
                any_s = next((d["S"] for d in summ.values() if d), 0)
                out = ([*g] if keys else []) + [model, str(any_s),
                                                str(min(nd)) if nd else "—"]
                out += [_fmt(summ[c], prec) for c, _ in cols]
                rows_out.append("| " + " | ".join(out) + " |")
        return rows_out + [""]

    _LEN_N = [(c + "_n", n) for c, n in _LEN]
    L += _table(_COVER, "Coverage (nominal 0.95)", 3)
    L += _table(_LEN, "Mean interval length (outcome units)", 4)
    L += _table(_LEN_N, "Mean interval length / sd(Y)", 4)
    L += _table([("sd_Y", "sd(Y)")], "Outcome scale of the cells", 4)

    if malc_bs:
        b = ", ".join(str(x) for x in sorted(malc_bs))
        L += ["", f"MALC bootstrap B = {b}."
              + ("" if malc_bs == {1000} else
                 " NOTE: the project's other tables use B=1000 K=1, so this column is"
                 " internally consistent but NOT comparable to them.")]
    L += ["",
          "THE SCM IS THE INDEPENDENT UNIT. Each SCM is collapsed to one coverage --",
          "its own average over datasets and queries -- and the mean, SE and CI are",
          "taken across those S numbers. The spread of them already carries both real",
          "between-SCM variation and the leftover Monte Carlo noise from finite R, so",
          "within-SCM SEs are not pooled in.",
          "",
          "SE = s_C/sqrt(S); the CI uses t_{0.975, S-1}, NOT 1.96 -- at S=10 that is",
          "2.262, and using 1.96 would understate the interval by 15%. The range is",
          "the observed spread of per-SCM coverages: a CI below 0.95 says the method",
          "under-covers on average, and a wide range says it is not uniform.",
          "",
          "len/sd(Y) divides each cell's length by that cell's own outcome SD, so",
          "lengths are comparable across d values and across benchmarks -- raw lengths",
          "are in the outcome's units and are not. sd(Y) is the SD of the pooled",
          "potential outcomes per replicate averaged over replicates, the same recipe",
          "length_normalizers.py uses. It is recorded when the cell is scored because",
          "it can only be computed from the generated data, which is deleted after.",
          "sd(Y) not sd(tau): the case-study generator adds the same eps to both arms,",
          "so tau is noiseless and many realizations have sd(tau) < 1e-6.",
          "",
          "Within one SCM, coverage is per-query over resampled observational datasets,",
          "averaged over queries -- unweighted, so read `min datasets` before trusting",
          "a row. v(x) and v(x) rho=1 are normal-approximation intervals from the",
          "density's moments; bayesian is the central 95% of the predictive tau",
          "density, raw and after MALC. A dash means no cell reported that column."]
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
