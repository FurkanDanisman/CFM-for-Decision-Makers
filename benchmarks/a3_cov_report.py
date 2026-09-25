#!/usr/bin/env python
"""Coverage / length tables from the a3 per-realization npz.

    python benchmarks/a3_cov_report.py --out $SCRATCH/a3_cov --smoother none

Each npz holds one value per REALIZATION (keys <method>__<cover|length|is05|crps>).
Cells are combined by CONCATENATING those arrays and then averaging, which is the
pooled per-realization mean; averaging per-cell means instead is only equal when
every cell has the same realization count, and the case-study cells do not.

The +- is std/sqrt(n) over realizations.

Case-study files are per (case, shift, d); --cs-d selects which d to report, the
same filter final_table.py applies via CS_D_DEFAULT.
"""
from __future__ import annotations
import argparse, glob, os, re
from collections import defaultdict

import numpy as np

RC = ["IHDP", "ACIC", "CPS", "PSID", "PSID_bal"]
CM = [5, 10, 20, 30, 40, 50]
CASES = ["Observed_Confounder", "Backdoor_Criterion", "Observed_Mediator",
         "Observed_Mediator_and_Confounder", "Unobserved_Confounder",
         "Frontdoor_Criterion"]
CS_D_DEFAULT = ["5", "10", "20", "30", "40", "50"]


def load(path, acc):
    try:
        z = np.load(path)
    except Exception:
        return
    for k in z.files:
        if "__" not in k:
            continue
        method, key = k.rsplit("__", 1)
        if key not in ("cover", "length", "is05", "crps"):
            continue
        v = np.asarray(z[k], dtype=float).ravel()
        if v.size:
            acc[method][key].append(v)


def stat(arrs):
    if not arrs:
        return None
    v = np.concatenate(arrs)
    n = v.size
    return v.mean(), (v.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")), n


def fmt(s, key):
    if s is None:
        return "--"
    return f"{s[0]:.3f}±{s[1]:.3f}" if key == "cover" else f"{s[0]:.2f}±{s[1]:.2f}"


def table(title, cols, get, methods, keys=("cover", "length")):
    print(f"\n### {title}")
    print("| method | metric | " + " | ".join(str(c) for c in cols) + " |")
    print("|" + "---|" * (len(cols) + 2))
    for m in methods:
        for key in keys:
            row = [fmt(get(m, c, key), key) for c in cols]
            label = "coverage" if key == "cover" else key
            print(f"| {m} | {label} | " + " | ".join(row) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoother", default="none")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--cs-d", nargs="+", default=CS_D_DEFAULT)
    a = ap.parse_args()
    base = os.path.join(a.out, a.smoother)

    # rc / cm live in flat files whose names embed the model; cs lives per-model.
    rc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    cm = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for f in glob.glob(os.path.join(base, "*.npz")):
        b = os.path.basename(f)[:-4]
        for ds in sorted(RC, key=len, reverse=True):
            if b.startswith("rc_") and b.endswith("_" + ds):
                load(f, rc[b[3:-len(ds) - 1]][ds]); break
        m = re.match(r"^cm_(.+)_n(\d+)$", b)
        if m:
            load(f, cm[m.group(1)][int(m.group(2))])

    cs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for d in glob.glob(os.path.join(base, "perreal", "*")):
        if not os.path.isdir(d):
            continue
        model = os.path.basename(d)
        for f in glob.glob(os.path.join(d, "*.npz")):
            mm = re.match(r"^[a-z]+__d(\d+)_(.+)__shift(\S+)\.npz$",
                          os.path.basename(f))
            if mm and mm.group(1) in a.cs_d:
                load(f, cs[model][mm.group(2)])

    models = a.models or sorted(set(rc) | set(cm) | set(cs))
    print(f"# a3 coverage — smoother={a.smoother}  cs_d={','.join(a.cs_d)}")
    for model in models:
        print(f"\n## {model}")
        meths = sorted({m for sel in list(rc[model].values())
                        + list(cm[model].values()) + list(cs[model].values())
                        for m in sel})
        if not meths:
            print("  (nothing)"); continue
        if rc[model]:
            table("RealCause", RC,
                  lambda m, c, k: stat(rc[model][c][m][k]), meths)
        if cm[model]:
            table("ComplexMech (N=1000, total)", CM,
                  lambda m, c, k: stat(cm[model][c][m][k]), meths)
        if cs[model]:
            short = [c.replace("Observed_", "Obs_").replace("_Criterion", "")
                      .replace("Unobserved_", "Unobs_") for c in CASES]
            table("Case study (shifts + d pooled)", short,
                  lambda m, c, k, _s=short: stat(
                      cs[model][CASES[_s.index(c)]][m][k]), meths)
    print()


if __name__ == "__main__":
    main()
