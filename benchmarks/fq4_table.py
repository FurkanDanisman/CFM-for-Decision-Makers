#!/usr/bin/env python
"""The four-column fixed-query coverage table, per model.

    v(x) coverage | v(x) rho=1 coverage | bayesian | bayesian MALC

The first two are NORMAL-APPROXIMATION intervals built from the moments of the
model's predictive density: est +- 1.96*sqrt(v), with v the head's own coupling or
(s1-s0)^2 under perfect positive dependence. The last two are the CENTRAL 95%
interval of the predictive tau distribution itself -- raw, and after MALC
recalibration at B=1000 K=1. They answer different questions: the first pair asks
whether the head's claimed sampling variability matches its actual spread, the
second whether its distributional shape is calibrated.

This does not recompute any of them. fixedq_ci_coverage.py produces the first pair
and cate_density_metrics.py the second, both already used for every other table in
this project; duplicating either would create a second definition of a number that
is already reported elsewhere. This merges their markdown into one table so the
four are read side by side.

    python benchmarks/fq4_table.py \
        --vx    $SCRATCH/fq4_vx.md \
        --bayes $SCRATCH/fq4_raw.md \
        --malc  $SCRATCH/fq4_malc.md \
        --out   $SCRATCH/fq4_table.md
"""
from __future__ import annotations

import argparse
import os
import re
import sys

# cate_density_metrics names methods by harness (uwyk1d-noanc, cpfn2d, ...) while
# the fixed-query dumps are one root per checkpoint, so fixedq_ci_coverage names
# rows after the root. Neither is wrong; they are different keys for the same model,
# and the merge has to bridge them explicitly rather than hope the strings match.
_ALIAS = {
    "cpfn2d": "cpfn2d_eta0",
    "cpfn1d": "cpfn1d_j1024",
}


def _norm(name: str) -> str:
    n = name.strip().strip("`*").strip()
    return _ALIAS.get(n, n)


def parse_md(path, col_names, label):
    """-> {model: value} for the first of col_names present in the table."""
    if not path or not os.path.exists(path):
        return {}, None
    rows, hdr, col = {}, None, None
    for ln in open(path):
        if not ln.lstrip().startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if hdr is None:
            hdr = [c.lower() for c in cells]
            for want in col_names:
                if want in hdr:
                    col = hdr.index(want)
                    break
            if col is None:
                print(f"[{label}] no column from {col_names} in {path}\n"
                      f"          header was {hdr}", file=sys.stderr)
                return {}, None
            continue
        if set("".join(cells)) <= set("-: "):      # the |---|---| separator
            continue
        if col >= len(cells):
            continue
        v = cells[col]
        try:
            rows[_norm(cells[0])] = float(v.replace(",", ""))
        except ValueError:
            continue
    return rows, col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vx", required=True,
                    help="fixedq_ci_coverage.py markdown (both v(x) columns)")
    ap.add_argument("--bayes", default=None,
                    help="cate_density_metrics.py markdown, --tau-smoother none")
    ap.add_argument("--malc", default=None,
                    help="cate_density_metrics.py markdown, --tau-smoother malc")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    vx, _ = parse_md(a.vx, ["cover v(x)"], "vx")
    r1, _ = parse_md(a.vx, ["cover rho=1"], "rho1")
    nd, _ = parse_md(a.vx, ["datasets"], "datasets")
    bay, _ = parse_md(a.bayes, ["coverage95"], "bayes")
    mal, _ = parse_md(a.malc, ["coverage95"], "malc")

    if not vx:
        sys.exit(f"no rows parsed from {a.vx}")

    # Union, not intersection: a model missing from one scorer is a fact to show,
    # not a row to drop -- a silently shorter table reads as "these are all the
    # models" when it is really "these are the ones both scorers happened to emit".
    models = sorted(set(vx) | set(bay) | set(mal))
    ttl = f"  ({a.label})" if a.label else ""
    L = [f"## Fixed-query coverage, four intervals{ttl}", "",
         "| model | datasets | v(x) | v(x) rho=1 | bayesian | bayesian MALC |",
         "|" + "---|" * 6]
    def f(d, m):
        return f"{d[m]:.3f}" if m in d else "—"
    for m in models:
        n = f"{int(nd[m])}" if m in nd else "—"
        L.append(f"| {m} | {n} | {f(vx, m)} | {f(r1, m)} | {f(bay, m)} "
                 f"| {f(mal, m)} |")
    miss = [m for m in models if m not in bay and m not in mal]
    if miss:
        L += ["", "Missing from the density scorer: " + ", ".join(miss) + "."]
    L += ["",
          "v(x) and v(x) rho=1 are normal-approximation intervals from the density's",
          "MOMENTS: est +- 1.96*sqrt(v), with v the head's own coupling and",
          "(s1-s0)^2 respectively. bayesian is the CENTRAL 95% interval of the",
          "predictive tau distribution, and bayesian MALC the same after MALC at",
          "B=1000 K=1. The pairs answer different questions -- whether the claimed",
          "sampling variability matches the actual spread, versus whether the",
          "distributional shape is calibrated -- so they are not expected to agree.",
          "",
          "Coverage is over RESAMPLED OBSERVATIONAL DATASETS with the query rows",
          "held fixed, pooled over those rows. Each is its own fixed estimand, so",
          "this is a mean of single-estimand coverages, NOT coverage averaged over",
          "random queries as the main tables report."]
    txt = "\n".join(L)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt + "\n")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
