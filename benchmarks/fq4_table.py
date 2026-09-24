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
        # "0.1304 ± 0.0046" -> 0.1304: cate_density_metrics reports mean ± SEM in
        # its length column, and a bare float() on that raises and drops the row.
        v = cells[col].replace(",", "").split("±")[0].split("+-")[0].strip()
        try:
            rows[_norm(cells[0])] = float(v)
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
    ap.add_argument("--sdy", default=None,
                    help="this cell's sd(Y) json from fq4_cell_sdy.py. Carried into "
                         "the output so lengths stay comparable across datasets of "
                         "different scale after the raw cell is deleted.")
    ap.add_argument("--n-models", type=int, default=None,
                    help="how many model directories the cell held when scored. Stored "
                         "so a later pass can tell a partial scoring from a complete "
                         "one and re-score instead of accepting the partial table.")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json-out", default=None,
                    help="machine-readable copy of this cell's row values, for the "
                         "across-realization aggregator to average")
    a = ap.parse_args()

    vx, _ = parse_md(a.vx, ["cover v(x)"], "vx")
    r1, _ = parse_md(a.vx, ["cover rho=1"], "rho1")
    nd, _ = parse_md(a.vx, ["datasets"], "datasets")
    s0, _ = parse_md(a.vx, ["sd(y0)"], "sd0")
    s1, _ = parse_md(a.vx, ["sd(y1)"], "sd1")
    bay, _ = parse_md(a.bayes, ["coverage95"], "bayes")
    mal, _ = parse_md(a.malc, ["coverage95"], "malc")
    # Coverage without length is not interpretable: an interval can cover by being
    # correct or by being wide, and only the pair distinguishes them.
    bayl, _ = parse_md(a.bayes, ["length"], "bayes-len")
    mall, _ = parse_md(a.malc, ["length"], "malc-len")
    wvx, _ = parse_md(a.vx, ["mean width"], "vx-len")
    wr1, _ = parse_md(a.vx, ["mean width rho=1"], "rho1-len")

    if not vx:
        sys.exit(f"no rows parsed from {a.vx}")

    # Union, not intersection: a model missing from one scorer is a fact to show,
    # not a row to drop -- a silently shorter table reads as "these are all the
    # models" when it is really "these are the ones both scorers happened to emit".
    models = sorted(set(vx) | set(bay) | set(mal))
    ttl = f"  ({a.label})" if a.label else ""
    L = [f"## Fixed-query coverage, four intervals{ttl}", "",
         "| model | datasets | sd(Y0) | sd(Y1) | v(x) | len | v(x) rho=1 | len "
         "| bayesian | len | bayesian MALC | len |", "|" + "---|" * 12]
    def f(d, m):
        return f"{d[m]:.3f}" if m in d else "—"
    for m in models:
        n = f"{int(nd[m])}" if m in nd else "—"
        def g(d):
            return f"{d[m]:.4f}" if m in d else "—"
        L.append(f"| {m} | {n} | {g(s0)} | {g(s1)} "
                 f"| {f(vx, m)} | {g(wvx)} | {f(r1, m)} | {g(wr1)} "
                 f"| {f(bay, m)} | {g(bayl)} | {f(mal, m)} | {g(mall)} |")
    miss = [m for m in models if m not in bay and m not in mal]
    if miss:
        L += ["", "Missing from the density scorer: " + ", ".join(miss) + "."]
    L += ["",
          "bayesian and bayesian MALC are the project's standard CATE coverage --",
          "the same cov raw / cov T that cate_density_metrics reports for every",
          "other table here, at B=1000 K=1. Nothing new is computed for them.",
          "",
          "Each coverage is followed by its mean interval LENGTH. Coverage alone",
          "cannot be read: an interval can cover because it is correct or because it",
          "is wide, and only the pair separates the two. The v(x) lengths come from",
          "the same normal approximation as their coverage, the bayesian lengths from",
          "cate_density_metrics; they are in the outcome's units, so they compare",
          "across models on one dataset but not across datasets of different scale.",
          "",
          "sd(Y0) / sd(Y1) are the per-arm predictive spreads averaged over the",
          "replicates and frozen queries; v(x) rho=1 is (sd(Y1)-sd(Y0))^2, so a",
          "near-zero width there is visible as two nearly equal arm spreads.",
          "",
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
    sdy = None
    if a.sdy and os.path.exists(a.sdy):
        import json as _j
        try:
            sdy = _j.load(open(a.sdy))
        except Exception:
            sdy = None
    if a.json_out:
        import json
        with open(a.json_out, "w") as fh:
            json.dump({"label": a.label, "sd_Y": sdy, "n_models": a.n_models,
                       "models": {m: {"datasets": nd.get(m), "sd_y0": s0.get(m),
                                      "sd_y1": s1.get(m),
                                      "cover_vx": vx.get(m), "len_vx": wvx.get(m),
                                      "cover_rho1": r1.get(m), "len_rho1": wr1.get(m),
                                      "bayesian": bay.get(m), "len_bayesian": bayl.get(m),
                                      "bayesian_malc": mal.get(m),
                                      "len_bayesian_malc": mall.get(m)}
                                  for m in models}}, fh, indent=2)
        print(f"wrote {a.json_out}")
    txt = "\n".join(L)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt + "\n")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
