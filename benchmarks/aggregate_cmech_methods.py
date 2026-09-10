"""Per-method mean +/- SEM table for the ComplexMech PEHE benchmark.

One row per method variant, not per 1D/2D pair:

    dopfn_native, dopfn_bb
    uwyk1d-noanc,  uwyk1d-v3a,  uwyk1d-v3b
    graph2d-noanc, graph2d-v3a, graph2d-v3b
    cpfn1d
    cpfn2d-pooled, cpfn2d-log

Reports sqrt(PEHE) and the relative ATE error, each as mean +/- SEM over
datasets, alongside the null (predicting tau=0) so a row can be judged.

Two cautions the table prints for itself:

* SEM assumes roughly symmetric spread; per-dataset PEHE here is heavy tailed,
  so mean +/- SEM understates the uncertainty. The median columns of
  aggregate_cmech_1d_vs_2d.py disagree with the means at n >= 30. Use this table
  for reporting alongside that one, not instead of it.
* On the `zero` subset the true ATE is 0, so the harnesses' relative ATE error
  divides by a floor (max(|ATE|, 0.1)) rather than by the ATE. It is a scaled
  absolute error there, not a relative one, and is reported as `abs_ate/0.1`.

Usage
-----
    python benchmarks/aggregate_cmech_methods.py --root $SCRATCH/cmech_full
    python benchmarks/aggregate_cmech_methods.py --root ... --nodes 5 20
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

# (row label, output dir under --root, npz tag or None)
METHODS = [
    ("dopfn_native",   "dopfn_native",  None),
    ("dopfn_bb",       "dopfn_bb",      None),
    ("uwyk1d-noanc",   "uwyk1d",        "noanc"),
    ("uwyk1d-v3a",     "uwyk1d",        "v3a"),
    ("uwyk1d-v3b",     "uwyk1d",        "v3b"),
    ("graph2d-noanc",  "graph2d",       "noanc"),
    ("graph2d-v3a",    "graph2d",       "v3a"),
    ("graph2d-v3b",    "graph2d",       "v3b"),
    ("cpfn1d",         "cpfn1d",        None),
    ("cpfn2d-pooled",  "cpfn2d_pooled", None),
    ("cpfn2d-log",     "cpfn2d_log",    None),
]
NODE_COUNTS = (5, 20, 30, 40, 50)
SUBSETS = ("nonzero", "zero")


def _pick(z, tag, kind):
    """kind: 'pehe' or 'ate'. Returns a float or None."""
    if kind == "pehe":
        names = [f"pehe_raw_{tag}"] if tag else []
        names += ["pehe_raw", "pehe"]
    else:
        names = [f"err_raw_{tag}"] if tag else []
        names += ["err_raw", "eps_ate", "err"]
    for n in names:
        if n in z.files:
            v = np.asarray(z[n]).reshape(-1)
            if v.size:
                return float(v[0])
    return None


def load_cell(root, subdir, tag, dataset):
    """{realization: (pehe, ate_err)} for one method x cell."""
    d = os.path.join(root, subdir, dataset)
    if not os.path.isdir(d):
        return {}
    summary = os.path.join(d, "summary.npz")
    if os.path.isfile(summary):                      # dopfn_bb layout
        with np.load(summary, allow_pickle=True) as z:
            if "pehe" in z.files:
                p = np.asarray(z["pehe"], dtype=np.float64).reshape(-1)
                a = (np.asarray(z["eps_ate"], dtype=np.float64).reshape(-1)
                     if "eps_ate" in z.files else np.full(p.shape, np.nan))
                return {i: (float(p[i]), float(a[i]) if i < a.size else np.nan)
                        for i in range(p.size)}
    out = {}
    for path in sorted(glob.glob(os.path.join(d, "*.npz"))):
        digits = "".join(c for c in os.path.basename(path).rsplit("r", 1)[-1]
                         if c.isdigit())
        if not digits:
            continue
        with np.load(path, allow_pickle=True) as z:
            pe = _pick(z, tag, "pehe")
            ae = _pick(z, tag, "ate")
        if pe is not None and np.isfinite(pe):
            out[int(digits)] = (pe, ae if ae is not None else np.nan)
    return out


def null_pehe(n_nodes, subset, data_root):
    cell = os.path.join(data_root, "complexmech", f"{n_nodes}node",
                        "path_TY", "hide_0.0")
    vals = []
    for p in sorted(glob.glob(os.path.join(cell, "r*.npz"))):
        with np.load(p) as z:
            tau = np.asarray(z["true_cate"], dtype=np.float64)
        tau = tau[tau != 0] if subset == "nonzero" else tau[tau == 0]
        if tau.size:
            vals.append(float(np.sqrt((tau ** 2).mean())))
    return (float(np.mean(vals)), float(np.std(vals, ddof=1) / np.sqrt(len(vals)))) \
        if vals else (float("nan"), float("nan"))


def ms(v):
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return float("nan"), float("nan"), 0
    sem = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else float("nan")
    return float(v.mean()), sem, int(v.size)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--data-root", default=os.environ.get(
        "UWYK_FIG34_DATA",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "UWYK_Fig3_4", "data")))
    ap.add_argument("--nodes", type=int, nargs="+", default=list(NODE_COUNTS))
    ap.add_argument("--subsets", nargs="+", default=list(SUBSETS), choices=SUBSETS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for n in args.nodes:
        for subset in args.subsets:
            ds = f"CMECH_n{n}_{subset}"
            nl_m, nl_s = null_pehe(n, subset, args.data_root)
            rows.append({"method": "— null (predict 0) —", "n_nodes": n,
                         "subset": subset, "n_datasets": "",
                         "pehe": nl_m, "pehe_sem": nl_s,
                         "ate_err": float("nan"), "ate_sem": float("nan")})
            for label, subdir, tag in METHODS:
                got = load_cell(args.root, subdir, tag, ds)
                if not got:
                    continue
                pm, ps, k = ms([v[0] for v in got.values()])
                am, asem, _ = ms([v[1] for v in got.values()])
                rows.append({"method": label, "n_nodes": n, "subset": subset,
                             "n_datasets": k, "pehe": pm, "pehe_sem": ps,
                             "ate_err": am, "ate_sem": asem})

    if not any(r["n_datasets"] != "" for r in rows):
        raise SystemExit(f"no method results under {args.root}")

    L = ["# ComplexMech — per-method PEHE and ATE error (mean +/- SEM)", "",
         "`null` is the score of predicting tau=0 everywhere: a method matching it",
         "has no skill. On the `zero` subset the null is 0 by construction, so PEHE",
         "there is a pure false-positive measure.", "",
         "ATE error on the `zero` subset is NOT relative — true ATE is 0, so the",
         "harnesses divide by a floor of 0.1. Read it as |ATE_hat| / 0.1 there.", "",
         "SEM assumes symmetric spread; per-dataset PEHE is heavy tailed, so these",
         "intervals are optimistic. Cross-check against the median columns of",
         "aggregate_cmech_1d_vs_2d.py, which disagree with the means at n >= 30.", "",
         "| method | n | subset | n_ds | sqrt(PEHE) | ATE err |",
         "|---|---|---|---|---|---|"]
    for r in rows:
        pe = "—" if not np.isfinite(r["pehe"]) else (
            f"{r['pehe']:.4f} ± {r['pehe_sem']:.4f}"
            if np.isfinite(r["pehe_sem"]) else f"{r['pehe']:.4f}")
        ae = "—" if not np.isfinite(r["ate_err"]) else (
            f"{r['ate_err']:.4f} ± {r['ate_sem']:.4f}"
            if np.isfinite(r["ate_sem"]) else f"{r['ate_err']:.4f}")
        L.append(f"| {r['method']} | {r['n_nodes']} | {r['subset']} | "
                 f"{r['n_datasets']} | {pe} | {ae} |")
    md = "\n".join(L) + "\n"
    print(md)
    with open(os.path.join(args.root, "cmech_methods.json"), "w") as f:
        json.dump(rows, f, indent=2, default=str)
    out = args.out or os.path.join(args.root, "cmech_methods.md")
    with open(out, "w") as f:
        f.write(md)
    print(f"[written] {out}")


if __name__ == "__main__":
    main()
