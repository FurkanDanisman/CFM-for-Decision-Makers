"""Per-method table for the ComplexMech benchmark: mean ± std and mean ± SEM.

One row per method variant:

    dopfn_native, dopfn_bb
    uwyk1d-noanc,  uwyk1d-v3a,  uwyk1d-v3b
    graph2d-noanc, graph2d-v3a, graph2d-v3b
    cpfn1d
    cpfn2d-pooled, cpfn2d-log

Metrics, each reported as mean ± std (spread across datasets) and mean ± SEM
(uncertainty in the mean):

    sqrt(PEHE)   sqrt(mean_i (tau_hat_i - tau_i)^2), per dataset
    ATE L1       |ATE_hat - ATE_true|, per dataset -- an absolute error, NOT the
                 harnesses' relative `err_raw` (which divides by a 0.1 floor and
                 is meaningless when the true ATE is near zero)

Only the `nonzero` subset is reported by default: on the `zero` subset tau is
identically 0, so PEHE there measures hallucinated effect rather than CATE
accuracy. Pass --subsets zero to see it.

The `null` row is the score of predicting tau=0 everywhere. A method matching it
has no skill, so read every row against it rather than against 0.

Caveat kept visible: per-dataset PEHE is heavy tailed, so mean ± SEM is
optimistic and the mean can disagree with the median (it does, at n >= 30 --
see aggregate_cmech_1d_vs_2d.py). The std column is there to make that spread
visible rather than hidden behind a narrow SEM.

Usage
-----
    python benchmarks/aggregate_cmech_methods.py --root $SCRATCH/cmech_v2
    python benchmarks/aggregate_cmech_methods.py --root ... --nodes 5 20
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

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


def _first(z, names):
    for n in names:
        if n in z.files:
            v = np.asarray(z[n]).reshape(-1)
            if v.size:
                return float(v[0])
    return None


def load_cell(root, subdir, tag, dataset):
    """{realization: (sqrt_pehe, ate_l1)} for one method x cell."""
    d = os.path.join(root, subdir, dataset)
    if not os.path.isdir(d):
        return {}

    summary = os.path.join(d, "summary.npz")           # dopfn_bb layout
    if os.path.isfile(summary):
        with np.load(summary, allow_pickle=True) as z:
            if "pehe" in z.files:
                p = np.asarray(z["pehe"], dtype=np.float64).reshape(-1)
                ap = np.asarray(z["ate_pred"], dtype=np.float64).reshape(-1) \
                    if "ate_pred" in z.files else np.full(p.shape, np.nan)
                at = np.asarray(z["true_ate"], dtype=np.float64).reshape(-1) \
                    if "true_ate" in z.files else np.full(p.shape, np.nan)
                out = {}
                for i in range(p.size):
                    l1 = (abs(ap[i] - at[i])
                          if i < ap.size and i < at.size else np.nan)
                    out[i] = (float(p[i]), float(l1))
                return out

    out = {}
    for path in sorted(glob.glob(os.path.join(d, "*.npz"))):
        digits = "".join(c for c in os.path.basename(path).rsplit("r", 1)[-1]
                         if c.isdigit())
        if not digits:
            continue
        with np.load(path, allow_pickle=True) as z:
            pe = _first(z, ([f"pehe_raw_{tag}"] if tag else []) + ["pehe_raw", "pehe"])
            ap = _first(z, ([f"ate_raw_{tag}"] if tag else []) + ["ate_raw", "ate_pred"])
            at = _first(z, ["true_ate"])
        if pe is None or not np.isfinite(pe):
            continue
        l1 = abs(ap - at) if (ap is not None and at is not None) else np.nan
        out[int(digits)] = (pe, l1)
    return out


def null_row(n_nodes, subset, data_root):
    """Score of predicting tau=0: PEHE = RMS(tau), ATE L1 = |true ATE|."""
    cell = os.path.join(data_root, "complexmech", f"{n_nodes}node",
                        "path_TY", "hide_0.0")
    pe, l1 = [], []
    for p in sorted(glob.glob(os.path.join(cell, "r*.npz"))):
        with np.load(p) as z:
            tau = np.asarray(z["true_cate"], dtype=np.float64)
        tau = tau[tau != 0] if subset == "nonzero" else tau[tau == 0]
        if tau.size:
            pe.append(float(np.sqrt((tau ** 2).mean())))
            l1.append(abs(float(tau.mean())))
    return pe, l1


def stats(v):
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return None
    sd = float(v.std(ddof=1)) if v.size > 1 else float("nan")
    return {"mean": float(v.mean()), "std": sd,
            "sem": sd / np.sqrt(v.size) if v.size > 1 else float("nan"),
            "n": int(v.size)}


def _fmt(s, key):
    if s is None or not np.isfinite(s["mean"]):
        return "—"
    if not np.isfinite(s[key]):
        return f"{s['mean']:.4f}"
    return f"{s['mean']:.4f} ± {s[key]:.4f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--data-root", default=os.environ.get(
        "UWYK_FIG34_DATA",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "UWYK_Fig3_4", "data")))
    ap.add_argument("--nodes", type=int, nargs="+", default=list(NODE_COUNTS))
    ap.add_argument("--subsets", nargs="+", default=["nonzero"],
                    choices=["nonzero", "zero"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for subset in args.subsets:
        for n in args.nodes:
            ds = f"CMECH_n{n}_{subset}"
            npe, nl1 = null_row(n, subset, args.data_root)
            rows.append({"method": "— null (predict 0) —", "n_nodes": n,
                         "subset": subset, "pehe": stats(npe), "ate": stats(nl1)})
            for label, subdir, tag in METHODS:
                got = load_cell(args.root, subdir, tag, ds)
                if not got:
                    continue
                rows.append({"method": label, "n_nodes": n, "subset": subset,
                             "pehe": stats([v[0] for v in got.values()]),
                             "ate": stats([v[1] for v in got.values()])})

    if not any(r["method"] != "— null (predict 0) —" for r in rows):
        raise SystemExit(f"no method results under {args.root}")

    L = ["# ComplexMech — per-method sqrt(PEHE) and ATE L1", "",
         "Subset: " + ", ".join(args.subsets) +
         ".  `null` = predicting tau=0 everywhere; a row matching it has no skill.", "",
         "ATE L1 is |ATE_hat - ATE_true|, an absolute error — not the harnesses'",
         "relative `err_raw`, which divides by a 0.1 floor.", "",
         "std = spread across datasets; SEM = uncertainty in the mean. Per-dataset",
         "PEHE is heavy tailed, so SEM is optimistic and the mean can disagree with",
         "the median (it does at n >= 30 — see aggregate_cmech_1d_vs_2d.py).", "",
         "| method | n | n_ds | sqrt(PEHE) mean±std | sqrt(PEHE) mean±SEM | "
         "ATE L1 mean±std | ATE L1 mean±SEM |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        nds = r["pehe"]["n"] if r["pehe"] else 0
        L.append(f"| {r['method']} | {r['n_nodes']} | {nds} | "
                 f"{_fmt(r['pehe'],'std')} | {_fmt(r['pehe'],'sem')} | "
                 f"{_fmt(r['ate'],'std')} | {_fmt(r['ate'],'sem')} |")
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
