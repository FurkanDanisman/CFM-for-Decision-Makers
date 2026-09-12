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
    ("cpfn2d",         "cpfn2d_pooled", None),
    # cpfn2d_log is not reported: the log target transform is not affine in tau,
    # so it cannot emit a CATE density, and its point estimates were within
    # noise of pooled everywhere. The sbatch still runs it (task numbering is
    # unchanged); it is simply left out of the tables.
]
NODE_COUNTS = (5, 10, 20, 30, 40, 50)


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
                    out[i] = (float(p[i]),
                              float(ap[i]) if i < ap.size else np.nan,
                              float(at[i]) if i < at.size else np.nan)
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
        out[int(digits)] = (pe, ap if ap is not None else np.nan,
                            at if at is not None else np.nan)
    return out


def subset_sources(n_nodes, subset, data_root):
    """Adapter index -> source realization, mirroring UWYKFig34Dataset's skipping.

    The eval harnesses name their npz by the ADAPTER index (`for r in
    range(ds.n_tables)`), and the adapter drops realizations whose subset is
    empty. The nonzero and zero subsets therefore drop different realizations,
    so their index r means a different dataset in each. Combining them requires
    mapping both back to the source realization, which this reconstructs from
    the benchmark data itself.
    """
    cell = os.path.join(data_root, "complexmech", f"{n_nodes}node",
                        "path_TY", "hide_0.0")
    src, counts = [], {}
    for p in sorted(glob.glob(os.path.join(cell, "r*.npz")),
                    key=lambda q: int(os.path.basename(q)[1:-4])):
        i = int(os.path.basename(p)[1:-4])
        with np.load(p) as z:
            tau = np.asarray(z["true_cate"], dtype=np.float64)
        n1, n0 = int((tau != 0).sum()), int((tau == 0).sum())
        counts[i] = (n0, n1)
        if (n1 if subset == "nonzero" else n0) > 0:
            src.append(i)
    return src, counts


def combine_total(got_nz, got_z, n_nodes, data_root):
    """Pool the disjoint zero / nonzero query sets back into all queries.

    PEHE is an RMS over queries, so it pools exactly:
        pehe_all^2 = (n0*pehe_zero^2 + n1*pehe_nonzero^2) / (n0 + n1)
    and the ATE is a plain mean, so it pools by the same weights. Realizations
    present in only one subset keep that subset's value.
    """
    src_nz, counts = subset_sources(n_nodes, "nonzero", data_root)
    src_z, _ = subset_sources(n_nodes, "zero", data_root)
    by_src_nz = {s: got_nz[i] for i, s in enumerate(src_nz) if i in got_nz}
    by_src_z = {s: got_z[i] for i, s in enumerate(src_z) if i in got_z}

    out = {}
    for s in sorted(set(by_src_nz) | set(by_src_z)):
        n0, n1 = counts.get(s, (0, 0))
        a, b = by_src_nz.get(s), by_src_z.get(s)
        if a is not None and b is not None and (n0 + n1) > 0:
            pe = float(np.sqrt((n1 * a[0] ** 2 + n0 * b[0] ** 2) / (n0 + n1)))
            ap = (n1 * a[1] + n0 * b[1]) / (n0 + n1)
            at = (n1 * a[2] + n0 * b[2]) / (n0 + n1)
        else:
            pe, ap, at = (a or b)
        out[s] = (pe, ap, at)
    return out


def null_row(n_nodes, subset, data_root):
    """Score of predicting tau=0: PEHE = RMS(tau), ATE L1 = |true ATE|."""
    cell = os.path.join(data_root, "complexmech", f"{n_nodes}node",
                        "path_TY", "hide_0.0")
    pe, l1 = [], []
    for p in sorted(glob.glob(os.path.join(cell, "r*.npz"))):
        with np.load(p) as z:
            tau = np.asarray(z["true_cate"], dtype=np.float64)
        if subset == "nonzero":
            tau = tau[tau != 0]
        elif subset == "zero":
            tau = tau[tau == 0]
        # subset == "total": keep every query
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
                    choices=["nonzero", "zero", "total"],
                    help="`total` pools zero+nonzero exactly; it needs BOTH "
                         "subsets to have been run (SUBSET=zero in the sbatch).")
    ap.add_argument("--out", default=None)
    ap.add_argument("--all-contexts", action="store_true",
                    help="--root holds N<ctx>/ subdirs (as the sbatch writes): "
                         "emit one table per context size.")
    args = ap.parse_args()

    if args.all_contexts:
        subs = sorted(
            (int(os.path.basename(d)[1:]), d)
            for d in glob.glob(os.path.join(args.root, "N*"))
            if os.path.isdir(d) and os.path.basename(d)[1:].isdigit())
        if not subs:
            raise SystemExit(f"no N<ctx>/ subdirs under {args.root}")
        parts = []
        for ctx, d in subs:
            sub = argparse.Namespace(**vars(args))
            sub.root, sub.all_contexts, sub.out = d, False, None
            parts.append(f"\n\n## Context N = {ctx}\n\n" + _build(sub))
        md = "# ComplexMech — per-method tables by context size\n" + "".join(parts)
        out = args.out or os.path.join(args.root, "cmech_methods_all_contexts.md")
        with open(out, "w") as f:
            f.write(md)
        print(md)
        print(f"[written] {out}")
        return

    print(_build(args))


def _build(args):
    rows = []
    for subset in args.subsets:
        for n in args.nodes:
            ds = f"CMECH_n{n}_{subset}"
            npe, nl1 = null_row(n, subset, args.data_root)
            rows.append({"method": "— null (predict 0) —", "n_nodes": n,
                         "subset": subset, "pehe": stats(npe), "ate": stats(nl1)})
            for label, subdir, tag in METHODS:
                if subset == "total":
                    gz = load_cell(args.root, subdir, tag, f"CMECH_n{n}_zero")
                    gn = load_cell(args.root, subdir, tag, f"CMECH_n{n}_nonzero")
                    if not gz or not gn:
                        continue
                    got = combine_total(gn, gz, n, args.data_root)
                else:
                    got = load_cell(args.root, subdir, tag, ds)
                if not got:
                    continue
                rows.append({"method": label, "n_nodes": n, "subset": subset,
                             "pehe": stats([v[0] for v in got.values()]),
                             "ate": stats([abs(v[1] - v[2])
                                           for v in got.values()])})

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
    with open(os.path.join(args.root, "cmech_methods.json"), "w") as f:
        json.dump(rows, f, indent=2, default=str)
    if args.out is not None or not getattr(args, "_nested", False):
        out = args.out or os.path.join(args.root, "cmech_methods.md")
        with open(out, "w") as f:
            f.write(md)
    return md


if __name__ == "__main__":
    main()
