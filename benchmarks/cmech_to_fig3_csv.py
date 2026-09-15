"""Convert ComplexMech results into the tidy CSV plot_fig3_dotgrid.py expects.

The two pipelines do not share a schema:

    case studies                  cmech
    ------------------------      ---------------------------------
    d                             n_nodes
    case (6 case studies)         subset (nonzero / zero / total)
    N as a column                 N<ctx>/ SUBDIRECTORIES
    tidy CSV                      markdown + cmech_methods.json per context
    uwyk_noanc, graph2d_v3a       uwyk1d-noanc, graph2d-v3a

This walks <root>/N<ctx>/ , runs aggregate_cmech_methods._build per context to
get its rows, and emits one CSV row per (n_nodes, N, method).

AXIS CHOICE. cmech varies n_nodes AND N, but only has 1-3 subsets, so putting
subsets on the columns wastes the layout. Instead:

    rows    <- d      = n_nodes
    columns <- case   = "N=<ctx>"      (a label, not a case study)
    N       <- a sentinel so --n passes the filter
    shift   <- "cmech"

giving the n_nodes x N grid directly. --by-subset puts subsets on the columns
instead, for a single context.

    python benchmarks/cmech_to_fig3_csv.py --root $SCRATCH/cmech_dens \
        --out cmech_fig3.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import aggregate_cmech_methods as A          # noqa: E402

SENTINEL_N = 1000          # plot_fig3_dotgrid filters on N; keep one value

# cmech label -> the model key plot_fig3_dotgrid's PAIRS expects.
MODEL_MAP = {
    "dopfn_native": "dopfn_native",
    "dopfn_bb": "dopfn_bb",
    "uwyk1d-noanc": "uwyk_noanc",
    "uwyk1d-v3a": "uwyk_v3a",
    "uwyk1d-v3b": "uwyk_v3b",
    "graph2d-noanc": "graph2d_noanc",
    "graph2d-v3a": "graph2d_v3a",
    "graph2d-v3b": "graph2d_v3b",
    "cpfn1d": "cpfn1d_perarm",
    "cpfn2d": "cpfn2d_pooled",
}

HEADER = ("shift,d,N,case,model,"
          "pehe_raw,pehe_raw_sem,pehe_raw_med,"
          "l1_raw,l1_raw_sem,l1_raw_med,n\n")


def _stat(s, key, default=float("nan")):
    if not s:
        return default
    v = s.get(key)
    return default if v is None else float(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="holds N<ctx>/ subdirs")
    ap.add_argument("--data-root", default=os.environ.get("CMECH_DATA_ROOT", ""))
    ap.add_argument("--nodes", type=int, nargs="+", default=list(A.NODE_COUNTS))
    ap.add_argument("--subset", default="nonzero")
    ap.add_argument("--by-subset", action="store_true",
                    help="columns = subset for ONE context (--context) instead "
                         "of columns = N")
    ap.add_argument("--context", type=int, default=None)
    ap.add_argument("--subsets", nargs="+", default=["nonzero", "zero", "total"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ctxs = sorted((int(os.path.basename(d)[1:]), d)
                  for d in glob.glob(os.path.join(a.root, "N*"))
                  if os.path.isdir(d) and os.path.basename(d)[1:].isdigit())
    if not ctxs:
        raise SystemExit(f"no N<ctx>/ subdirs under {a.root}")
    if a.context is not None:
        ctxs = [(c, d) for c, d in ctxs if c == a.context]
        if not ctxs:
            raise SystemExit(f"context {a.context} not found")

    subsets = a.subsets if a.by_subset else [a.subset]
    out_rows, skipped = [], 0
    for ctx, cdir in ctxs:
        ns = argparse.Namespace(root=cdir, data_root=a.data_root,
                                nodes=a.nodes, subsets=subsets,
                                out=None, all_contexts=False, _nested=True)
        A._build(ns)                       # writes cmech_methods.json in cdir
        import json
        rows = json.load(open(os.path.join(cdir, "cmech_methods.json")))
        for r in rows:
            key = MODEL_MAP.get(r["method"])
            if key is None:                # the null row, or an unmapped label
                skipped += 1
                continue
            case = (r["subset"] if a.by_subset else f"N={ctx}")
            out_rows.append((
                "cmech", int(r["n_nodes"]), SENTINEL_N, case, key,
                _stat(r["pehe"], "mean"), _stat(r["pehe"], "sem"),
                _stat(r["pehe"], "median"),
                _stat(r["ate"], "mean"), _stat(r["ate"], "sem"),
                _stat(r["ate"], "median"),
                int(_stat(r["pehe"], "n", 0))))

    if not out_rows:
        raise SystemExit("no mappable rows produced")
    with open(a.out, "w") as f:
        f.write(HEADER)
        for r in out_rows:
            f.write("%s,%d,%d,%s,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d\n" % r)
    cols = sorted({r[3] for r in out_rows})
    print(f"[cmech->fig3] {len(out_rows)} rows -> {a.out}")
    print(f"  rows (d)  = {sorted({r[1] for r in out_rows})}")
    print(f"  columns   = {cols}")
    print(f"  models    = {sorted({r[4] for r in out_rows})}")
    if skipped:
        print(f"  skipped {skipped} unmapped rows (null baseline / unknown label)")


if __name__ == "__main__":
    main()
