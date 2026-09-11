"""Progress / health check for the ComplexMech context-sweep array.

Scans $OUT_ROOT/N<ctx>/<model>/<dataset>/ and reports, per context size, how
many realizations each (model, node count) cell has produced — so a stalled,
failed, or silently-thinned cell is visible without reading 210 logs.

Flags two failure modes the aggregator would otherwise hide:

  MISSING   cell has no output at all (task failed, or has not started)
  THIN      cell has noticeably fewer realizations than the benchmark holds.
            At small N this is usually treatment-arm collapse: subsampling
            1000 rows down to 50 can leave a realization with zero treated
            units, and any harness slicing on T==1 then yields nan. That is a
            sampling artifact, not a modelling result.

Usage
-----
    python benchmarks/cmech_progress.py --root $SCRATCH/cmech_v3
    python benchmarks/cmech_progress.py --root ... --verbose   # per-cell counts
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

MODELS = ("dopfn_native", "dopfn_bb", "uwyk1d", "graph2d",
          "cpfn1d", "cpfn2d_pooled", "cpfn2d_log")
NODES = (5, 10, 20, 30, 40, 50)
CONTEXTS = (50, 100, 250, 500, 1000)


def cell_count(root, ctx, model, n, subset):
    d = os.path.join(root, f"N{ctx}", model, f"CMECH_n{n}_{subset}")
    if not os.path.isdir(d):
        return None
    summary = os.path.join(d, "summary.npz")
    if os.path.isfile(summary):                      # dopfn_bb layout
        with np.load(summary, allow_pickle=True) as z:
            if "pehe" in z.files:
                v = np.asarray(z["pehe"]).reshape(-1)
                return int(np.isfinite(v).sum())
    n_files = len(glob.glob(os.path.join(d, "*r[0-9]*.npz")))
    return n_files if n_files else None


def expected(n, data_root, subset):
    """How many realizations the benchmark actually holds for this cell."""
    cell = os.path.join(data_root, "complexmech", f"{n}node", "path_TY", "hide_0.0")
    tot = 0
    for p in glob.glob(os.path.join(cell, "r*.npz")):
        with np.load(p) as z:
            tau = np.asarray(z["true_cate"])
            m = (tau != 0) if subset == "nonzero" else (tau == 0)
            if m.any():
                tot += 1
    return tot


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--data-root", default=os.environ.get(
        "UWYK_FIG34_DATA",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "UWYK_Fig3_4", "data")))
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--subset", default="nonzero", choices=["nonzero", "zero"],
                    help="which query subset to audit (run once per subset)")
    ap.add_argument("--thin-frac", type=float, default=0.9,
                    help="flag a cell as THIN below this fraction of expected")
    args = ap.parse_args()

    exp = {n: expected(n, args.data_root, args.subset) for n in NODES}
    print(f"expected realizations per node count ({args.subset} subset): "
          + "  ".join(f"n={n}:{exp[n]}" for n in NODES) + "\n")

    done = total = 0
    missing, thin = [], []
    for ctx in CONTEXTS:
        print(f"── N = {ctx} " + "─" * 52)
        print(f"{'model':<16}" + "".join(f"{f'n={n}':>8}" for n in NODES))
        for m in MODELS:
            cells = []
            for n in NODES:
                total += 1
                c = cell_count(args.root, ctx, m, n, args.subset)
                if c is None:
                    cells.append("  --")
                    missing.append((ctx, m, n))
                else:
                    done += 1
                    e = exp[n] or 1
                    mark = "" if c >= args.thin_frac * e else "!"
                    if mark:
                        thin.append((ctx, m, n, c, e))
                    cells.append(f"{c}{mark}".rjust(4))
            print(f"{m:<16}" + "".join(c.rjust(8) for c in cells))
        print()

    print(f"cells complete: {done}/{total}"
          + (f"   ({100*done/total:.0f}%)" if total else ""))
    if missing:
        print(f"\nMISSING ({len(missing)}) — task failed or not yet run:")
        for ctx, m, n in missing[:15]:
            print(f"   N={ctx:<5} {m:<16} n={n}")
        if len(missing) > 15:
            print(f"   ... and {len(missing)-15} more")
    if thin:
        print(f"\nTHIN ({len(thin)}) — fewer realizations than the benchmark holds.")
        print("   At small N this is usually treatment-arm collapse (a subsampled")
        print("   context with zero treated units), not a modelling result.")
        for ctx, m, n, c, e in thin[:15]:
            print(f"   N={ctx:<5} {m:<16} n={n:<3} {c}/{e}")
        if len(thin) > 15:
            print(f"   ... and {len(thin)-15} more")
    if not missing and not thin:
        print("\nall cells present and full.")


if __name__ == "__main__":
    main()
