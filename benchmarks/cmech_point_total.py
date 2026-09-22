#!/usr/bin/env python
"""Correct PEHE / L1_ATE for ComplexMech `total` (= nonzero + zero queries).

WHY THIS EXISTS. point_raw_em.run_cell forms `total` by pairing the nonzero and
zero cell dirs BY POSITION:

    n_real = min(len(fs) for fs in per_dir)      # min(100, 9) = 9
    for r in range(n_real): ... fs[r] ...        # nonzero/r000 with zero/r000

Its docstring assumes both dirs hold the same realizations in the same order.
That is false whenever only some realizations have an exactly-zero-tau query:
UWYKFig34Dataset keeps only realizations whose requested subset is non-empty, so
CMECH_n5_nonzero is indexed r000..r099 over all 100 realizations while
CMECH_n5_zero is indexed r000..r008 over the 9 that have a zero-tau query. The
position pairing therefore (a) truncates to 9 realizations and (b) joins queries
from UNRELATED realizations.

WHAT THIS DOES INSTEAD. Rebuild each subset's realization -> source mapping from
the DATA, exactly the way the dataset does (a realization belongs to `nonzero` if
any true_cate != 0, to `zero` if any == 0), then group dump files by SOURCE
realization and concatenate that realization's queries. Every query appears once,
so PEHE is the root-mean-square over the true union and the per-realization ATE is
a mean over all of that realization's queries.

    python benchmarks/cmech_point_total.py \
        --dumps $SCRATCH/cmech_dumps_rho99 --data $SCRATCH/cmech_data_rho99 \
        --nodes 5 10 20 30 40 50 --out total.md
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "UWYK_Fig3_4"))
sys.path.insert(0, os.path.join(_REPO, "realcause_eval"))
sys.path.insert(0, os.path.join(_REPO, "realcause_eval", "Table1"))

from cate_density_metrics import METHODS, _resolve_dir          # noqa: E402
from point_raw_em import cate_for_file, _files_in               # noqa: E402


def subset_sources(data_cell):
    """(nonzero_sources, zero_sources): source realization index per subset, in the
    order the dataset enumerates them. Mirrors UWYKFig34Dataset's skip rule."""
    files = sorted(glob.glob(os.path.join(data_cell, "r*.npz")),
                   key=lambda p: int(os.path.basename(p)[1:-4]))
    nz, ze = [], []
    for p in files:
        src = int(os.path.basename(p)[1:-4])
        try:
            with np.load(p) as z:
                t = np.asarray(z["true_cate"]).ravel()
        except Exception:
            continue
        if (t != 0).any():
            nz.append(src)
        if (t == 0).any():
            ze.append(src)
    return nz, ze


def score(dumps_model, subdir, tag, data_cell, n, mode="raw", ate_metric="l1"):
    """-> (n_real, pehe, pehe_se, ate, ate_se) over the query union per realization."""
    nz_src, ze_src = subset_sources(data_cell)
    by_src: dict[int, list[str]] = {}
    for sub, srcs in (("nonzero", nz_src), ("zero", ze_src)):
        d = _resolve_dir(os.path.join(dumps_model, "N1000"), subdir,
                         f"CMECH_n{n}_{sub}")
        if not d:
            continue
        fs = _files_in(d)
        if len(fs) != len(srcs):
            # Loud, not silent: the mapping is only valid if the dump count matches
            # what the data says the subset contains.
            print(f"    WARN {os.path.basename(dumps_model)} n={n} {sub}: "
                  f"{len(fs)} dumps but data says {len(srcs)} realizations",
                  file=sys.stderr)
        for k, f in enumerate(fs):
            if k < len(srcs):
                by_src.setdefault(srcs[k], []).append(f)
    if not by_src:
        return None
    pehe, ate = [], []
    for src in sorted(by_src):
        taus, truths = [], []
        for f in by_src[src]:
            try:
                got = cate_for_file(f, tag, mode)
            except Exception:
                got = None
            if got is None:
                continue
            taus.append(np.asarray(got[0]).ravel())
            truths.append(np.asarray(got[1]).ravel())
        if not taus:
            continue
        tau = np.concatenate(taus); tru = np.concatenate(truths)
        if tau.size == 0 or not np.isfinite(tau).all():
            continue
        pehe.append(float(np.sqrt(np.mean((tau - tru) ** 2))))
        d = float(abs(tau.mean() - tru.mean()))
        if ate_metric == "rel":
            d /= max(abs(float(tru.mean())), 0.1)
        ate.append(d)
    if not pehe:
        return None
    p = np.asarray(pehe); a = np.asarray(ate)
    # PEHE pools as an RMS across realizations because it is itself an RMSE.
    return (len(p), float(np.sqrt(np.mean(p ** 2))),
            float(p.std(ddof=1) / np.sqrt(p.size)) if p.size > 1 else float("nan"),
            float(a.mean()),
            float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--nodes", type=int, nargs="+", default=[5, 10, 20, 30, 40, 50])
    ap.add_argument("--regime", default="path_TY")
    ap.add_argument("--hide", type=float, default=0.0)
    ap.add_argument("--mode", default="raw", choices=["raw", "em"])
    ap.add_argument("--ate-metric", default="l1", choices=["l1", "rel"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    L = []
    for n in a.nodes:
        cell = os.path.join(a.data, "complexmech", f"{n}node", a.regime,
                            f"hide_{a.hide}")
        nz, ze = subset_sources(cell)
        rows = []
        for md in sorted(d for d in glob.glob(f"{a.dumps}/*") if os.path.isdir(d)):
            for label, subdir, tag in METHODS:
                r = score(md, subdir, tag, cell, n, a.mode, a.ate_metric)
                if r:
                    rows.append((f"{label}", *r))
        if not rows:
            continue
        L += ["", f"## ComplexMech total — n={n}   "
                  f"(union of {len(nz)} nonzero + {len(ze)} zero subset entries "
                  f"over {len(set(nz) | set(ze))} realizations)", "",
              "| model | realizations | PEHE | L1_ATE |" if a.ate_metric == "l1"
              else "| model | realizations | PEHE | eps_ATE |",
              "|---|---|---|---|"]
        for nm, k, p, pse, at, ase in sorted(rows, key=lambda t: t[2]):
            L.append(f"| {nm} | {k} | {p:.4f} ± {pse:.4f} | {at:.4f} ± {ase:.4f} |")
    L += ["", "Queries are grouped by SOURCE realization, not by position in the",
          "cell dir: the nonzero and zero subsets enumerate DIFFERENT realization",
          "sets, so point_raw_em's positional pairing truncated to min(len) and",
          "joined unrelated realizations. Every query is counted exactly once here.",
          "PEHE pools as an RMS across realizations; ATE error as a mean."]
    txt = "\n".join(L)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt + "\n")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
