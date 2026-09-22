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


def score(dumps_model, subdir, tag, data_cell, n, mode="raw"):
    """-> dict with PEHE and BOTH ATE errors over the query union per realization.

    Both metrics come from the same per-realization tau/truth, so L1 and relative
    are guaranteed consistent rather than produced by two separate passes. The
    relative form floors the denominator at 0.1, matching eval_dopfn_bb_raw, so a
    near-zero true ATE cannot turn a small absolute error into an enormous ratio.
    """
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
    pehe, ate_l1, ate_rel = [], [], []
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
        ate_l1.append(d)
        ate_rel.append(d / max(abs(float(tru.mean())), 0.1))
    if not pehe:
        return None
    p = np.asarray(pehe)
    sem = lambda v: (float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1
                     else float("nan"))
    # PEHE pools as an RMS across realizations because it is itself an RMSE;
    # both ATE errors pool as plain means.
    return dict(n=len(p), pehe=float(np.sqrt(np.mean(p ** 2))), pehe_se=sem(p),
                l1=float(np.mean(ate_l1)), l1_se=sem(np.asarray(ate_l1)),
                rel=float(np.mean(ate_rel)), rel_se=sem(np.asarray(ate_rel)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--nodes", type=int, nargs="+", default=[5, 10, 20, 30, 40, 50])
    ap.add_argument("--regime", default="path_TY")
    ap.add_argument("--hide", type=float, default=0.0)
    ap.add_argument("--mode", default="raw", choices=["raw", "em"])

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
                r = score(md, subdir, tag, cell, n, a.mode)
                if r:
                    r["model"] = label
                    rows.append(r)
        if not rows:
            continue
        L += ["", f"## ComplexMech total — n={n}   "
                  f"(union of {len(nz)} nonzero + {len(ze)} zero subset entries "
                  f"over {len(set(nz) | set(ze))} realizations)", "",
              "| model | realizations | PEHE | L1_ATE | eps_ATE (relative) |",
              "|---|---|---|---|---|"]
        for r in sorted(rows, key=lambda d: d["pehe"]):
            L.append(f"| {r['model']} | {r['n']} | "
                     f"{r['pehe']:.4f} ± {r['pehe_se']:.4f} | "
                     f"{r['l1']:.4f} ± {r['l1_se']:.4f} | "
                     f"{r['rel']:.4f} ± {r['rel_se']:.4f} |")
    L += ["", "Queries are grouped by SOURCE realization, not by position in the",
          "cell dir: the nonzero and zero subsets enumerate DIFFERENT realization",
          "sets, so point_raw_em's positional pairing truncated to min(len) and",
          "joined unrelated realizations. Every query is counted exactly once here.",
          "PEHE pools as an RMS across realizations; both ATE errors as means.",
          "",
          "L1_ATE is |mean(tau_hat) - mean(tau_true)|. eps_ATE divides that by",
          "max(|mean(tau_true)|, 0.1) -- the 0.1 floor matches eval_dopfn_bb_raw and",
          "stops a near-zero true ATE from inflating a small absolute error. On",
          "ComplexMech many realizations have a true ATE near zero, so the relative",
          "column is dominated by that floor for them; read L1 as primary here and",
          "eps_ATE only for comparability with the RealCause tables."]
    txt = "\n".join(L)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt + "\n")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
