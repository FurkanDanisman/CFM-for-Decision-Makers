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

from cate_density_metrics import METHODS          # noqa: E402
from point_raw_em import cate_for_file, _files_in               # noqa: E402


def display(model_dir, label):
    """Row name: the MODEL directory, keeping the ancestry suffix that
    distinguishes several rows from one root. Without this every cpfn1d_* root
    printed as bare 'cpfn1d' and the four dopfn roots all printed 'dopfn_native',
    so the rows could not be told apart. Matches final_table.display_name.
    """
    for suf in ("-noanc", "-v3ab", "-v3a", "-v3b"):
        if label.endswith(suf):
            return model_dir + suf
    return model_dir


_SS_CACHE = {}
def subset_sources(data_cell):
    if data_cell in _SS_CACHE: return _SS_CACHE[data_cell]
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
    _SS_CACHE[data_cell] = (nz, ze)
    return nz, ze


def score(dumps_model, subdir, tag, data_cell, n, mode="raw", ctx=1000):
    """-> dict with PEHE and BOTH ATE errors over the query union per realization.

    Both metrics come from the same per-realization tau/truth, so L1 and relative
    are guaranteed consistent rather than produced by two separate passes. The
    relative form floors the denominator at 0.1, matching eval_dopfn_bb_raw, so a
    near-zero true ATE cannot turn a small absolute error into an enormous ratio.
    """
    nz_src, ze_src = subset_sources(data_cell)
    by_src: dict[int, list[str]] = {}
    for sub, srcs in (("nonzero", nz_src), ("zero", ze_src)):
        # Recursive: the cell dir's depth under the model root varies by harness,
        # so an assumed "N1000/<subdir>/<cell>" path found nothing and every model
        # scored zero. Match <subdir>/<cell> at ANY depth, as the progress tracker
        # does.
        # Scoped to N<ctx>: an unscoped recursive glob matches EVERY context
        # directory at once, so a multi-context dump tree would pool N50 with
        # N1000 and report a number belonging to neither.
        dirs = [d for d in glob.glob(
                    os.path.join(dumps_model, f"N{ctx}", "**", subdir,
                                 f"CMECH_n{n}_{sub}"),
                    recursive=True) if os.path.isdir(d)]
        if not dirs:
            continue                       # this model simply is not that method
        fs = []
        for d in dirs:
            fs += _files_in(d)
        fs = sorted(fs, key=lambda q: os.path.basename(q))
        if not fs:
            continue
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
    # MEAN is primary: point_raw_em reports mean-of-per-realization PEHE, so every
    # previously published number uses that convention and switching to RMS here
    # would make the tables silently incomparable. RMS is kept alongside because it
    # is the correct pooling if you treat all queries as one set -- it is always >=
    # the mean, and the gap tells you how skewed the per-realization PEHEs are.
    return dict(n=len(p), pehe=float(np.mean(p)), pehe_se=sem(p),
                pehe_rms=float(np.sqrt(np.mean(p ** 2))),
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

    ap.add_argument("--contexts", type=int, nargs="+", default=[1000],
                    help="context sizes to report, one block per (n, context)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--csv", default=None,
                    help="also write tidy long-form rows for plotting")
    a = ap.parse_args()

    L, csv_rows = [], []
    for ctx in a.contexts:
      for n in a.nodes:
        cell = os.path.join(a.data, "complexmech", f"{n}node", a.regime,
                            f"hide_{a.hide}")
        nz, ze = subset_sources(cell)
        print(f"[progress] N={ctx} n={n}: {len(nz)} nonzero + {len(ze)} zero "
              f"realizations, scoring models ...", flush=True)
        rows = []
        for md in sorted(d for d in glob.glob(f"{a.dumps}/*") if os.path.isdir(d)):
            for label, subdir, tag in METHODS:
                r = score(md, subdir, tag, cell, n, a.mode, ctx)
                if r:
                    r["model"] = display(os.path.basename(md), label)
                    r["nodes"], r["context"] = n, ctx
                    rows.append(r); csv_rows.append(r)
        if not rows:
            continue
        L += ["", f"## ComplexMech total — n={n}, N={ctx}   "
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
          "All three are means +- SEM over realizations, matching the convention\npoint_raw_em uses for every other table (mode=raw throughout; EM is never\ncomputed).",
          "",
          "L1_ATE is |mean(tau_hat) - mean(tau_true)|. eps_ATE divides that by",
          "max(|mean(tau_true)|, 0.1) -- the 0.1 floor matches eval_dopfn_bb_raw and",
          "stops a near-zero true ATE from inflating a small absolute error. On",
          "ComplexMech many realizations have a true ATE near zero, so the relative",
          "column is dominated by that floor for them; read L1 as primary here and",
          "eps_ATE only for comparability with the RealCause tables."]
    if a.csv:
        import csv as _csv
        os.makedirs(os.path.dirname(os.path.abspath(a.csv)) or ".", exist_ok=True)
        with open(a.csv, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["nodes", "context", "model", "n_real", "pehe", "pehe_sem",
                        "l1_ate", "l1_sem", "eps_ate", "eps_sem"])
            for r in csv_rows:
                w.writerow([r["nodes"], r["context"], r["model"], r["n"],
                            f"{r['pehe']:.6f}", f"{r['pehe_se']:.6f}",
                            f"{r['l1']:.6f}", f"{r['l1_se']:.6f}",
                            f"{r['rel']:.6f}", f"{r['rel_se']:.6f}"])
        print(f"wrote {a.csv} ({len(csv_rows)} rows)", file=sys.stderr)

    txt = "\n".join(L)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt + "\n")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
