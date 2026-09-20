"""Coverage aggregated the right way: one number per realization, then mean +- std.

THE PROBLEM WITH THE POOLED MEAN. cate_density_metrics flattens every
realization's per-query scores into one list and takes the mean, so a
case-study cell is one average over 100 x 100 = 10,000 Bernoullis and its
reported SE is std/sqrt(10000). That treats queries as independent draws. They
are not: all queries in a realization share one context and one SCM, so they
are clustered. The point estimate is fine; the error bar is roughly 10x too
small, which makes small differences between models look significant when they
are not.

WHAT THIS DOES INSTEAD. For each realization, compute coverage (and length, IS,
CRPS) over its own queries -- one number per realization -- then report the
mean and the standard deviation ACROSS realizations. The spread is then the
real realization-to-realization variability, and the SE is std/sqrt(n_real).

It also prints the spread itself, which the pooled mean destroys: a method at
0.95 mean coverage made of realizations at 0.7 and 1.0 is not the same as one
where every realization sits at 0.95, and only this view distinguishes them.

Reads the same dumps through the same loader, so nothing is recomputed.

Usage:
    # RealCause / case studies (selector is --dataset)
    python benchmarks/coverage_by_realization.py --root $SCRATCH/rc_dens_uni \\
        --dataset IHDP
    # ComplexMech
    python benchmarks/coverage_by_realization.py --root $SCRATCH/cmech_dens \\
        --context 1000 --nodes 5 --subset total
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "UWYK_Fig3_4"))

from cate_density_metrics import (METHODS, score_file, _resolve_dir,          # noqa: E402
                                  _configure_tau_smoother)


def _valid_set(path):
    """{realization index} judged usable, pooled over every cell in the manifest.

    Realization indices are unique per cell, and a scoring run targets one cell,
    so pooling the indices is safe and avoids having to reconstruct which cell a
    dump came from.
    """
    if not path:
        return None
    import json
    man = json.load(open(path))
    keep = set()
    for rows in man.get("cells", {}).values():
        for r, v in rows.items():
            if v.get("valid"):
                keep.add(int(r))
    print(f"[validity] {path}: keeping {len(keep)} realization indices "
          f"(criteria {man.get('criteria')})", flush=True)
    return keep

_KEYS = ("cover", "length", "is05", "crps")


def cells(a):
    """[(label, tag, [files])] for the requested selector.

    --root takes several roots so the case studies can be pooled over their
    (shift x d) cells: each cell is a separate scorer root, and a per-case
    number wants every cell's realizations in one sample. Realizations from
    different cells are different SCMs, which is fine here -- coverage is being
    averaged over realizations either way, and that is exactly the population
    the case-study tables describe.
    """
    out = []
    roots = a.root if isinstance(a.root, list) else [a.root]
    for label, subdir, tag in METHODS:
        if a.methods and label not in a.methods:
            continue
        files = []
        for root in roots:
            if a.dataset:
                d = _resolve_dir(root, subdir, a.dataset)
                if d:
                    files += sorted(glob.glob(os.path.join(d, "*.npz")))
            else:
                subs = (["nonzero", "zero"] if a.subset == "total" else [a.subset])
                for sub in subs:
                    d = _resolve_dir(os.path.join(root, f"N{a.context}"), subdir,
                                     f"CMECH_n{a.nodes}_{sub}")
                    if d:
                        files += sorted(glob.glob(os.path.join(d, "*.npz")))
        files = [f for f in files if os.path.basename(f) != "summary.npz"]
        keep = getattr(a, "_keep", None)
        if keep is not None:
            def _r(p):
                m = re.search(r"r(\d+)", os.path.basename(p))
                return int(m.group(1)) if m else -1
            files = [f for f in files if _r(f) in keep]
        if a.max_real:
            files = files[: a.max_real]
        out.append((label, tag, files))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, nargs="+",
                    help="one or more scorer roots; several are pooled (use a "
                         "glob for the case studies' shift x d cells)")
    ap.add_argument("--dataset", default=None,
                    help="RealCause dataset or case-study case name")
    ap.add_argument("--context", type=int, default=1000)
    ap.add_argument("--nodes", type=int, default=5)
    ap.add_argument("--subset", default="total",
                    choices=["nonzero", "zero", "total"])
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--max-real", type=int, default=None)
    ap.add_argument("--valid-manifest", default=None,
                    help="cmech_validity.py JSON; realizations marked invalid "
                         "are skipped. Saturated ComplexMech realizations have "
                         "no estimable effect, so including them measures the "
                         "DGP's pathology rather than any model.")
    ap.add_argument("--tag", default=None,
                    help="density key suffix; defaults to each METHODS entry's own")
    # Variant T. The MALC runs only ever persisted aggregated tables, so a
    # per-realization MALC number cannot be re-derived from disk -- the fits
    # have to be redone. Off by default; enabling it costs real time (~0.065 s
    # per query at B=100, ~1.2 s at B=1000, divided by --malc-workers).
    # Coupling passthroughs to score_file. Without --joint-coupling the
    # forced-independent ablation could only be run under the POOLED mean, so the
    # 2D heads' RealCause indep columns would use a different aggregation from
    # every other coverage number in the same table.
    ap.add_argument("--coupling", default="indep",
                    choices=["indep", "comonotonic"],
                    help="how 1D heads turn two marginals into p(tau).")
    ap.add_argument("--joint-coupling", default="learned",
                    choices=["learned", "indep", "comonotonic"],
                    help="2D heads: 'learned' uses the anti-diagonal projection; "
                         "'indep' rebuilds tau from the joint's own marginals "
                         "under rho = 0. RealCause-only ablation.")
    # Per-realization arrays, not just their summary. Two reasons:
    #   1. POOLING. The case-study number is the pooling of shifts 0/+2/-2, and
    #      averaging three per-shift means is not the pooled per-realization mean
    #      unless every cell has the same realization count. Concatenating the
    #      arrays at report time is exact however the work was split.
    #   2. PARTIAL RESULTS. Each cell writes its own file as it finishes, so a job
    #      dying midway costs that cell and nothing else.
    ap.add_argument("--dump-per-real", default=None,
                    help="write per-realization arrays to this .npz "
                         "(keys <method>__<cover|length|is05|crps>)")
    ap.add_argument("--tau-smoother", choices=["none", "malc"], default="none")
    ap.add_argument("--malc-B", type=int, default=1000)
    ap.add_argument("--malc-K", type=int, default=1)
    ap.add_argument("--malc-seed", type=int, default=20180621)
    ap.add_argument("--n-tau", type=int, default=4001)
    ap.add_argument("--malc-workers", type=int, default=1)
    a = ap.parse_args()

    a._keep = _valid_set(a.valid_manifest)

    if a.tau_smoother == "malc":
        from tau_smoother import SmootherConfig
        cfg = SmootherConfig(enabled=True, B=a.malc_B, K=a.malc_K,
                             seed=a.malc_seed, n_tau=a.n_tau,
                             n_workers=a.malc_workers)
        _configure_tau_smoother(cfg)
        print(f"[tau-smoother] MALC-1D ACTIVE  {cfg}", flush=True)
    else:
        _configure_tau_smoother(None)

    sel = a.dataset or f"CMECH_n{a.nodes}_{a.subset} (N={a.context})"
    print(f"roots={len(a.root)}"
          + (f"  ({a.root[0]} ...)" if len(a.root) > 1 else f"  {a.root[0]}")
          + f"\nselector={sel}\n"
          f"variant={'T (MALC B=%d K=%d)' % (a.malc_B, a.malc_K) if a.tau_smoother == 'malc' else 'raw'}\n")
    print(f"{'method':17s} {'n_real':>6s} | "
          f"{'coverage':>8s} {'sd':>7s} {'se':>7s} | "
          f"{'length':>10s} {'sd':>9s} | {'IS_0.05':>10s} | {'CRPS':>9s}")
    print("-" * 104)

    _acc: dict = {}
    for label, tag, files in cells(a):
        per = {k: [] for k in _KEYS}
        for f in files:
            got = score_file(f, a.tag if a.tag is not None else tag,
                             a.coupling, a.joint_coupling)
            if not got:
                continue
            for k in _KEYS:
                v = [g[k] for g in got if g.get(k) is not None]
                if v:
                    per[k].append(float(np.mean(v)))
        n = len(per["cover"])
        if not n:
            print(f"{label:17s} {'—':>6s} | (no dumps)")
            continue
        m = {k: np.asarray(per[k]) for k in _KEYS}
        if a.dump_per_real:
            _acc.update({f"{label}__{k}": m[k] for k in _KEYS})
        sd = lambda v: float(v.std(ddof=1)) if v.size > 1 else float("nan")
        print(f"{label:17s} {n:6d} | "
              f"{m['cover'].mean():8.4f} {sd(m['cover']):7.4f} "
              f"{sd(m['cover'])/np.sqrt(n):7.4f} | "
              f"{m['length'].mean():10.4f} {sd(m['length']):9.4f} | "
              f"{m['is05'].mean():10.4f} | {m['crps'].mean():9.4f}")

    if a.dump_per_real:
        _save(a.dump_per_real, _acc)


def _save(path, acc):
    """Atomic: write a temp file and rename, so a killed job never leaves a
    half-written .npz that a collector would read as complete."""
    if not acc:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    np.savez(tmp, **acc)
    os.replace(tmp if tmp.endswith(".npz") else tmp + ".npz", path)
    print(f"[per-real] wrote {path} ({len(acc)} arrays)", flush=True)


if __name__ == "__main__":
    main()
