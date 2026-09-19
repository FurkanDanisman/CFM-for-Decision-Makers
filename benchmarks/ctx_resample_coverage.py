"""Coverage under context resampling: fixed target, repeated data.

This is the only construction in the study where coverage means what a
statistician expects. The target tau(x_q) is FIXED -- one SCM, one query point --
and the 100 realizations differ only in their context. So

    coverage(x_q) = (1/100) sum_r 1{ tau(x_q) in [l_rq, u_rq] }

is the classical quantity: fix theta, resample the data, count. It asks each
model's interval to behave as a CONFIDENCE interval for a fixed tau, which is
the right question here because the DGP shares arm noise (rho = 1) and therefore
tau has no conditional randomness for a predictive interval to cover.

What the numbers mean. A model whose interval carries aleatoric arm noise that
tau does not have will OVER-cover, with width far exceeding its estimation
error. A model that captured the rho = 1 coupling emits estimation uncertainty
alone and should land near 0.95. Per-query coverage is reported as well as the
mean, because 10 targets at 100 draws gives a binomial SE of ~2.2pp per query --
enough to separate 0.95 from 1.00, not enough for finer claims.

Reads the density dumps through cate_density_metrics, so every model's schema,
per-arm scaling and rebinning are handled by code already in use.

Usage:
    python benchmarks/ctx_resample_coverage.py \\
        --root $SCRATCH/ctxresample_dumps/shift0/d5/ctx1000 \\
        --case Observed_Confounder
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

from cate_density_metrics import METHODS, _query_pmfs, interval_95, _resolve_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="<dumps>/shift<S>/d<D>/ctx<N>")
    ap.add_argument("--case", default="Observed_Confounder")
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--n-query", type=int, default=10)
    a = ap.parse_args()

    todo = [m for m in METHODS if a.methods is None or m[0] in a.methods]
    print(f"root={a.root}\ncase={a.case}\n")

    for label, subdir, tag in todo:
        d = _resolve_dir(a.root, subdir, a.case)
        files = sorted(glob.glob(os.path.join(d, "*.npz"))) if d else []
        files = [f for f in files if os.path.basename(f) != "summary.npz"]
        if not files:
            print(f"{label:16s} (no dumps)")
            continue

        # cov[q] accumulates hits for query q across realizations; wid[q] widths.
        cov, wid, tgt, nr = None, None, None, 0
        for f in files:
            got = _query_pmfs(f, tag)
            if got is None:
                continue
            atoms, pmfs, y_true = got
            nq = min(a.n_query, pmfs.shape[0], y_true.size)
            if cov is None:
                cov, wid = np.zeros(nq), np.zeros(nq)
                tgt = np.asarray(y_true[:nq], dtype=np.float64)
            for q in range(nq):
                lo, hi = interval_95(atoms, pmfs[q])
                cov[q] += float(lo <= y_true[q] <= hi)
                wid[q] += float(hi - lo)
            nr += 1
        if not nr:
            print(f"{label:16s} (no scorable dumps)")
            continue
        cov, wid = cov / nr, wid / nr

        # binomial SE per query at n = nr
        se = np.sqrt(np.clip(cov * (1 - cov), 0, None) / nr)
        print(f"{label:16s} n_realizations={nr}")
        print(f"{'':16s} mean coverage = {cov.mean():.3f}   "
              f"mean length = {wid.mean():.4f}")
        print(f"{'':16s} {'q':>2s} {'tau_true':>10s} {'coverage':>9s} "
              f"{'+-SE':>6s} {'length':>10s}")
        for q in range(len(cov)):
            print(f"{'':16s} {q:2d} {tgt[q]:10.5f} {cov[q]:9.2f} "
                  f"{se[q]:6.3f} {wid[q]:10.4f}")
        print()


if __name__ == "__main__":
    main()
