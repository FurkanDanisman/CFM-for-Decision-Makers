"""Calibration metrics for the predicted CATE density: coverage, length, WIS, CRPS.

Builds p(tau | x) per query from whatever each model emits, then scores that
predictive distribution against the true tau for that query.

    2D models (graph2d, cpfn2d, dopfn_bb)   diagonal projection of the joint
    1D models (cpfn1d, uwyk1d, dopfn_native) independence convolution of the
                                             two arm marginals

Both are the RAW constructions of density_calc.md section 4 — no MALC smoothing.
Diagonal sums reuse density_common._diag_sums / _diag_sums_product so the
offset convention matches the rest of the repo: S[k] = sum_i p[i, i+k], i.e.
k = (y1 bin) - (y0 bin), tau = k * bin_width.

Metrics, per query, against the realized true tau:

    coverage   is true tau inside the central 95% interval of the predictive?
    length     width of that interval
    CRPS       integral (F(x) - 1{x >= tau_true})^2 dx, exact for a step CDF
    WIS        weighted interval score over the standard 11-alpha set

Units: tau is a DIFFERENCE of outcomes, so the affine y_raw = y_scaled*y_scale
+ y_shift contributes only its scale — tau_raw = tau_scaled * y_scale, the shift
cancels. true_cate_per_query is already in raw units.

Requires the eval runs to have been made with DENSITY_DUMP=1.

Usage
-----
    python UWYK_Fig3_4/cate_density_metrics.py --root $SCRATCH/cmech_dens \
        --context 1000 --nodes 5
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (os.path.join(_REPO, "benchmarks"),
           os.path.join(_REPO, "benchmarks", "eval_graph2d"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from density_common import _diag_sums, _diag_sums_product  # noqa: E402

# Standard WIS interval set (Bracher et al. 2021, as used by the COVID hubs).
WIS_ALPHAS = (0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


# ── predictive distribution over tau ──────────────────────────────────────────

def tau_atoms(J: int, bin_w: float) -> np.ndarray:
    """Support of the difference distribution: k * bin_w for k = -(J-1)..(J-1)."""
    return np.arange(-(J - 1), J, dtype=np.float64) * bin_w


def tau_pmf_joint(joint: np.ndarray) -> np.ndarray:
    """2D head: diagonal projection. joint[i, j] = p(y0 bin i, y1 bin j)."""
    s = _diag_sums(np.asarray(joint, dtype=np.float64))
    tot = s.sum()
    return s / tot if tot > 0 else s


def tau_pmf_indep(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """1D head: assume Y0 ⟂ Y1 | X and convolve the two arm marginals."""
    s = _diag_sums_product(np.asarray(p0, dtype=np.float64),
                           np.asarray(p1, dtype=np.float64))
    tot = s.sum()
    return s / tot if tot > 0 else s


# ── scoring rules ─────────────────────────────────────────────────────────────

def _quantile(t: np.ndarray, F: np.ndarray, q: float) -> float:
    """Smallest atom whose CDF reaches q (the usual discrete quantile)."""
    return float(t[min(int(np.searchsorted(F, q, side="left")), len(t) - 1)])


def crps_discrete(t: np.ndarray, p: np.ndarray, y: float) -> float:
    """Exact CRPS for a discrete predictive: int (F(x) - 1{x>=y})^2 dx.

    F is a step function, so the integral is a finite sum over the gaps between
    atoms, with the one segment containing y split at y and the tails handled
    separately.
    """
    F = np.cumsum(p)
    F[-1] = 1.0
    dt = np.diff(t)
    Fi = F[:-1]
    ind = (t[:-1] >= y).astype(np.float64)
    total = float(np.sum((Fi - ind) ** 2 * dt))

    k = int(np.searchsorted(t, y, side="right")) - 1
    if 0 <= k <= len(t) - 2:
        a, b = t[k], t[k + 1]
        total -= (Fi[k] - ind[k]) ** 2 * (b - a)
        total += (Fi[k] - 0.0) ** 2 * (y - a) + (Fi[k] - 1.0) ** 2 * (b - y)
    if y < t[0]:
        total += float(t[0] - y)      # F=0, indicator=1 on [y, t0)
    if y > t[-1]:
        total += float(y - t[-1])     # F=1, indicator=0 on [t_last, y)
    return total


def wis_discrete(t, p, y, alphas=WIS_ALPHAS) -> float:
    """Weighted interval score: mean of the median AE and the alpha-weighted
    interval scores, normalised by (K + 1/2)."""
    F = np.cumsum(p)
    F[-1] = 1.0
    total = 0.5 * abs(y - _quantile(t, F, 0.5))
    for a in alphas:
        lo = _quantile(t, F, a / 2.0)
        hi = _quantile(t, F, 1.0 - a / 2.0)
        isc = (hi - lo) + (2.0 / a) * max(0.0, lo - y) + (2.0 / a) * max(0.0, y - hi)
        total += (a / 2.0) * isc
    return float(total / (len(alphas) + 0.5))


def interval_95(t, p):
    F = np.cumsum(p)
    F[-1] = 1.0
    return _quantile(t, F, 0.025), _quantile(t, F, 0.975)


def score_query(t, p, y):
    lo, hi = interval_95(t, p)
    return dict(cover=float(lo <= y <= hi), length=float(hi - lo),
                crps=crps_discrete(t, p, y), wis=wis_discrete(t, p, y))


# ── per-realization driver ────────────────────────────────────────────────────

def _bin_width(z, J):
    """Scaled bin width: from `edges` when dumped, else the [-1,1] convention."""
    if "edges" in z.files:
        e = np.asarray(z["edges"], dtype=np.float64).reshape(-1)
        if e.size >= 2:
            return float(np.mean(np.diff(e)))
    return 2.0 / J


def score_file(path, tag=None):
    """All per-query scores for one realization npz. None if it has no density."""
    with np.load(path, allow_pickle=True) as z:
        def g(base):
            for k in ((f"{base}_{tag}",) if tag else ()) + (base,):
                if k in z.files:
                    return np.asarray(z[k], dtype=np.float64)
            return None

        y_true = g("true_cate_per_query")
        if y_true is None:
            return None
        y_true = y_true.reshape(-1)
        y_scale = float(z["y_scale"]) if "y_scale" in z.files else 1.0

        joint, p0, p1 = g("p_joint_scaled"), g("p_y0_scaled"), g("p_y1_scaled")
        if joint is not None and joint.ndim == 3:
            J = joint.shape[-1]
            bw = _bin_width(z, J)
            atoms = tau_atoms(J, bw) * y_scale
            pmfs = (tau_pmf_joint(joint[q]) for q in range(joint.shape[0]))
            n_q = joint.shape[0]
        elif p0 is not None and p1 is not None and p0.ndim == 2:
            J = p0.shape[-1]
            bw = _bin_width(z, J)
            atoms = tau_atoms(J, bw) * y_scale
            pmfs = (tau_pmf_indep(p0[q], p1[q]) for q in range(p0.shape[0]))
            n_q = p0.shape[0]
        else:
            return None

    out = []
    for q, pmf in enumerate(pmfs):
        if q >= y_true.size:
            break
        if not np.isfinite(pmf).all() or pmf.sum() <= 0:
            continue
        out.append(score_query(atoms, pmf, float(y_true[q])))
    return out or None


METHODS = [
    ("dopfn_native",  "dopfn_native",  None),
    ("dopfn_bb",      "dopfn_bb",      None),
    ("uwyk1d-noanc",  "uwyk1d",        "noanc"),
    ("uwyk1d-v3a",    "uwyk1d",        "v3a"),
    ("graph2d-noanc", "graph2d",       "noanc"),
    ("graph2d-v3a",   "graph2d",       "v3a"),
    ("cpfn1d",        "cpfn1d",        None),
    ("cpfn2d",        "cpfn2d_pooled", None),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--context", type=int, default=1000)
    ap.add_argument("--nodes", type=int, default=5)
    ap.add_argument("--subset", default="nonzero",
                    choices=["nonzero", "zero", "total"])
    ap.add_argument("--max-real", type=int, default=None,
                    help="cap realizations per method (quick look)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ds = f"CMECH_n{args.nodes}_{args.subset}"
    rows = []
    for label, subdir, tag in METHODS:
        d = os.path.join(args.root, f"N{args.context}", subdir, ds)
        files = sorted(glob.glob(os.path.join(d, "*.npz")))
        if args.max_real:
            files = files[: args.max_real]
        acc, n_files = [], 0
        for f in files:
            got = score_file(f, tag)
            if got:
                acc += got
                n_files += 1
        if not acc:
            print(f"[skip] {label}: no density dumps under {d}")
            continue
        arr = {k: np.array([a[k] for a in acc]) for k in acc[0]}
        rows.append(dict(
            method=label, n_real=n_files, n_query=len(acc),
            coverage95=float(arr["cover"].mean()),
            length=float(arr["length"].mean()),
            length_sem=float(arr["length"].std(ddof=1) / np.sqrt(len(acc))),
            crps=float(arr["crps"].mean()),
            crps_sem=float(arr["crps"].std(ddof=1) / np.sqrt(len(acc))),
            wis=float(arr["wis"].mean()),
            wis_sem=float(arr["wis"].std(ddof=1) / np.sqrt(len(acc))),
        ))

    if not rows:
        raise SystemExit(
            f"no density dumps under {args.root}/N{args.context}. "
            "Re-run the evals with DENSITY_DUMP=1.")

    hdr = ["method", "n_real", "n_query", "coverage95", "length", "crps", "wis"]
    L = [f"# CATE density calibration — d={args.nodes}, N={args.context}, "
         f"{args.subset}", "",
         "2D heads: diagonal projection of the joint. 1D heads: independence",
         "convolution of the two arm marginals. Raw densities, no MALC.", "",
         "coverage95 should be ~0.95; below means over-confident, above means",
         "the intervals are wider than they need to be. Read it WITH length —",
         "a wide interval buys coverage for free. CRPS and WIS are proper, so",
         "lower is better on both and they penalise that trade-off.", "",
         "| " + " | ".join(hdr) + " |", "|" + "|".join("---" for _ in hdr) + "|"]
    for r in rows:
        L.append("| {method} | {n_real} | {n_query} | {coverage95:.3f} | "
                 "{length:.4f} ± {length_sem:.4f} | {crps:.4f} ± {crps_sem:.4f} | "
                 "{wis:.4f} ± {wis_sem:.4f} |".format(**r))
    md = "\n".join(L) + "\n"
    print(md)
    out = args.out or os.path.join(
        args.root, f"cate_density_d{args.nodes}_N{args.context}_{args.subset}.md")
    with open(out, "w") as f:
        f.write(md)
    with open(out.replace(".md", ".json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[written] {out}")


if __name__ == "__main__":
    main()
