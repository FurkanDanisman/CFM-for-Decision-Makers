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


_FFT_MIN_J = 256


def tau_pmf_indep(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """1D head: assume Y0 ⟂ Y1 | X and convolve the two arm marginals.

    Direct correlation is O(J^2) per query. cpfn1d runs J=1024, so at ~85k
    queries per subset that is ~10^11 operations and the script appears to
    hang. Above _FFT_MIN_J we do the same convolution by FFT in O(J log J);
    below it the direct path is faster and is kept as the reference.
    np.correlate(a, v, 'full') == np.convolve(a, v[::-1], 'full') for real
    input, which is what the transform computes.
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    J = p0.shape[-1]
    if J >= _FFT_MIN_J:
        n = 2 * J - 1
        nfft = 1 << (n - 1).bit_length()
        s = np.fft.irfft(np.fft.rfft(p1, nfft) * np.fft.rfft(p0[::-1], nfft),
                         nfft)[:n]
        s = np.maximum(s, 0.0)          # kill FFT round-off negatives
    else:
        s = _diag_sums_product(p0, p1)
    tot = s.sum()
    return s / tot if tot > 0 else s


def tau_pmf_comonotonic(p0: np.ndarray, p1: np.ndarray, J: int) -> np.ndarray:
    """1D head, opposite coupling: perfectly rank-dependent (comonotonic).

    Independence is not neutral here — it is one extreme. On this benchmark the
    two arms share their exogenous noise, so corr(Y0, Y1) is 0.93-0.99 and the
    true tau is 3.6-11x narrower than a convolution predicts. Scoring 1D heads
    ONLY under independence would therefore measure the coupling assumption
    rather than the head.

    The comonotonic coupling is the other extreme: pair the quantiles, i.e.
    Y1 = F1^-1(F0(Y0)). It gives the NARROWEST tau consistent with the same two
    marginals, so together the two bracket what any 1D head could achieve
    without modelling dependence. Built by the northwest-corner rule on the two
    CDFs.
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    F0, F1 = np.cumsum(p0), np.cumsum(p1)
    out = np.zeros(2 * J - 1, dtype=np.float64)
    i = j = 0
    prev = 0.0
    while i < J and j < J:
        c = min(F0[i], F1[j])
        m = c - prev
        if m > 0:
            out[(j - i) + (J - 1)] += m
        prev = c
        if F0[i] <= F1[j]:
            i += 1
        else:
            j += 1
    tot = out.sum()
    return out / tot if tot > 0 else out


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
    # Predictive mean and sd are carried alongside the scores as a units check.
    # A density living on the wrong scale still produces finite CRPS/coverage,
    # it just looks like a very bad model — so the only way to tell a broken
    # unit conversion from a genuinely poor forecast is to compare the
    # predictive spread against the spread of the truth. cpfn1d hard-asserts
    # this at dump time; the other harnesses do not.
    mu = float(np.sum(t * p))
    sd = float(np.sqrt(max(np.sum((t - mu) ** 2 * p), 0.0)))
    return dict(cover=float(lo <= y <= hi), length=float(hi - lo),
                crps=crps_discrete(t, p, y), wis=wis_discrete(t, p, y),
                pred_mean=mu, pred_sd=sd, bias=mu - y, y_true=y)


# ── per-realization driver ────────────────────────────────────────────────────

def _bin_width(z, J):
    """Scaled bin width: from `edges` when dumped, else the [-1,1] convention."""
    if "edges" in z.files:
        e = np.asarray(z["edges"], dtype=np.float64).reshape(-1)
        if e.size >= 2:
            return float(np.mean(np.diff(e)))
    return 2.0 / J


def _load_arrays(path, tag=None, coupling="indep"):
    """(atoms, pmf generator, y_true, n_q) for one realization npz, or None."""
    with np.load(path, allow_pickle=True) as z:
        # Density keys must match the REQUESTED tag. Falling back to the
        # un-suffixed key when a tag is asked for produced two identical rows
        # for uwyk1d-noanc and uwyk1d-v3a, because uwyk1d used to dump only one
        # mode's density under un-suffixed keys. Silently scoring the wrong
        # mode is worse than reporting nothing, so the fallback is gone for
        # density arrays; shared metadata (edges, y_scale, truth) is never
        # per-mode and still resolves unsuffixed.
        def g(base, strict=False):
            if tag:
                k = f"{base}_{tag}"
                if k in z.files:
                    return np.asarray(z[k], dtype=np.float64)
                if strict:
                    return None
            if base in z.files:
                return np.asarray(z[base], dtype=np.float64)
            return None

        y_true = g("true_cate_per_query")
        if y_true is None:
            return None
        y_true = y_true.reshape(-1)
        y_scale = float(z["y_scale"]) if "y_scale" in z.files else 1.0

        joint = g("p_joint_scaled", strict=True)
        p0 = g("p_y0_scaled", strict=True)
        p1 = g("p_y1_scaled", strict=True)
        if joint is not None and joint.ndim == 3:
            J = joint.shape[-1]
            atoms = tau_atoms(J, _bin_width(z, J)) * y_scale
            pmfs = [tau_pmf_joint(joint[q]) for q in range(joint.shape[0])]
        elif p0 is not None and p1 is not None and p0.ndim == 2:
            J = p0.shape[-1]
            atoms = tau_atoms(J, _bin_width(z, J)) * y_scale
            if coupling == "comonotonic":
                pmfs = [tau_pmf_comonotonic(p0[q], p1[q], J)
                        for q in range(p0.shape[0])]
            else:
                pmfs = [tau_pmf_indep(p0[q], p1[q]) for q in range(p0.shape[0])]
        else:
            return None
    _LAST_J["_J"] = int(len(atoms))
    return atoms, pmfs, y_true, len(pmfs)


def _query_pmfs(path, tag=None, coupling="indep"):
    """(atoms, pmfs (n_q, K), y_true (n_q,)) for one npz, or None."""
    got = _load_arrays(path, tag, coupling)
    if got is None:
        return None
    atoms, pmfs, y_true, _ = got
    pm = np.asarray(pmfs, dtype=np.float64)
    return atoms, pm, y_true[: pm.shape[0]]


def score_file(path, tag=None, coupling="indep"):
    """All per-query scores for one realization npz. None if it has no density."""
    got = _load_arrays(path, tag, coupling)
    if got is None:
        return None
    atoms, pmfs, y_true, _ = got
    out = []
    for q, pmf in enumerate(pmfs):
        if q >= y_true.size:
            break
        if not np.isfinite(pmf).all() or pmf.sum() <= 0:
            continue
        out.append(score_query(atoms, pmf, float(y_true[q])))
    return out or None


_LAST_J: dict = {}

def _bary(atoms, pmfs):
    """Wasserstein barycenter of per-query tau densities -> one ATE predictive.

    density_calc.md section 5: the ATE density is the 1D 2-Wasserstein
    barycenter of the per-query CATE densities. Uses the repo's reference
    implementation so this matches the IHDP/ACIC pipeline.

    IMPORTANT about what this is. The 1D barycenter averages QUANTILE
    functions, so the barycenter of N(mu_i, s) is N(mean(mu_i), s) — the same
    width as a single query, NOT s/sqrt(n). It is a "typical CATE" density, not
    the sampling distribution of the ATE estimator. Scored against the realized
    ATE it will therefore look heavily over-dispersed (near-total coverage, long
    intervals). That is a property of the definition, not a bug.
    """
    import sys as _s
    _mp = os.path.join(_REPO, "MALC", "Optimal_Transport")
    if _mp not in _s.path:
        _s.path.insert(0, _mp)
    from ot_barycenter import wasserstein_barycenter_1d

    atoms = np.asarray(atoms, dtype=np.float64)
    dx = float(atoms[1] - atoms[0])
    dens = np.asarray(pmfs, dtype=np.float64) / dx          # pmf -> density
    bary = wasserstein_barycenter_1d(dens, atoms)
    pmf = np.maximum(bary, 0.0) * dx
    tot = pmf.sum()
    return pmf / tot if tot > 0 else pmf


def ate_rows(root, ctx, nodes, subdir, tag, coupling, subset, data_root,
             max_real=None):
    """One ATE predictive per source realization, with its realized ATE.

    Under `total` the two subsets are two files of the SAME dataset, so their
    queries are pooled before the barycenter and the truth is the ATE over all
    of them. Scoring the halves separately would give the ATE of a query subset,
    which is not the estimand.
    """
    import importlib
    agg = importlib.import_module("aggregate_cmech_methods")
    subs = ["nonzero", "zero"] if subset == "total" else [subset]

    bysrc: dict = {}
    for sub in subs:
        src, _counts = agg.subset_sources(nodes, sub, data_root)
        d = os.path.join(root, f"N{ctx}", subdir, f"CMECH_n{nodes}_{sub}")
        files = sorted(glob.glob(os.path.join(d, "*.npz")))
        if max_real:
            files = files[:max_real]
        for f in files:
            digits = "".join(c for c in os.path.basename(f).rsplit("r", 1)[-1]
                             if c.isdigit())
            if not digits:
                continue
            idx = int(digits)
            got = _query_pmfs(f, tag, coupling)
            if got is None:
                continue
            atoms, pmfs, y = got
            s_id = src[idx] if idx < len(src) else (idx, sub)
            cur = bysrc.setdefault(s_id, dict(atoms=atoms, pmfs=[], y=[]))
            if cur["atoms"].shape != atoms.shape or not np.allclose(cur["atoms"], atoms):
                continue      # differing support between halves: cannot pool
            cur["pmfs"].append(pmfs)
            cur["y"].append(y)

    out = []
    for s_id, c in sorted(bysrc.items(), key=lambda kv: str(kv[0])):
        pm = np.concatenate(c["pmfs"], axis=0)
        yy = np.concatenate(c["y"])
        if pm.shape[0] == 0:
            continue
        out.append((c["atoms"], _bary(c["atoms"], pm), float(yy.mean())))
    return out


METHODS = [
    ("dopfn_native",  "dopfn_native",  None),
    ("dopfn_bb",      "dopfn_bb",      None),
    ("uwyk1d-noanc",  "uwyk1d",        "noanc"),
    ("uwyk1d-v3a",    "uwyk1d",        "v3a"),
    ("graph2d-noanc", "graph2d",       "noanc"),
    ("graph2d-v3a",   "graph2d",       "v3a"),
    # cpfn1d's density run uses STD_MODE=pooled, NOT the per_arm its point
    # estimates use: under per_arm each arm carries its own y_shift/y_scale and
    # the harness refuses to dump, because one affine map cannot un-scale the
    # difference. So this row is pooled-scaled while the cpfn1d row in the
    # PEHE/ATE tables is per_arm. Worth a footnote if both are reported.
    ("cpfn1d",        "cpfn1d",        None),
    ("cpfn2d",        "cpfn2d_pooled", None),
    # cpfn2d_log is deliberately absent: under a log target transform tau is not
    # an affine function of the scaled difference, so the harness refuses to dump
    # a density for it (and is right to). It still appears in the point-estimate
    # tables.
]


def _parse():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--context", type=int, default=1000)
    ap.add_argument("--nodes", type=int, nargs="+", default=[5],
                    help="one or more node counts; each gets its own table")
    ap.add_argument("--subset", default="nonzero",
                    choices=["nonzero", "zero", "total"])
    ap.add_argument("--max-real", type=int, default=None,
                    help="cap realizations per method (quick look)")
    ap.add_argument("--coupling", default="indep",
                    choices=["indep", "comonotonic"],
                    help="how 1D heads turn two marginals into p(tau). "
                         "indep = convolution (widest); comonotonic = quantile "
                         "coupling (narrowest). 2D heads ignore this. On this "
                         "benchmark the arms are near-comonotonic, so reporting "
                         "only `indep` scores the assumption, not the head.")
    ap.add_argument("--methods", nargs="+", default=None,
                    help="subset of method labels to score (default: all). "
                         "Useful to skip dopfn_native, whose bar distribution "
                         "has far more bins than the others and dominates the "
                         "runtime.")
    ap.add_argument("--skip", nargs="+", default=(),
                    help="method labels to exclude.")
    ap.add_argument("--data-root", default=os.environ.get(
        "UWYK_FIG34_DATA", os.path.join(_HERE, "data")),
        help="benchmark data root; needed by --target ate to map each npz back "
             "to its source realization when pooling the two subsets.")
    ap.add_argument("--target", default="cate", choices=["cate", "ate"],
                    help="cate = score each query's tau density against its true "
                         "tau. ate = Wasserstein-barycenter the per-query "
                         "densities into one ATE density per dataset and score "
                         "it against that dataset's realized ATE.")
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def _build_ate(args, todo):
    """One ATE density per dataset (barycenter of its query CATE densities),
    scored against that dataset's realized ATE."""
    rows = []
    for label, subdir, tag in todo:
        print(f"[scoring-ate] {label} ...", end="", flush=True)
        recs = ate_rows(args.root, args.context, args.nodes, subdir, tag,
                        args.coupling, args.subset, args.data_root,
                        args.max_real)
        print(f" {len(recs)} datasets", flush=True)
        if not recs:
            continue
        sc = [score_query(a, p, y) for a, p, y in recs]
        arr = {k: np.array([x[k] for x in sc]) for k in sc[0]}
        n = len(sc)
        sem = lambda v: float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        true_sd = float(arr["y_true"].std())
        rows.append(dict(
            method=label, n_files=n, n_query=n,
            pred_sd=float(arr["pred_sd"].mean()), true_sd=true_sd,
            sd_ratio=float(arr["pred_sd"].mean() / true_sd) if true_sd > 0 else float("nan"),
            bias=float(arr["bias"].mean()),
            coverage95=float(arr["cover"].mean()),
            length=float(arr["length"].mean()), length_sem=sem(arr["length"]),
            crps=float(arr["crps"].mean()), crps_sem=sem(arr["crps"]),
            wis=float(arr["wis"].mean()), wis_sem=sem(arr["wis"]),
        ))
    if not rows:
        raise SystemExit(f"no density dumps under {args.root}/N{args.context}")
    return _render(rows, args)


def build_table(args):
    # `total` = every query. These metrics are per-query means, so unlike PEHE
    # (an RMS needing weighted pooling) the two disjoint subsets simply
    # concatenate.
    subsets = (["nonzero", "zero"] if args.subset == "total" else [args.subset])
    rows = []
    todo = [m for m in METHODS
            if (args.methods is None or m[0] in args.methods)
            and m[0] not in args.skip]

    if args.target == "ate":
        return _build_ate(args, todo)

    for label, subdir, tag in todo:
        acc, n_files = [], 0
        print(f"[scoring] {label} ...", end="", flush=True)
        for sub in subsets:
            d = os.path.join(args.root, f"N{args.context}", subdir,
                             f"CMECH_n{args.nodes}_{sub}")
            files = sorted(glob.glob(os.path.join(d, "*.npz")))
            if args.max_real:
                files = files[: args.max_real]
            for f in files:
                got = score_file(f, tag, args.coupling)
                if got:
                    acc += got
                    n_files += 1
        print(f" {len(acc)} queries from {n_files} file(s)", flush=True)
        if not acc:
            print(f"[skip] {label}: no density dumps for "
                  f"{'/'.join(subsets)} under "
                  f"{os.path.join(args.root, f'N{args.context}', subdir)}")
            continue
        arr = {k: np.array([a[k] for a in acc]) for k in acc[0]}
        true_sd = float(arr["y_true"].std())
        print(f"           tau support atoms: {_LAST_J.get('_J', '?')}", flush=True)
        rows.append(dict(
            method=label, n_files=n_files, n_query=len(acc),
            pred_sd=float(arr["pred_sd"].mean()), true_sd=true_sd,
            sd_ratio=float(arr["pred_sd"].mean() / true_sd) if true_sd > 0 else float("nan"),
            bias=float(arr["bias"].mean()),
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

    return _render(rows, args)


def _render(rows, args):
    hdr = ["method", "n_files", "n_query", "coverage95", "length", "crps", "wis",
           "pred_sd", "true_sd", "sd_ratio", "bias"]
    _what = ("CATE density per query, scored against that query's true tau"
             if args.target == "cate" else
             "ATE density per dataset (Wasserstein barycenter of its per-query "
             "CATE densities), scored against that dataset's realized ATE")
    L = [f"# {args.target.upper()} density calibration — d={args.nodes}, "
         f"N={args.context}, {args.subset}, 1D coupling = {args.coupling}", "",
         f"Scored object: {_what}.", "",
         "2D heads: diagonal projection of the joint. 1D heads: independence",
         "convolution of the two arm marginals. Raw densities, no MALC.", "",
         "coverage95 should be ~0.95; below means over-confident, above means",
         "the intervals are wider than they need to be. Read it WITH length —",
         "a wide interval buys coverage for free. CRPS and WIS are proper, so",
         "lower is better on both and they penalise that trade-off.", "",
         "`n_files` counts npz files scored, not distinct datasets: under",
         "`--subset total` each dataset contributes up to two (its zero and its",
         "non-zero queries), so n_files exceeds the realization count. `n_query`",
         "is the honest total and is exact — the subsets are disjoint and",
         "together complete.", "",
         "`sd_ratio` = mean predictive sd / sd of the true tau. A UNITS CHECK:",
         "a density dumped on the wrong scale still scores finitely, it just",
         "looks like a bad model. Values near 1 are well-calibrated in spread;",
         "a ratio in the tens means the density is on the wrong axis, not that",
         "the model is that much worse. `bias` is mean(E[tau]) - tau_true.", "",
         "| " + " | ".join(hdr) + " |", "|" + "|".join("---" for _ in hdr) + "|"]
    for r in rows:
        L.append("| {method} | {n_files} | {n_query} | {coverage95:.3f} | "
                 "{length:.4f} ± {length_sem:.4f} | {crps:.4f} ± {crps_sem:.4f} | "
                 "{wis:.4f} ± {wis_sem:.4f} | {pred_sd:.4f} | {true_sd:.4f} | "
                 "{sd_ratio:.1f} | {bias:+.4f} |".format(**r))
    md = "\n".join(L) + "\n"
    print(md)
    out = args.out or os.path.join(
        args.root, f"{args.target}_density_d{args.nodes}_N{args.context}"
        f"_{args.subset}_{args.coupling}.md")
    with open(out, "w") as f:
        f.write(md)
    with open(out.replace(".md", ".json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[written] {out}")
    return md


def main():
    args = _parse()
    nodes = args.nodes
    parts = []
    for n in nodes:
        args.nodes = n
        print(f"\n{'='*62}\n  d = {n}\n{'='*62}", flush=True)
        try:
            parts.append(build_table(args))
        except SystemExit as exc:
            print(f"[skip] d={n}: {exc}", flush=True)
    if len(nodes) > 1 and parts:
        out = os.path.join(
            args.root,
            f"{args.target}_density_allnodes_N{args.context}"
            f"_{args.subset}_{args.coupling}.md")
        with open(out, "w") as f:
            f.write("\n\n".join(parts))
        print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
