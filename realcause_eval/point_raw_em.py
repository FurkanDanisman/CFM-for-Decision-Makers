"""PEHE and eps_ATE from the density dumps, under raw-mean vs EM-mean.

No MALC anywhere in this file. The point estimate is a functional of the
predicted arm densities, and the question here is only which functional:

  raw   E[Y_a] = sum_j p_a[j] * centre[j]
        the plug-in mean of the discretised predictive law.

  EM    the deconvolution mean already used inside the eval harnesses
        (_em_mean_1d): treat the binned mass as a Gaussian-blurred view of a
        point and iterate to the location that reproduces it. It is the same
        routine MALC uses in step 1 of its component fit, which is why it is
        available here without new machinery.

Either way tau = E[Y_1] - E[Y_0], and PEHE / eps_ATE follow. For a 2D method
the arm marginals are the joint's row and column sums, so both estimators are
defined for every model in the table.

WHY THIS RUNS ON THE DUMPS. The densities were already written by the eval
harnesses; nothing here needs a GPU or a re-run. It reads the identical NPZs
that cate_density_metrics scores, so the point and interval columns describe
the same estimator on the same realizations.

PER-ARM SCALING. When a dump carries arm0_/arm1_ shift and scale separately,
tau = (s1*m1 + h1) - (s0*m0 + h0) and the offsets do NOT cancel. Using the
pooled formula there drops (h1 - h0) and produces a pure location error --
wide intervals that miss, or a biased PEHE that still looks plausible. Handled
explicitly below.

Usage:
    python realcause_eval/point_raw_em.py \\
        --root /scratch/.../rc_dens_uni --dataset IHDP \\
        --out-md /scratch/.../point_raw_em_IHDP.md
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (os.path.join(_REPO, "UWYK_Fig3_4"), os.path.join(_REPO, "MALC")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from malc_1d import _em_mean_1d                                    # noqa: E402
from cate_density_metrics import METHODS, _resolve_dir, _bin_width  # noqa: E402


def _tagged(z, base, tag):
    """Fetch base_<tag> when a tag is asked for, else base. Never cross-falls.

    Falling back to the untagged key when a tag was requested is how two
    different ancestor modes ended up reported as identical rows once already.
    """
    if tag:
        k = f"{base}_{tag}"
        if k in z.files:
            return np.asarray(z[k], dtype=np.float64)
        return None
    return np.asarray(z[base], dtype=np.float64) if base in z.files else None


def _arm_marginals(z, tag):
    """(p0, p1) as (n_q, J), from explicit marginals or from the joint."""
    p0, p1 = _tagged(z, "p_y0_scaled", tag), _tagged(z, "p_y1_scaled", tag)
    if p0 is not None and p1 is not None:
        return p0, p1
    pj = _tagged(z, "p_joint_scaled", tag)
    if pj is None:
        return None, None
    # Convention fixed by tau_pmf_joint: axis 0 is y0, axis 1 is y1.
    return pj.sum(axis=2), pj.sum(axis=1)


def _scaling(z):
    """((h0, s0), (h1, s1)) -- per-arm when dumped, else the pooled pair."""
    per_arm = all(k in z.files for k in
                  ("arm0_shift", "arm0_scale", "arm1_shift", "arm1_scale"))
    f = lambda k, d: (float(np.asarray(z[k]).reshape(-1)[0]) if k in z.files else d)
    if per_arm:
        return (f("arm0_shift", 0.0), f("arm0_scale", 1.0)), \
               (f("arm1_shift", 0.0), f("arm1_scale", 1.0))
    h, s = f("y_shift", 0.0), f("y_scale", 1.0)
    return (h, s), (h, s)


def _centres(z, J):
    if "bucket_means" in z.files:
        c = np.asarray(z["bucket_means"], dtype=np.float64).reshape(-1)
        if c.size == J:
            return c, None
    if "edges" in z.files:
        e = np.asarray(z["edges"], dtype=np.float64).reshape(-1)
        if e.size == J + 1:
            return 0.5 * (e[:-1] + e[1:]), e
    w = _bin_width(z, J)
    e = (np.arange(J + 1) - J / 2.0) * w
    return 0.5 * (e[:-1] + e[1:]), e


def _arm_means(p, centres, edges, mode):
    """(n_q,) arm means in SCALED units, raw plug-in or EM deconvolution."""
    if mode == "raw":
        return p @ centres
    out = np.empty(p.shape[0])
    width = float(np.mean(np.diff(edges)))
    for q in range(p.shape[0]):
        pq = p[q] / max(p[q].sum(), 1e-300)
        mu0 = float(pq @ centres)
        sigma = float(np.sqrt(max(pq @ (centres - mu0) ** 2, 0.0) + width ** 2 / 12.0))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = width
        out[q] = _em_mean_1d(pq, edges, sigma=sigma, start=mu0)
    return out


def cate_for_file(path, tag, mode):
    """(tau_hat, tau_true) in RAW units for one realization, or None."""
    with np.load(path, allow_pickle=True) as z:
        y_true = _tagged(z, "true_cate_per_query", None)
        if y_true is None:
            return None
        p0, p1 = _arm_marginals(z, tag)
        if p0 is None:
            return None
        J = p0.shape[-1]
        centres, edges = _centres(z, J)
        if edges is None:
            w = float(centres[1] - centres[0])
            edges = np.concatenate([centres - w / 2, [centres[-1] + w / 2]])
        (h0, s0), (h1, s1) = _scaling(z)

        m0 = _arm_means(p0, centres, edges, mode)
        m1 = _arm_means(p1, centres, edges, mode)
        tau = (s1 * m1 + h1) - (s0 * m0 + h0)
        n = min(tau.size, y_true.size)
        return tau[:n], y_true[:n]


def _files_in(cell_dir):
    files = [f for f in sorted(glob.glob(os.path.join(cell_dir, "*.npz")))
             if os.path.basename(f) != "summary.npz"]
    if not files:
        files = [f for f in sorted(glob.glob(os.path.join(cell_dir, "*", "*.npz")))
                 if os.path.basename(f) != "summary.npz"]
    return files


_ATE_METRIC = "l1"          # set from --ate-metric in main()
_ATE_LABEL = "L1_ATE"


def run_cell(cell_dirs, tag, mode, max_real=None):
    """PEHE / eps_ATE over one or more cell dirs treated as ONE query set.

    Several dirs is how ComplexMech's `total` is formed: its `nonzero` and
    `zero` cells split the QUERIES of the same realizations, not the
    realizations themselves. PEHE is a root-mean-square, so the two cannot be
    averaged after the fact -- sqrt(mean(e^2)) over the union is not the mean
    of the two sqrt(mean(e^2)). The queries are concatenated per realization
    first, then scored, which is exact.

    eps_ATE likewise: |mean(tau) - mean(truth)| over the union, since the ATE
    is a mean over all queries of the realization.
    """
    if isinstance(cell_dirs, str):
        cell_dirs = [cell_dirs]
    per_dir = [_files_in(d) for d in cell_dirs]
    per_dir = [fs for fs in per_dir if fs]
    if not per_dir:
        return None
    n_real = min(len(fs) for fs in per_dir)
    if max_real:
        n_real = min(n_real, max_real)

    pehe, eps_ate, n_ok = [], [], 0
    for r in range(n_real):
        taus, truths = [], []
        for fs in per_dir:
            try:
                got = cate_for_file(fs[r], tag, mode)
            except Exception:
                got = None
            if got is None:
                continue
            taus.append(got[0]); truths.append(got[1])
        if not taus:
            continue
        tau = np.concatenate(taus); truth = np.concatenate(truths)
        if tau.size == 0 or not np.isfinite(tau).all():
            continue
        pehe.append(float(np.sqrt(np.mean((tau - truth) ** 2))))
        # L1 vs RELATIVE. This column has been labelled eps_ATE while computing
        # plain |dATE|, whereas eval_dopfn_bb_raw.py divides by
        # max(|ATE_true|, 0.1) -- so numbers from the two scripts were not
        # comparable. RealCause reports the relative eps_ATE; the other two
        # benchmarks report L1. Default follows --ate-metric, and the header says
        # which one was used.
        _d = float(abs(tau.mean() - truth.mean()))
        if _ATE_METRIC == "rel":
            # Floor at 0.1, matching eval_dopfn_bb_raw, so a near-zero true ATE
            # cannot turn a small absolute error into an enormous ratio.
            _d /= max(abs(float(truth.mean())), 0.1)
        eps_ate.append(_d)
        n_ok += 1
    if not n_ok:
        return None
    return dict(n=n_ok,
                pehe=float(np.mean(pehe)), pehe_se=float(np.std(pehe) / np.sqrt(n_ok)),
                eps=float(np.mean(eps_ate)), eps_se=float(np.std(eps_ate) / np.sqrt(n_ok)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="density-dump root (per-method dirs under it)")
    ap.add_argument("--dataset", required=True, nargs="+",
                    help="one or more cell names. Several are treated as ONE "
                         "query set (ComplexMech's `total` = nonzero + zero), "
                         "concatenated per realization before scoring.")
    ap.add_argument("--methods", nargs="+", default=None,
                    help="subset of the cate_density_metrics METHODS names")
    ap.add_argument("--modes", nargs="+", default=["raw", "em"], choices=["raw", "em"])
    ap.add_argument("--ate-metric", default="rel", choices=["rel", "l1"],

                    help="rel = |dATE| / max(|ATE_true|, 0.1), the RealCause convention and what eval_dopfn_bb_raw reports; l1 = plain |dATE|, used for ComplexMech and the case studies.")

    ap.add_argument("--max-real", type=int, default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()
    global _ATE_METRIC, _ATE_LABEL
    _ATE_METRIC = args.ate_metric
    _ATE_LABEL = "eps_ATE" if args.ate_metric == "rel" else "L1_ATE"

    todo = [m for m in METHODS if args.methods is None or m[0] in args.methods]
    lines = [f"### Point estimates — {' + '.join(args.dataset)} (no MALC)", "",
             "PEHE and eps_ATE recomputed from the density dumps under two mean",
             "functionals. `raw` is the plug-in mean of the discretised law; `em`",
             "is the deconvolution mean (`_em_mean_1d`). Mean +- SE over realizations.",
             ""]
    # Stamp the metric into the file. Tables written before --ate-metric existed
    # report plain |dATE| under an "eps_ATE" header, so the header alone cannot
    # tell a correct relative table from a stale absolute one -- and a progress
    # view counting those files reports 100% complete for the wrong quantity.
    stamp = f"<!-- ate_metric={_ATE_METRIC} -->"
    head = "| method | n | " + " | ".join(
        f"PEHE ({m}) | {_ATE_LABEL} ({m})" for m in args.modes) + " |"
    lines += [head, "|" + "---|" * (2 + 2 * len(args.modes))]

    for name, subdir, tag in todo:
        cells = {}
        for mode in args.modes:
            ds = [_resolve_dir(args.root, subdir, n) for n in args.dataset]
            ds = [d for d in ds if d]
            cells[mode] = run_cell(ds, tag, mode, args.max_real) if ds else None
        if not any(cells.values()):
            print(f"[skip] {name}: no scorable dumps", flush=True)
            continue
        n = next(c["n"] for c in cells.values() if c)
        row = [name, str(n)]
        for mode in args.modes:
            c = cells[mode]
            row += ([f"{c['pehe']:.4f} ± {c['pehe_se']:.4f}",
                     f"{c['eps']:.4f} ± {c['eps_se']:.4f}"] if c else ["—", "—"])
        lines.append("| " + " | ".join(row) + " |")
        print(f"[ok] {name}  n={n}", flush=True)

    out = stamp + "\n" + "\n".join(lines)
    print("\n" + out)
    if args.out_md:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_md)), exist_ok=True)
        with open(args.out_md, "w") as fh:
            fh.write(out + "\n")
        print(f"\nwrote {args.out_md}", flush=True)


if __name__ == "__main__":
    main()
