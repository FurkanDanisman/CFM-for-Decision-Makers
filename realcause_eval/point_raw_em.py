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


def run_cell(cell_dir, tag, mode, max_real=None):
    files = [f for f in sorted(glob.glob(os.path.join(cell_dir, "*.npz")))
             if os.path.basename(f) != "summary.npz"]
    if not files:
        files = [f for f in sorted(glob.glob(os.path.join(cell_dir, "*", "*.npz")))
                 if os.path.basename(f) != "summary.npz"]
    if max_real:
        files = files[:max_real]

    pehe, eps_ate, n_ok = [], [], 0
    for f in files:
        try:
            got = cate_for_file(f, tag, mode)
        except Exception:
            got = None
        if got is None:
            continue
        tau, truth = got
        if tau.size == 0 or not np.isfinite(tau).all():
            continue
        pehe.append(float(np.sqrt(np.mean((tau - truth) ** 2))))
        eps_ate.append(float(abs(tau.mean() - truth.mean())))
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
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--methods", nargs="+", default=None,
                    help="subset of the cate_density_metrics METHODS names")
    ap.add_argument("--modes", nargs="+", default=["raw", "em"], choices=["raw", "em"])
    ap.add_argument("--max-real", type=int, default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()

    todo = [m for m in METHODS if args.methods is None or m[0] in args.methods]
    lines = [f"### Point estimates — {args.dataset} (no MALC)", "",
             "PEHE and eps_ATE recomputed from the density dumps under two mean",
             "functionals. `raw` is the plug-in mean of the discretised law; `em`",
             "is the deconvolution mean (`_em_mean_1d`). Mean +- SE over realizations.",
             ""]
    head = "| method | n | " + " | ".join(
        f"PEHE ({m}) | eps_ATE ({m})" for m in args.modes) + " |"
    lines += [head, "|" + "---|" * (2 + 2 * len(args.modes))]

    for name, subdir, tag in todo:
        cells = {}
        for mode in args.modes:
            d = _resolve_dir(args.root, subdir, args.dataset)
            cells[mode] = run_cell(d, tag, mode, args.max_real) if d else None
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

    out = "\n".join(lines)
    print("\n" + out)
    if args.out_md:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_md)), exist_ok=True)
        with open(args.out_md, "w") as fh:
            fh.write(out + "\n")
        print(f"\nwrote {args.out_md}", flush=True)


if __name__ == "__main__":
    main()
