#!/usr/bin/env python
"""Per-arm predictive spread for ONE query of ONE observational dataset.

For each model, read its dumped density for a single (realization, query) and
report sd(Y0) and sd(Y1) in RAW outcome units, plus the implied sd(tau) under the
head's own coupling and under independence. The gap between those two is what the
joint representation is buying on this query:

    Var(tau) = s0^2 + s1^2 - 2 rho s0 s1

so sd_tau_indep is the rho = 0 value, and sd_tau_head is what the head actually
predicts. rho_implied backs out the correlation the head is using.

Marginals come from whatever the head dumped: a 2D head's joint summed over one
axis (exactly what a 1D head would have predicted for that arm), or the per-arm
pmfs directly. Both are mapped back to raw units before any moment is taken --
the two arms can sit on DIFFERENT affine maps (arm0_shift/scale vs
arm1_shift/scale), so taking moments in scaled space and rescaling once would be
wrong for exactly the heads this comparison is about.

    python benchmarks/arm_sd_one_query.py --root $SCRATCH/cs_fixedq_dumps/shift0/d0/ctx1000 \
        --dataset Observed_Confounder --realization 0 --query 0
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "UWYK_Fig3_4"))

from cate_density_metrics import METHODS, _resolve_dir, _bin_width  # noqa: E402


def _files_in(cell_dir):
    """*.npz minus summary.npz, sorted. Local rather than imported from
    point_raw_em: that module pulls in the EM machinery, which this does not
    need and which is slow to import."""
    import glob
    return [f for f in sorted(glob.glob(os.path.join(cell_dir, '*.npz')))
            if os.path.basename(f) != 'summary.npz']


def _centers_scaled(z, J):
    """Bin centers on the SCALED grid: from `edges` when dumped, else [-1,1]."""
    if "edges" in z.files:
        e = np.asarray(z["edges"], dtype=np.float64).reshape(-1)
        if e.size == J + 1:
            return 0.5 * (e[:-1] + e[1:])
        if e.size == J:
            return e
    bw = _bin_width(z, J)
    return (np.arange(J) - (J - 1) / 2.0) * bw


def _affine(z, arm):
    """(shift, scale) for one arm, falling back to the shared y_shift/y_scale."""
    for k_m, k_s in ((f"arm{arm}_shift", f"arm{arm}_scale"), ("y_shift", "y_scale")):
        if k_m in z.files and k_s in z.files:
            return (float(np.asarray(z[k_m]).reshape(-1)[0]),
                    float(np.asarray(z[k_s]).reshape(-1)[0]))
    return 0.0, float(np.asarray(z["y_scale"]).reshape(-1)[0]) if "y_scale" in z.files else 1.0


# Names the harnesses use for the dumped point estimate. Probed rather than
# assumed: mean(Y1) - mean(Y0) under the model's own density SHOULD equal it, and
# a mismatch means the density and the point estimate disagree -- which is exactly
# what the density_scale_r2 gate exists to catch.
_CATE_KEYS = ("cate_pred", "tau_pred", "cate_hat", "pred_cate", "cate_raw",
              "cate", "tau_hat")


def _dumped_cate(z, q):
    for k in _CATE_KEYS:
        if k in z.files:
            v = np.asarray(z[k], dtype=np.float64).ravel()
            if q < v.size:
                return float(v[q]), k
    return float("nan"), None


def _moments(p, centers):
    p = np.asarray(p, dtype=np.float64).ravel()
    s = p.sum()
    if not np.isfinite(s) or s <= 0:
        return float("nan"), float("nan")
    p = p / s
    m = float(np.sum(p * centers))
    v = float(np.sum(p * (centers - m) ** 2))
    return m, float(np.sqrt(max(v, 0.0)))


def arms_for(path, q):
    """-> (m0, s0, m1, s1, sd_tau_head) in raw units, or None."""
    with np.load(path, allow_pickle=True) as z:
        keys = set(z.files)
        if "p_joint_scaled" in keys:
            J_ = np.asarray(z["p_joint_scaled"], dtype=np.float64)
            if J_.ndim != 3 or q >= J_.shape[0]:
                return None
            joint = J_[q]
            J = joint.shape[-1]
            p0 = joint.sum(axis=1)               # marginal of Y0
            p1 = joint.sum(axis=0)               # marginal of Y1
            c = _centers_scaled(z, J)
            m0s, s0s = _moments(p0, c)
            m1s, s1s = _moments(p1, c)
            # sd(tau) under the head's OWN coupling, from the joint itself
            pj = joint / max(joint.sum(), 1e-300)
            d = c[None, :] - c[:, None]          # y1 - y0 on the scaled grid
            mt = float((pj * d).sum())
            st = float(np.sqrt(max((pj * (d - mt) ** 2).sum(), 0.0)))
            a0, b0 = _affine(z, 0)
            cp, ck = _dumped_cate(z, q)
            return (m0s * b0 + a0, s0s * b0, m1s * b0 + a0, s1s * b0,
                    st * b0, cp, ck)
        if "p_y0_scaled" in keys and "p_y1_scaled" in keys:
            P0 = np.asarray(z["p_y0_scaled"], dtype=np.float64)
            P1 = np.asarray(z["p_y1_scaled"], dtype=np.float64)
            if P0.ndim != 2 or q >= P0.shape[0]:
                return None
            J = P0.shape[-1]
            c = _centers_scaled(z, J)
            a0, b0 = _affine(z, 0)
            a1, b1 = _affine(z, 1)
            m0s, s0s = _moments(P0[q], c)
            m1s, s1s = _moments(P1[q], c)
            # Comonotonic needs both arms on ONE grid; only meaningful when the
            # two affine maps agree, which is the case the dumps assert for the
            # pooled scaling. Otherwise leave it blank rather than mis-scale.
            cp, ck = _dumped_cate(z, q)
            # A 1D head dumps no joint, so it has no coupling of its own.
            return (m0s * b0 + a0, s0s * b0, m1s * b1 + a1, s1s * b1,
                    float("nan"), cp, ck)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", nargs="+", required=True,
                    help="one or more dirs holding <harness>/<dataset>. Several "
                         "because the 13 checkpoints reuse 6 harness names and so "
                         "must live in separate roots; the row is named after the "
                         "root when it disambiguates.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--realization", type=int, default=0)
    ap.add_argument("--query", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    L = [f"## Per-arm predictive spread — {a.dataset}, "
         f"realization {a.realization}, query {a.query}", "",
         "| model | CATE estimate | sd(Y0) | sd(Y1) | v(x) | 95% CI "
         "| v(x) rho=1 | 95% CI rho=1 |", "|" + "---|" * 8]
    seen, rows_out = 0, []

    def _rootname(root):
        """First ancestor of `root` that is not a layout component."""
        _LAYOUT = ("cs", "rc")
        parts = [q for q in os.path.normpath(root).split(os.sep) if q]
        for q in reversed(parts):
            if (q.startswith(("shift", "ctx")) or q in _LAYOUT
                    or re.fullmatch(r"d\d+", q)):
                continue
            return q
        return os.path.basename(root)

    for root in a.root:
        # Resolve first, then name. A root holding SEVERAL harnesses is the base
        # sweep, where the harness label already IS the model (dopfn_native,
        # cpfn1d, ...). A root holding ONE is a per-checkpoint root, where the
        # harness label is shared across checkpoints and only the directory
        # identifies the model -- naming by root there, by label here.
        found = []
        for label, subdir, tag in METHODS:
            d = _resolve_dir(root, subdir, a.dataset)
            if d and os.path.isdir(d):
                found.append((label, subdir, d))
        by_harness = {sd for _, sd, _ in found}
        use_root = len(by_harness) == 1
        rootname = _rootname(root)
        for label, subdir, d in found:
            fs = _files_in(d)
            if a.realization >= len(fs):
                continue
            got = arms_for(fs[a.realization], a.query)
            if got is None:
                continue
            if use_root:
                suf = next((x for x in ("-noanc", "-v3ab", "-v3a", "-v3b")
                            if label.endswith(x)), "")
                name = rootname + suf
            else:
                name = label
            rows_out.append((name, *got))
    mismatch = []
    for name, m0, s0, m1, s1, st, cp, ck in rows_out:
        si = float(np.sqrt(s0 ** 2 + s1 ** 2))          # rho = 0
        rho = ((s0 ** 2 + s1 ** 2 - st ** 2) / (2 * s0 * s1)
               if np.isfinite(st) and s0 > 0 and s1 > 0 else float("nan"))
        f = lambda v: "—" if not np.isfinite(v) else f"{v:.4f}"
        # v(x) = Var(Y^do(1) - Y^do(0) | X = x) = s1^2 + s0^2 - 2 rho s1 s0.
        # A 2D head supplies rho through its joint; a 1D head has none, so rho = 0
        # and v collapses to s0^2 + s1^2. The `from` column says which was used,
        # because the two are not the same estimand-under-assumption.
        # v(x) as each head can actually form it: the joint's own coupling for a
        # 2D head, rho = 0 for a 1D head (it has no dependence to use).
        vx = st ** 2 if np.isfinite(st) else si ** 2
        v1_ = (s1 - s0) ** 2                    # every model forced to rho = 1
        # 6 significant digits, not 6 decimals: v|rho=1 is a difference of nearly
        # equal sds, so a fixed 6-dp format prints 9e-08 as 0.000000 and the
        # column cannot be re-derived from what is shown.
        g = lambda v: "—" if not np.isfinite(v) else f"{v:.6g}"
        # The intervals, so nobody has to recompute them from rounded output.
        ci = lambda m, v: ("—" if not np.isfinite(v)
                           else f"[{m - 1.96 * v ** 0.5:+.4f}, "
                                f"{m + 1.96 * v ** 0.5:+.4f}]")
        tau = m1 - m0
        if np.isfinite(cp) and abs(cp - tau) > 1e-3 * max(1.0, abs(cp)):
            mismatch.append((name, tau, cp, ck))
        L.append(f"| {name} | {f(tau)} | {f(s0)} | {f(s1)} | "
                 f"{g(vx)} | {ci(tau, vx)} | {g(v1_)} | {ci(tau, v1_)} |")
        seen += 1
    L += ["",
          "CATE estimate = mean(Y1) - mean(Y0) under the model's own predictive",
          "density. sd(Y0), sd(Y1) are that density's per-arm spreads, in raw",
          "outcome units.",
          "",
          "v(x) = Var(Y^do(1) - Y^do(0) | X = x) = s0^2 + s1^2 - 2 rho s0 s1,",
          "evaluated the way each head can actually form it: a 2D head uses its",
          "joint's own coupling, a 1D head has none so rho = 0.",
          "v(x) rho=1 forces rho = 1 for EVERY model, giving (s1 - s0)^2 -- the",
          "same marginals under perfect positive dependence. It is a DIFFERENCE",
          "of nearly equal sds, so it is printed to 6 significant digits and",
          "cannot be recomputed from the 4-dp sd columns: 0.2383 - 0.2380 keeps",
          "one significant digit and squaring it keeps none.",
          "",
          "95% CI = CATE estimate +- 1.96 * sqrt(v), given for both v columns.",
          "",
          "On the CASE STUDIES the true v(x) is 0: the generator adds one shared",
          "noise draw to both arms, so Y^do(1) - Y^do(0) = mu_1 - mu_0 exactly.",
          "Every positive v(x) is predicted spread the DGP does not contain."]
    txt = "\n".join(L)
    print(txt)
    if not seen:
        print(f"\nno model dumps found under {a.root}", file=sys.stderr)
        return 1
    for nm, tau, cp, ck in mismatch:
        print(f"WARN {nm}: mean(Y1)-mean(Y0) = {tau:.6f} but the dump's {ck} "
              f"= {cp:.6f} -- the density and the point estimate disagree",
              file=sys.stderr)
    if a.out:
        open(a.out, "w").write(txt + "\n")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
