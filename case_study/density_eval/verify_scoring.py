"""Self-contained correctness check for a cpfn1d/cpfn2d density dump.

Uses ONLY the model's own numbers -- no cross-model comparison, no assumption
about which model should be wider. The invariant is arithmetic:

    mean of the tau density  ==  the model's own point CATE

Both come from the same histograms. E[Y_a] in RAW units is
(p_a @ centers)*scale_a + shift_a, so the point CATE is

    CATE = (p1 @ c)*s1 + m1  -  (p0 @ c)*s0 + m0

and the tau density must integrate to exactly that. If the de-standardisation
is wrong, the mean moves and the check fails -- regardless of how wide the
density is or how it compares to any other model.

It also scores each dump BOTH ways for per-arm inputs:
    v2  per-arm: each arm on its own RAW grid (correct)
    v1  shared grid with arm0's scale applied to both (the old bug)
so the size of the error is shown, not asserted.

    python case_study/density_eval/verify_scoring.py <density npz> [...]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from density_cpfn import cpfn1d_tau_density_raw, raw_edges, tau_support_raw  # noqa
from run_density_scm import p_tau_atoms, densify                            # noqa

_TRAPZ = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
_ARMS = ('arm0_shift', 'arm0_scale', 'arm1_shift', 'arm1_scale')


def check(path, n_show=3, n_grid=8001):
    z = np.load(path, allow_pickle=True)
    keys = set(z.files)
    edges = np.asarray(z['edges'], np.float64).reshape(-1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    per_arm = all(k in keys for k in _ARMS)
    joint = 'p_joint_scaled' in keys
    kind = '2D joint' if joint else ('1D per-arm' if per_arm else '1D pooled')
    print(f"\n{os.path.basename(path)}   [{kind}]")

    if joint:
        pj = np.asarray(z['p_joint_scaled'], np.float64)
        p0, p1 = pj.sum(axis=2), pj.sum(axis=1)
    else:
        p0 = np.asarray(z['p_y0_scaled'], np.float64)
        p1 = np.asarray(z['p_y1_scaled'], np.float64)
    p0 = p0 / p0.sum(1, keepdims=True)
    p1 = p1 / p1.sum(1, keepdims=True)

    if per_arm:
        m0 = float(np.asarray(z['arm0_shift']).reshape(-1)[0])
        s0 = float(np.asarray(z['arm0_scale']).reshape(-1)[0])
        m1 = float(np.asarray(z['arm1_shift']).reshape(-1)[0])
        s1 = float(np.asarray(z['arm1_scale']).reshape(-1)[0])
        print(f"  arm0 (shift={m0:+.4f} scale={s0:.4f})   "
              f"arm1 (shift={m1:+.4f} scale={s1:.4f})")
        if abs(s1 - s0) < 1e-9 and abs(m1 - m0) < 1e-9:
            print("  NOTE arms share scaling -> per-arm and pooled agree here; "
                  "this dump cannot distinguish the two code paths.")
    else:
        ys = float(np.asarray(z['y_scale']).reshape(-1)[0]) if 'y_scale' in keys else 1.0
        m0 = m1 = 0.0
        s0 = s1 = ys
        print(f"  shared y_scale={ys:.4f} (shift cancels in a difference)")

    # the invariant: point CATE straight from the histograms
    cate_pt = (p1 @ centers) * s1 + m1 - ((p0 @ centers) * s0 + m0)

    rows = []
    if per_arm:
        e0, e1 = raw_edges(edges, m0, s0), raw_edges(edges, m1, s1)
        lo, hi = tau_support_raw(e0, e1)
        pad = 0.02 * (hi - lo)
        g2 = np.linspace(lo - pad, hi + pad, n_grid)
        # v1 (buggy): shared grid, arm0 scale for both arms
        pa, ts, _ = p_tau_atoms(z)
        g1 = np.linspace(ts[0] * s0 * 1.02, ts[-1] * s0 * 1.02, n_grid)
        d1 = densify(pa, ts * s0, g1)
        for q in range(min(n_show, p0.shape[0])):
            pv2 = cpfn1d_tau_density_raw(p0[q], e0, p1[q], e1, g2)
            mu2 = float(_TRAPZ(pv2 * g2, g2) / _TRAPZ(pv2, g2))
            dq = d1[q] / max(float(_TRAPZ(d1[q], g1)), 1e-300)
            mu1 = float(_TRAPZ(dq * g1, g1))
            rows.append((q, cate_pt[q], mu2, mu1))
        print(f"  {'q':>3s} {'point CATE':>12s} {'v2 per-arm':>12s} {'err':>10s}"
              f" {'v1 shared':>12s} {'err':>10s}")
        for q, c, a, b in rows:
            print(f"  {q:3d} {c:12.5f} {a:12.5f} {a-c:10.2e} {b:12.5f} {b-c:10.2e}")
        e2 = max(abs(a - c) for _, c, a, _ in rows)
        e1_ = max(abs(b - c) for _, c, _, b in rows)
        print(f"  max |err|:  v2 = {e2:.2e}   v1 = {e1_:.2e}")
        print("  VERDICT: " + ("v2 matches the model's own CATE; v1 does not."
                               if e2 < 1e-6 <= e1_ else
                               "v2 matches." if e2 < 1e-6 else
                               "*** v2 does NOT match -- scoring is wrong ***"))
        return e2 < 1e-6
    # pooled / joint: single grid
    pa, ts, ys = p_tau_atoms(z)
    tr = ts * ys
    g = np.linspace(tr[0] * 1.02, tr[-1] * 1.02, n_grid)
    d = densify(pa, tr, g)
    errs = []
    print(f"  {'q':>3s} {'point CATE':>12s} {'density mean':>14s} {'err':>10s}")
    for q in range(min(n_show, p0.shape[0])):
        dq = d[q] / max(float(_TRAPZ(d[q], g)), 1e-300)
        mu = float(_TRAPZ(dq * g, g))
        errs.append(abs(mu - cate_pt[q]))
        print(f"  {q:3d} {cate_pt[q]:12.5f} {mu:14.5f} {mu-cate_pt[q]:10.2e}")
    print(f"  max |err| = {max(errs):.2e}   VERDICT: "
          + ("OK" if max(errs) < 1e-4 else "*** MISMATCH ***"))
    return max(errs) < 1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--n-show", type=int, default=3)
    a = ap.parse_args()
    files = []
    for p in a.paths:
        files.extend(sorted(glob.glob(os.path.join(p, "*.npz")))
                     if os.path.isdir(p) else [p])
    files = [f for f in files if 'summary' not in os.path.basename(f)]
    ok = all(check(f, a.n_show) for f in files[:10])
    print("\nALL OK" if ok else "\nFAILURES PRESENT")


if __name__ == "__main__":
    main()
