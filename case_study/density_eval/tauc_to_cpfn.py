"""Convert tauC prediction dumps into the cpfn density schema.

WHY. Two dump schemas mean two scorers and two costs:

  cpfn1d/cpfn2d   store a FINISHED discrete density (p_y0_scaled,
                  p_y1_scaled, p_joint_scaled on a shared bin grid). Getting
                  p(tau) is an anti-diagonal sum -- O(J^2) array ops.

  tauC models     store the raw ingredients (joint_logits, uwyk_pred0/1,
                  bar_edges, tail scales). Getting p(tau) means numerically
                  integrating  p(tau) = int f0(y) f1(y+tau) dy  with n_y0
                  quadrature points, for every tau and every query -- millions
                  of ops, ~0.05-0.14 s/query, which is the whole runtime of
                  dsweep_density_report.

But the discrete masses are already in the tauC dumps:

    joint      Joint2D.from_pred(...).p_mat     (J, J) bin probabilities
    1D heads   UWYK1D.rebin(edges2d)            bar probabilities, CDF-
                                                interpolated onto the grid

Both are one softmax / one np.interp per query. So converting to the cpfn
schema costs almost nothing and lets ONE scorer
(UWYK_Fig3_4/cate_density_metrics.py) produce coverage / length / IS_0.05 for
every model on identical footing -- which is what a comparison across models
requires anyway.

TAILS ARE DROPPED. The tauC objects carry exponential tail regions outside
the bin grid; the cpfn schema has none. Each query is renormalised over the
interior, and the interior mass is recorded per realization as
`interior_mass_mean` so the size of the approximation stays visible rather
than implicit. (The repo already sanctions this: `uwyk_matched` is defined as
UWYK rebinned onto edges2d.)

    python tauc_to_cpfn.py $RES $OUT --ctx 1000
    python tauc_to_cpfn.py $RES $OUT --ctx 1000 --shifts shift0 --ds 5

Writes  <out>/<shift>/d<K>/ctx<N>/<model>/<case>/<CASE>_r###.npz
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from density_common import Joint2D, UWYK1D, DoPFN1D          # noqa: E402
from density_tauc import DIR_METHOD                          # noqa: E402

SHIFTS = ["shift0", "shift+2", "shift-2"]
DS = [2, 3, 5, 10, 20, 30, 40, 50]
CASES = ["Observed_Confounder", "Backdoor_Criterion", "Observed_Mediator",
         "Observed_Mediator_and_Confounder", "Unobserved_Confounder",
         "Frontdoor_Criterion"]
MODELS = list(DIR_METHOD)

# One tauC run writes BOTH heads into the same npz: joint_logits from the
# graph2d checkpoint and uwyk_pred0/1 from the UWYK checkpoint, each
# conditioned on that run's adjacency. DIR_METHOD names only ONE method per
# directory, so converting by it alone discards the joint head from the
# uwyk_v3a / uwyk_noanc dumps -- which is exactly where graph2d's ancestor
# variants live. They need no new GPU work; they are already on disk.
#
#   source dir  -> [(method, output dir name), ...]
EMIT = {
    "uwyk":         [("uwyk_native", "uwyk"),
                     ("joint",       "graph2d")],
    "uwyk_v3a":     [("uwyk_native", "uwyk_v3a"),
                     ("joint",       "graph2d_v3a")],
    "uwyk_noanc":   [("uwyk_native", "uwyk_noanc"),
                     ("joint",       "graph2d_noanc")],
    # graph2d/ runs the same config as uwyk/ (MODEL_FAMILY=uwyk,
    # ANC_VARIANT=full), so its joint duplicates uwyk's. Kept so the tree is
    # complete if uwyk/ is ever absent.
    "graph2d":      [("joint",       "graph2d")],
    "dopfn_native": [("dopfn_native", "dopfn_native")],
    "dopfn_bb":     [("dopfn_joint",  "dopfn_bb")],
}


def _g(z, k, default=None):
    return np.asarray(z[k]) if k in z.files else default


def convert_one(path, method):
    """-> dict of cpfn-schema arrays for one realization npz, or None."""
    with np.load(path, allow_pickle=True) as z:
        if "tau_grid" not in z.files:
            return None
        y_scale = float(np.asarray(z["y_scale"]).reshape(-1)[0])
        y_shift = float(np.asarray(z["y_shift"]).reshape(-1)[0])
        true_cate = np.asarray(z["true_cate"], dtype=np.float64).reshape(-1)

        if method in ("joint", "dopfn_joint"):
            key = "joint_logits" if method == "joint" else "dopfn_joint_logits"
            ekey = "edges2d" if method == "joint" else "dopfn_edges2d"
            if key not in z.files:
                return None
            logits = np.asarray(z[key], dtype=np.float64)
            edges = np.asarray(z[ekey], dtype=np.float64).reshape(-1)
            J = int(np.asarray(z["J" if method == "joint" else "dopfn_J"]
                               ).reshape(-1)[0])
            joints, inner = [], []
            for q in range(logits.shape[0]):
                jt = Joint2D.from_pred(logits[q], J, edges)
                p = jt.p_mat
                joints.append(p / max(p.sum(), 1e-300))
                inner.append(float(jt.w[0]))     # region 0 = interior
            return dict(
                edges=edges.astype(np.float32),
                p_joint_scaled=np.asarray(joints, dtype=np.float32),
                y_shift=np.float32(y_shift), y_scale=np.float32(y_scale),
                true_cate_per_query=true_cate.astype(np.float32),
                interior_mass_mean=np.float32(np.mean(inner)),
            )

        # 1D bar-distribution heads, rebinned onto the joint grid.
        if method in ("uwyk_native", "uwyk_matched"):
            p0a, p1a = _g(z, "uwyk_pred0"), _g(z, "uwyk_pred1")
            if p0a is None or p1a is None:
                return None
            edges = np.asarray(z["edges2d"], dtype=np.float64).reshape(-1)
            be = np.asarray(z["bar_edges"], dtype=np.float64).reshape(-1)
            bw = np.asarray(z["bar_widths"], dtype=np.float64).reshape(-1)
            sL = float(np.asarray(z["base_sL"]).reshape(-1)[0])
            sR = float(np.asarray(z["base_sR"]).reshape(-1)[0])
            mk = lambda p: UWYK1D.from_pred(p, be, bw, sL, sR)
        elif method == "dopfn_native":
            p0a, p1a = _g(z, "dopfn_pred0"), _g(z, "dopfn_pred1")
            if p0a is None:
                return None
            b0 = np.asarray(z["dopfn_borders0_raw"], dtype=np.float64).reshape(-1)
            # DoPFN1D.from_pred returns edges on the SCALED axis
            #     edges = (borders[1:-1] - y_shift) / y_scale
            # so the rebin target must be scaled too. Rebinning onto the RAW
            # borders put the density on the wrong axis and the scorer's units
            # check caught it (sd_ratio 21.6 on CATE, 64.1 on ATE).
            #
            # DoPFN's borders are also NON-uniform, while the cpfn schema's
            # tau_atoms/_bin_width assume a uniform grid. Prefer the joint's
            # grid when the dump carries it -- uniform, scaled, and the same
            # grid dopfn_joint uses, which makes the two dopfn rows directly
            # comparable instead of living on 100 vs 10 bins.
            if "dopfn_edges2d" in z.files:
                edges = np.asarray(z["dopfn_edges2d"],
                                   dtype=np.float64).reshape(-1)
            else:
                _e = (b0[1:-1] - y_shift) / y_scale
                edges = np.linspace(float(_e[0]), float(_e[-1]), len(_e))
            mk = lambda p: DoPFN1D.from_pred(p, b0, y_shift=y_shift,
                                             y_scale=y_scale)
        else:
            return None

        out0, out1, inner = [], [], []
        for q in range(np.asarray(p0a).shape[0]):
            f0 = mk(np.asarray(p0a[q], dtype=np.float64))
            f1 = mk(np.asarray(p1a[q], dtype=np.float64))
            r0, r1 = f0.rebin(edges), f1.rebin(edges)
            a = np.exp(r0.log_pBars); b = np.exp(r1.log_pBars)
            inner.append(0.5 * (a.sum() + b.sum()))
            out0.append(a / max(a.sum(), 1e-300))
            out1.append(b / max(b.sum(), 1e-300))
        return dict(
            edges=edges.astype(np.float32),
            p_y0_scaled=np.asarray(out0, dtype=np.float32),
            p_y1_scaled=np.asarray(out1, dtype=np.float32),
            y_shift=np.float32(y_shift), y_scale=np.float32(y_scale),
            true_cate_per_query=true_cate.astype(np.float32),
            interior_mass_mean=np.float32(np.mean(inner)),
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--ctx", type=int, default=1000)
    ap.add_argument("--shifts", nargs="*", default=SHIFTS)
    ap.add_argument("--ds", nargs="*", type=int, default=DS)
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--cases", nargs="*", default=CASES)
    a = ap.parse_args()

    n_ok = n_skip = 0
    masses = []
    for s in a.shifts:
        for d in a.ds:
            for m in a.models:
                emits = EMIT.get(m)
                if emits is None:
                    method = DIR_METHOD.get(m)
                    if method is None:
                        print(f"[skip] no mapping for {m}"); continue
                    emits = [(method, m)]
                for method, outname in emits:
                    wrote = 0
                    for c in a.cases:
                        src = os.path.join(a.root, s, f"d{d}", f"ctx{a.ctx}",
                                           m, c, "predictions")
                        dst = os.path.join(a.out, s, f"d{d}", f"ctx{a.ctx}",
                                           outname, c)
                        files = sorted(glob.glob(
                            os.path.join(src, f"{c}_r*.npz")))
                        if not files:
                            continue
                        os.makedirs(dst, exist_ok=True)
                        for f in files:
                            got = convert_one(f, method)
                            if got is None:
                                n_skip += 1
                                continue
                            masses.append(float(got["interior_mass_mean"]))
                            np.savez(os.path.join(dst, os.path.basename(f)),
                                     **got)
                            n_ok += 1
                            wrote += 1
                    print(f"  {s} d{d} {m:14s} [{method:12s}] -> "
                          f"{outname:14s} {wrote}", flush=True)
    print(f"\nconverted {n_ok}, skipped {n_skip}")
    if masses:
        mm = np.asarray(masses)
        print(f"interior mass (tails dropped): mean={mm.mean():.5f} "
              f"min={mm.min():.5f}  -- each query renormalised over the grid")


if __name__ == "__main__":
    main()
