"""Pull one Twins query's predicted histogram of the OTHER twin's outcome from every model, for plotting.
python extract_query.py DATA_ROOT OUT.npz [--d 10] [--block 0] [--query 0]

The histogram is the one Bodik's task scores: 2D models, the learned joint conditioned on the known twin's outcome
(rows/columns with the known twin's outcome >= 1/2, or < 1/2); 1D models, the other arm's marginal. Raw outcome units.
Saved per model: edges, bin masses, and (UWYK 1D) the two tail masses beyond the first and last edge.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bodik_cf as C                                            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("out")
ap.add_argument("--d", type=int, default=10); ap.add_argument("--block", type=int, default=0)
ap.add_argument("--query", type=int, default=0)
a = ap.parse_args()

tz = np.load(os.path.join(a.data, "twins", f"d{a.d}", "truths", f"r{a.block:03d}.npz"))
q = a.query
t = int(np.rint(tz["t_test"][q])); y0, y1 = float(tz["y0"][q]), float(tz["y1"][q])
yf, ycf = (y1, y0) if t == 1 else (y0, y1)
out = dict(t=t, y_known=yf, y_other=ycf)
for label, mdir, tag in C.MODELS:
    f = os.path.join(a.data, "dumps", mdir, "twins", f"d{a.d}", f"CMECH_n5_r{a.block:03d}.npz")
    if not os.path.exists(f):
        print("missing", f); continue
    with np.load(f, allow_pickle=True) as z:
        g = lambda b: (np.asarray(z[f"{b}_{tag}" if tag else b], np.float64) if (f"{b}_{tag}" if tag else b) in z.files else None)
        joint, p0, p1 = g("p_joint_scaled"), g("p_y0_scaled"), g("p_y1_scaled")
        es = np.asarray(z["edges"], np.float64).reshape(-1)
        s0, m0 = C._map(z, 0); s1, m1 = C._map(z, 1)
        e0, e1 = es * s0 + m0, es * s1 + m1
        tails = None
        if joint is not None and joint.ndim == 3:
            J = joint[q] / joint[q].sum()
            ef, ecf = (e1, e0) if t == 1 else (e0, e1)
            af = C._above(ef)
            wf = af if yf >= 0.5 else 1 - af
            w = (wf[:, None] * J.T).sum(0) if t == 1 else (wf[:, None] * J).sum(0)   # over the known twin's bins
            e = ecf
        else:
            arm_cf = 0 if t == 1 else 1
            P = (p0 if arm_cf == 0 else p1)[q]
            e = e0 if arm_cf == 0 else e1
            if P.size == es.size + 1:                           # UWYK: [left tail, K bars, right tail]
                tails = np.array([P[0], P[-1]]); w = P[1:-1]
            else:
                w = P
        out[f"{label}|edges"] = e; out[f"{label}|w"] = w / (w.sum() + (0 if tails is None else tails.sum()))
        if tails is not None:
            out[f"{label}|tails"] = tails / (w.sum() + tails.sum())
    print("ok", label)
np.savez(a.out, **out)
print("wrote", a.out, "| known twin", "heavier" if t == 1 else "lighter", "outcome", yf, "| other twin outcome", ycf)
