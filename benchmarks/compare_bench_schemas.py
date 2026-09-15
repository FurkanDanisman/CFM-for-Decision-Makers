"""Are the three benchmarks really being scored the same way?

Prints, per (benchmark, model), the density keys present, the bin count, the
y_scale, whether `edges` are uniform, and therefore WHICH branch of
cate_density_metrics._load_arrays will fire. If a model takes a different
branch in different benchmarks, its numbers are not comparable across them.

    python compare_bench_schemas.py \
        --cs    $SCRATCH/cs_dvar_dens/shift0/d5/ctx1000 \
        --cmech $SCRATCH/cmech_1d2d/N1000 \
        --rc    $SCRATCH/rc_dens_uni
"""
import argparse, glob, os
import numpy as np

DENS = ("p_joint_scaled", "p_y0_scaled")


def describe(path):
    with np.load(path, allow_pickle=True) as z:
        keys = set(z.files)
        dens = [k for k in ("p_joint_scaled", "p_y0_scaled") if k in keys]
        dens += sorted(k for k in keys
                       if k.startswith(("p_joint_scaled_", "p_y0_scaled_")))
        if not dens:
            return "NO DENSITY", None, None, None, None
        arr = np.asarray(z[dens[0]])
        J = arr.shape[-1]
        ys = float(np.asarray(z["y_scale"]).reshape(-1)[0]) if "y_scale" in keys else None
        if "edges" in keys:
            e = np.asarray(z["edges"], dtype=np.float64).reshape(-1)
            w = np.diff(e)
            uni = bool(np.allclose(w, w.mean(), rtol=1e-3)) if w.size else None
            span = f"[{e[0]:.3g},{e[-1]:.3g}]"
        else:
            uni, span = None, "no edges"
        branch = ("joint" if dens[0].startswith("p_joint") else
                  ("rebin-nonuniform" if uni is False else "uniform-1d"))
        return dens[0], J, ys, f"{span} uniform={uni}", branch


ap = argparse.ArgumentParser()
ap.add_argument("--cs"); ap.add_argument("--cmech"); ap.add_argument("--rc")
a = ap.parse_args()

TREES = []
if a.cs:    TREES.append(("case-study", a.cs, "*/*.npz"))
if a.cmech: TREES.append(("cmech", a.cmech, "*/*.npz"))
if a.rc:    TREES.append(("realcause", a.rc, "*/*.npz"))

for label, root, pat in TREES:
    print(f"\n=== {label}: {root} ===")
    if not os.path.isdir(root):
        print("  (missing)"); continue
    for m in sorted(os.listdir(root)):
        d = os.path.join(root, m)
        if not os.path.isdir(d):
            continue
        fs = sorted(glob.glob(os.path.join(d, pat)))
        fs = [f for f in fs if os.path.basename(f) != "summary.npz"]
        if not fs:
            print(f"  {m:16s} (no npz)"); continue
        try:
            k, J, ys, sp, br = describe(fs[0])
        except Exception as e:
            print(f"  {m:16s} ERROR {e}"); continue
        print(f"  {m:16s} {k:24s} J={str(J):<6s} y_scale={ys!s:<10.10s} "
              f"{sp:28s} -> {br}")
print()
