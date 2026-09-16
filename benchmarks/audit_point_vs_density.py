"""Does each model's POINT estimate match the mean of the density it dumped?

A table pairing PEHE with interval scores is only meaningful if both describe
the same estimator. For dopfn_native they do not: its PEHE comes from
predict_cate (raw borders + STALE normalised half-widths, an upstream unit
bug) while its density is scored on raw edge centres -- measured 1.0631 vs
0.2794 PEHE on CMECH_n5. This checks every other model for the same split.

For each npz it computes tau three ways and compares:
    stored     the pehe_raw* recorded by the eval (drives published PEHE)
    density    PEHE of the mean of the dumped density (drives the intervals)

    python audit_point_vs_density.py <cell-dir> [<cell-dir> ...]
      e.g. $SCRATCH/cs_dvar_dens/shift0/d5/ctx1000/*
"""
import glob, os, sys
import numpy as np


def tau_from_density(z, J=None):
    """-> (n_q,) mean tau implied by the dumped density, or None."""
    keys = set(z.files)
    e = np.asarray(z["edges"], dtype=np.float64).reshape(-1) if "edges" in keys else None

    jk = [k for k in keys if k.startswith("p_joint_scaled")]
    if jk:
        j = np.asarray(z[sorted(jk)[0]], dtype=np.float64)
        Jn = j.shape[-1]
        c = (0.5*(e[:-1]+e[1:]) if e is not None and e.size == Jn+1
             else np.linspace(-1, 1, Jn))
        ys = float(np.asarray(z["y_scale"]).reshape(-1)[0]) if "y_scale" in keys else 1.0
        m0 = (j.sum(axis=2) @ c) * ys
        m1 = (j.sum(axis=1) @ c) * ys
        return m1 - m0

    z0 = [k for k in keys if k.startswith("p_y0_scaled")]
    z1 = [k for k in keys if k.startswith("p_y1_scaled")]
    if not z0 or not z1:
        return None
    p0 = np.asarray(z[sorted(z0)[0]], dtype=np.float64)
    p1 = np.asarray(z[sorted(z1)[0]], dtype=np.float64)
    Jn = p0.shape[-1]
    c = (0.5*(e[:-1]+e[1:]) if e is not None and e.size == Jn+1
         else np.linspace(-1, 1, Jn))
    ys = float(np.asarray(z["y_scale"]).reshape(-1)[0]) if "y_scale" in keys else 1.0
    return ((p1 @ c) - (p0 @ c)) * ys


def stored_pehe(z):
    ks = sorted(k for k in z.files if k.startswith("pehe_raw"))
    if not ks:
        return None, None
    k = ks[0]
    return k, float(np.asarray(z[k]).reshape(-1)[0])


for cell in sys.argv[1:]:
    name = os.path.basename(cell.rstrip("/"))
    fs = [f for f in sorted(glob.glob(os.path.join(cell, "*", "*.npz")))
          if os.path.basename(f) != "summary.npz"][:20]
    if not fs:
        fs = [f for f in sorted(glob.glob(os.path.join(cell, "*.npz")))
              if os.path.basename(f) != "summary.npz"][:20]
    if not fs:
        print(f"{name:16s} (no npz)"); continue
    sp, dp, key, n = [], [], None, 0
    for f in fs:
        try:
            with np.load(f, allow_pickle=True) as z:
                if "true_cate_per_query" not in z.files:
                    continue
                tc = np.asarray(z["true_cate_per_query"], dtype=np.float64).ravel()
                td = tau_from_density(z)
                if td is None or td.shape[0] != tc.shape[0]:
                    continue
                k, v = stored_pehe(z)
                key = key or k
                dp.append(float(np.sqrt(np.mean((td - tc) ** 2))))
                if v is not None:
                    sp.append(v)
                n += 1
        except Exception:
            continue
    if not n:
        print(f"{name:16s} (unreadable / no truth)"); continue
    s = f"{np.mean(sp):.4f}" if sp else "   n/a"
    d = np.mean(dp)
    flag = ""
    if sp and np.mean(sp) > 0:
        r = d / np.mean(sp)
        flag = "  <-- MISMATCH" if (r < 0.5 or r > 2.0) else ""
    print(f"{name:16s} n={n:3d}  stored[{key or '-'}]={s}  density={d:.4f}{flag}")
