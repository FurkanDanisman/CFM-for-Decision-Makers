"""How many npz in a sweep cell actually carry a density, vs point-eval only.

progress.py opens only the FIRST npz per (model, case) directory; if point-eval
files sort ahead of density files the cell reads as 0%. This opens every file's
key list (cheap -- npz is a zip, numpy reads the namelist without decompressing)
and reports per (model, case).

    python dens_inventory.py $RES [shift0] [d5] [ctx1000]

Layout walked: <root>/<shift>/d<K>/ctx<N>/<model>/<case>/*.npz
"""
import os, sys, glob
import numpy as np

DENSITY_KEYS = ("p_joint_scaled", "p_y0_scaled")
root  = sys.argv[1]
shift = sys.argv[2] if len(sys.argv) > 2 else "shift0"
d     = sys.argv[3] if len(sys.argv) > 3 else "d5"
ctx   = sys.argv[4] if len(sys.argv) > 4 else "ctx1000"

base = os.path.join(root, shift, d, ctx)
print(f"{base}\n")
print(f"{'model':16s} {'case':34s} {'npz':>6s} {'dens':>6s} {'point':>6s} {'bad':>4s}  sample density keys")
print("-" * 118)
for m in sorted(os.listdir(base)):
    mdir = os.path.join(base, m)
    if not os.path.isdir(mdir):
        continue
    tot = [0, 0, 0, 0]
    for c in sorted(os.listdir(mdir)):
        cell = os.path.join(mdir, c)
        if not os.path.isdir(cell):
            continue
        files = sorted(glob.glob(os.path.join(cell, "*.npz")))
        ndens = npoint = nbad = 0
        sample = ""
        for f in files:
            try:
                with np.load(f, allow_pickle=True) as z:
                    keys = set(z.files)
            except Exception:
                nbad += 1
                continue
            if keys & set(DENSITY_KEYS):
                ndens += 1
                if not sample:
                    sample = ",".join(sorted(
                        k for k in keys
                        if k.startswith(("p_y", "p_joint", "edges", "true_cate"))))[:40]
            else:
                npoint += 1
        has_metrics = os.path.isfile(os.path.join(cell, "metrics.json"))
        flag = " [scored]" if has_metrics else ""
        print(f"{m:16s} {c:34s} {len(files):6d} {ndens:6d} {npoint:6d} {nbad:4d}  {sample}{flag}")
        for i, v in enumerate((len(files), ndens, npoint, nbad)):
            tot[i] += v
    print(f"{m:16s} {'== TOTAL':34s} {tot[0]:6d} {tot[1]:6d} {tot[2]:6d} {tot[3]:4d}")
    print("-" * 118)
