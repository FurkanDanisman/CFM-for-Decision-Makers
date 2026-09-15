"""How many npz in a cell actually carry a density, vs point-eval only.

progress.py opens only the FIRST npz per directory; if point-eval files sort
ahead of density files the cell reads as 0%. This opens every file's key list
(cheap -- npz is a zip, numpy reads the namelist without decompressing).
"""
import os, sys, glob
import numpy as np

DENSITY_KEYS = ("p_joint_scaled", "p_y0_scaled")
root = sys.argv[1]
shift = sys.argv[2] if len(sys.argv) > 2 else "shift0"
d     = sys.argv[3] if len(sys.argv) > 3 else "d5"
ctx   = sys.argv[4] if len(sys.argv) > 4 else "ctx1000"

base = os.path.join(root, shift, d, ctx)
print(f"{base}\n")
print(f"{'model':16s} {'npz':>6s} {'density':>8s} {'point':>7s} {'unreadable':>11s}  example density keys")
print("-" * 100)
for m in sorted(os.listdir(base)):
    cell = os.path.join(base, m)
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
                sample = ",".join(sorted(k for k in keys
                                         if k.startswith(("p_y", "p_joint", "edges", "true_cate"))))[:46]
        else:
            npoint += 1
    print(f"{m:16s} {len(files):6d} {ndens:8d} {npoint:7d} {nbad:11d}  {sample}")
