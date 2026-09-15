"""Census of which npz in a sweep cell carry a density, and WHERE they sit.

progress.py opens only the FIRST npz per (model, case) directory, and these
trees hold density dumps in a `predictions/` subdirectory alongside point-eval
npz at the top level -- so a top-level-only scan reports 0 density when plenty
exists. This walks recursively, groups by the subdirectory relative to
<model>/<case>/, and opens every file's key list (cheap -- npz is a zip, numpy
reads the namelist without decompressing).

    python dens_inventory.py $RES [shift0] [d5] [ctx1000]
    python dens_inventory.py $RES shift0 d5 ctx1000 --by-model   # totals only

Layout walked: <root>/<shift>/d<K>/ctx<N>/<model>/<case>/**/*.npz
"""
import os, sys, glob
from collections import defaultdict
import numpy as np

DENSITY_KEYS = ("p_joint_scaled", "p_y0_scaled")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
BY_MODEL = "--by-model" in sys.argv

root  = args[0]
shift = args[1] if len(args) > 1 else "shift0"
d     = args[2] if len(args) > 2 else "d5"
ctx   = args[3] if len(args) > 3 else "ctx1000"

base = os.path.join(root, shift, d, ctx)
print(f"{base}\n")
hdr = f"{'model':16s} {'case':26s} {'subdir':14s} {'npz':>6s} {'dens':>6s} {'point':>6s} {'bad':>4s}"
if BY_MODEL:
    hdr = f"{'model':16s} {'subdir':14s} {'npz':>7s} {'dens':>7s} {'point':>7s} {'bad':>4s}"
print(hdr + "  sample density keys")
print("-" * (len(hdr) + 44))

for m in sorted(os.listdir(base)):
    mdir = os.path.join(base, m)
    if not os.path.isdir(mdir):
        continue
    agg = defaultdict(lambda: [0, 0, 0, 0, ""])   # subdir -> n,dens,point,bad,sample
    for c in sorted(os.listdir(mdir)):
        cell = os.path.join(mdir, c)
        if not os.path.isdir(cell):
            continue
        for f in sorted(glob.glob(os.path.join(cell, "**", "*.npz"), recursive=True)):
            sub = os.path.relpath(os.path.dirname(f), cell)
            sub = "." if sub == os.curdir else sub
            key = (c, sub) if not BY_MODEL else (sub,)
            rec = agg[key]
            rec[0] += 1
            try:
                with np.load(f, allow_pickle=True) as z:
                    keys = set(z.files)
            except Exception:
                rec[3] += 1
                continue
            if keys & set(DENSITY_KEYS):
                rec[1] += 1
                if not rec[4]:
                    rec[4] = ",".join(sorted(
                        k for k in keys
                        if k.startswith(("p_y", "p_joint", "edges", "true_cate"))))[:40]
            else:
                rec[2] += 1
    tot = [0, 0, 0, 0]
    for key in sorted(agg):
        n, nd, np_, nb, sample = agg[key]
        for i, v in enumerate((n, nd, np_, nb)):
            tot[i] += v
        if BY_MODEL:
            print(f"{m:16s} {key[0]:14s} {n:7d} {nd:7d} {np_:7d} {nb:4d}  {sample}")
        else:
            print(f"{m:16s} {key[0]:26s} {key[1]:14s} {n:6d} {nd:6d} {np_:6d} {nb:4d}  {sample}")
    pad = 31 if BY_MODEL else 57
    print(f"{m:16s} {'== TOTAL':{pad}s} {tot[0]:6d} {tot[1]:6d} {tot[2]:6d} {tot[3]:4d}")
    print("-" * (len(hdr) + 44))
