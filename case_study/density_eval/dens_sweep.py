"""Density-file census across the whole (shift, d) grid at one context.

dens_inventory.py inspects ONE cell in detail. This walks every (shift, d,
model) at a fixed context and reports how many npz carry density keys, so a
conclusion drawn from a single cell can be checked against the grid.

    python dens_sweep.py $RES --ctx 1000
    python dens_sweep.py $RES --ctx 1000 --shifts shift0 --models graph2d

Counts every *.npz recursively under <model>/<case>/, opening each file's key
list only (npz is a zip; numpy reads the namelist without decompressing).
"""
import argparse, os, glob
import numpy as np

DENSITY_KEYS = ("p_joint_scaled", "p_y0_scaled")
SHIFTS = ["shift0", "shift+2", "shift-2"]
DS = [2, 3, 5, 10, 20, 30, 40, 50]
MODELS = ["cpfn1d", "cpfn2d", "graph2d", "uwyk", "uwyk_v3a", "uwyk_noanc",
          "dopfn_native", "dopfn_bb"]

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--ctx", type=int, default=1000)
ap.add_argument("--shifts", nargs="*", default=SHIFTS)
ap.add_argument("--ds", nargs="*", type=int, default=DS)
ap.add_argument("--models", nargs="*", default=MODELS)
a = ap.parse_args()

print(f"root={a.root}  ctx={a.ctx}")
print("cells are  <density npz> / <all npz>   (expected 600 per cell: "
      "6 cases x 100 realizations)\n")

w = max(len(m) for m in a.models) + 1
for s in a.shifts:
    print(f"=== {s} ===")
    print(f"{'model':{w}s}" + "".join(f"{'d'+str(d):>14s}" for d in a.ds))
    print("-" * (w + 14 * len(a.ds)))
    for m in a.models:
        line = f"{m:{w}s}"
        for d in a.ds:
            mdir = os.path.join(a.root, s, f"d{d}", f"ctx{a.ctx}", m)
            if not os.path.isdir(mdir):
                line += f"{'-':>14s}"
                continue
            nd = n = 0
            for f in glob.glob(os.path.join(mdir, "*", "**", "*.npz"),
                               recursive=True):
                n += 1
                try:
                    with np.load(f, allow_pickle=True) as z:
                        if set(z.files) & set(DENSITY_KEYS):
                            nd += 1
                except Exception:
                    pass
            line += f"{f'{nd}/{n}':>14s}"
        print(line, flush=True)
    print()
