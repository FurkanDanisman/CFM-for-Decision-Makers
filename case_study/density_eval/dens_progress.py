"""Fast progress check for the density backfill -- file counts, no npz opened.

dens_sweep.py opens every npz to classify it, which is right for auditing but
too slow to poll while 18 jobs are writing (~86k files). This only counts
names, so it returns in seconds and is safe to run repeatedly mid-run.

Where the density artifact lives differs by model:
    cpfn1d / cpfn2d          <case>/<CASE>_r###.npz        (density inline)
    everything else (tauC)   <case>/predictions/<CASE>_r###.npz

    python dens_progress.py $RES --ctx 1000
    python dens_progress.py $RES --ctx 1000 --shifts shift0
"""
import argparse, os, glob

SHIFTS = ["shift0", "shift+2", "shift-2"]
DS = [2, 3, 5, 10, 20, 30, 40, 50]
MODELS = ["cpfn1d", "cpfn2d", "graph2d", "uwyk", "uwyk_v3a", "uwyk_noanc",
          "dopfn_native", "dopfn_bb"]
CASES = ["Observed_Confounder", "Backdoor_Criterion", "Observed_Mediator",
         "Observed_Mediator_and_Confounder", "Unobserved_Confounder",
         "Frontdoor_Criterion"]
CPFN = ("cpfn1d", "cpfn2d")

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--ctx", type=int, default=1000)
ap.add_argument("--shifts", nargs="*", default=SHIFTS)
ap.add_argument("--ds", nargs="*", type=int, default=DS)
ap.add_argument("--models", nargs="*", default=MODELS)
ap.add_argument("--cases", nargs="*", default=CASES)
ap.add_argument("--n-real", type=int, default=100)
a = ap.parse_args()

per_cell = len(a.cases) * a.n_real
print(f"root={a.root}  ctx={a.ctx}   cell = dumps / {per_cell} "
      f"({len(a.cases)} cases x {a.n_real} realizations)\n")

w = max(len(m) for m in a.models) + 1
grand = [0, 0]
for s in a.shifts:
    print(f"=== {s} ===")
    print(f"{'model':{w}s}" + "".join(f"{'d'+str(d):>11s}" for d in a.ds)
          + f"{'TOTAL':>11s}")
    print("-" * (w + 11 * (len(a.ds) + 1)))
    for m in a.models:
        line, tot = f"{m:{w}s}", 0
        for d in a.ds:
            n = 0
            for c in a.cases:
                cell = os.path.join(a.root, s, f"d{d}", f"ctx{a.ctx}", m, c)
                pat = (os.path.join(cell, f"{c}_r*.npz") if m in CPFN
                       else os.path.join(cell, "predictions", f"{c}_r*.npz"))
                n += len(glob.glob(pat))
            tot += n
            pct = 100.0 * n / per_cell
            line += f"{f'{pct:.0f}%':>11s}"
        line += f"{f'{100.0*tot/(per_cell*len(a.ds)):.0f}%':>11s}"
        print(line)
        grand[0] += tot
        grand[1] += per_cell * len(a.ds)
    print()
print(f"OVERALL: {grand[0]} / {grand[1]} dumps "
      f"({100.0*grand[0]/grand[1]:.1f}%)")
