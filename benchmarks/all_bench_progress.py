"""Completion percentages for the three density benchmarks in one view.

Each tree has a different shape, so a single glob will not do:

  case studies  <root>/<shift>/d<K>/ctx<N>/<model>/<case>/*.npz      600/cell
  ComplexMech   <root>/N<ctx>/<model>/CMECH_n<d>_<subset>/*.npz      100/cell
  RealCause     <root>/<model>/<DATASET>/*.npz            100/cell (ACIC 10)

dopfn_bb is counted separately everywhere: it writes ONE summary.npz per cell
plus density_r###.npz, so a plain file count is not comparable to the others.

    python all_bench_progress.py --cs $SCRATCH/cs_dvar_dens \
        --cmech $SCRATCH/cmech_1d2d --rc $SCRATCH/rc_dens_uni
"""
import argparse, glob, os

MODELS = ["dopfn_native", "dopfn_bb", "uwyk1d", "graph2d", "cpfn1d", "cpfn2d_pooled"]
DS = [2, 3, 5, 10, 20, 30, 40, 50]
SHIFTS = ["shift0", "shift+2", "shift-2"]
CMECH_N = [5, 10, 20, 30, 40, 50]
CTX = [50, 100, 250, 500, 1000]
RC = {"IHDP": 100, "ACIC": 10, "CPS": 100, "PSID": 100, "PSID_bal": 100}


def _n(pat):
    return len(glob.glob(pat))


def pct(done, total):
    return f"{100.0*done/total:5.0f}%" if total else "    -"


ap = argparse.ArgumentParser()
ap.add_argument("--cs"); ap.add_argument("--cmech"); ap.add_argument("--rc")
ap.add_argument("--cmech-subset", default="nonzero")
ap.add_argument("--cmech-models", nargs="*", default=["dopfn_native"],
                help="ComplexMech is usually re-run for a subset of models")
a = ap.parse_args()

if a.cs and os.path.isdir(a.cs):
    print(f"\n=== CASE STUDIES  {a.cs}   (600 npz per cell) ===")
    print(f"{'model':16s}" + "".join(f"{s:>10s}" for s in SHIFTS) + f"{'TOTAL':>10s}")
    for m in MODELS:
        row, td, tt = f"{m:16s}", 0, 0
        for s in SHIFTS:
            d_ = sum(_n(f"{a.cs}/{s}/d{d}/ctx1000/{m}/*/*.npz") for d in DS)
            t_ = 600 * len(DS)
            td += d_; tt += t_
            row += f"{pct(d_, t_):>10s}"
        print(row + f"{pct(td, tt):>10s}")

if a.cmech and os.path.isdir(a.cmech):
    print(f"\n=== COMPLEXMECH  {a.cmech}  subset={a.cmech_subset}  (100 per cell) ===")
    print(f"{'model':16s}" + "".join(f"{'N'+str(c):>9s}" for c in CTX) + f"{'TOTAL':>9s}")
    for m in a.cmech_models:
        row, td, tt = f"{m:16s}", 0, 0
        for c in CTX:
            d_ = sum(_n(f"{a.cmech}/N{c}/{m}/CMECH_n{n}_{a.cmech_subset}/*.npz")
                     for n in CMECH_N)
            t_ = 100 * len(CMECH_N)
            td += d_; tt += t_
            row += f"{pct(d_, t_):>9s}"
        print(row + f"{pct(td, tt):>9s}")

if a.rc and os.path.isdir(a.rc):
    print(f"\n=== REALCAUSE  {a.rc} ===")
    print(f"{'model':16s}" + "".join(f"{d:>11s}" for d in RC) + f"{'TOTAL':>9s}")
    for m in MODELS:
        row, td, tt = f"{m:16s}", 0, 0
        for d, exp in RC.items():
            got = _n(f"{a.rc}/{m}/{d}/*.npz")
            td += min(got, exp); tt += exp
            row += f"{f'{got}/{exp}':>11s}"
        print(row + f"{pct(td, tt):>9s}")
print()
