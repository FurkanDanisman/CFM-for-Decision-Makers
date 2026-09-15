"""Density coverage across the (shift, d) grid at one context.

WHY A PLAIN FILE COUNT IS WRONG. Three artifact schemas live in this tree:

  cpfn1d / cpfn2d      one npz per (case, realization), density keys inside
                       (p_y0_scaled / p_joint_scaled)          -> 600 files

  graph2d / uwyk* / dopfn_*   TWO files per (case, realization):
       <case>/<DATASET>_r###.npz              per-method metrics
       <case>/predictions/<DATASET>_r###.npz  tau_grid + *_logits  <- density
                       -> up to 1200 files for the same 600 realizations

  dopfn_bb point runs  one summary.npz per case holding per-realization ARRAYS

So "density files / all files" has a denominator that GROWS as coverage grows.
This counts unique (case, realization) pairs with a density artifact, out of
n_cases x n_realizations -- which is the thing you actually want to know.

    python dens_sweep.py $RES --ctx 1000
    python dens_sweep.py $RES --ctx 1000 --detail        # per-artifact counts
    python dens_sweep.py $RES --ctx 1000 --dupes         # cross-model overlap
"""
import argparse, os, re, glob
from collections import defaultdict
import numpy as np

CPFN_KEYS = ("p_joint_scaled", "p_y0_scaled")
SHIFTS = ["shift0", "shift+2", "shift-2"]
DS = [2, 3, 5, 10, 20, 30, 40, 50]
MODELS = ["cpfn1d", "cpfn2d", "graph2d", "uwyk", "uwyk_v3a", "uwyk_noanc",
          "dopfn_native", "dopfn_bb"]
N_REAL = 100
_R = re.compile(r"_r(\d+)\.npz$|^r(\d+)\.npz$")


def realization_of(path):
    m = _R.search(os.path.basename(path))
    if not m:
        return None
    return int(m.group(1) or m.group(2))


def classify(path):
    """-> 'cpfn' | 'tauc_pred' | 'tauc_metrics' | 'summary' | 'other' | 'bad'"""
    if os.path.basename(path) == "summary.npz":
        return "summary"
    try:
        with np.load(path, allow_pickle=True) as z:
            k = set(z.files)
    except Exception:
        return "bad"
    if k & set(CPFN_KEYS):
        return "cpfn"
    # density_tauc.is_tauc_prediction, verbatim
    if "tau_grid" in k and ("joint_logits" in k or "dopfn_joint_logits" in k):
        return "tauc_pred"
    return "tauc_metrics"


def scan_cell(mdir):
    """-> (covered set of (case, r), Counter of artifact kinds)"""
    covered, kinds = set(), defaultdict(int)
    for f in glob.glob(os.path.join(mdir, "*", "**", "*.npz"), recursive=True):
        case = os.path.relpath(f, mdir).split(os.sep)[0]
        kind = classify(f)
        kinds[kind] += 1
        if kind in ("cpfn", "tauc_pred"):
            r = realization_of(f)
            if r is not None:
                covered.add((case, r))
    return covered, kinds


ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--ctx", type=int, default=1000)
ap.add_argument("--shifts", nargs="*", default=SHIFTS)
ap.add_argument("--ds", nargs="*", type=int, default=DS)
ap.add_argument("--models", nargs="*", default=MODELS)
ap.add_argument("--n-cases", type=int, default=6)
ap.add_argument("--detail", action="store_true",
                help="also print per-artifact-kind file counts")
ap.add_argument("--dupes", action="store_true",
                help="report models whose covered set is identical (one tauC "
                     "run writes several method dirs)")
a = ap.parse_args()

TOTAL = a.n_cases * N_REAL
print(f"root={a.root}  ctx={a.ctx}")
print(f"cell = unique (case, realization) pairs WITH A DENSITY ARTIFACT, "
      f"out of {TOTAL}\n")

w = max(len(m) for m in a.models) + 1
for s in a.shifts:
    print(f"=== {s} ===")
    print(f"{'model':{w}s}" + "".join(f"{'d'+str(d):>12s}" for d in a.ds))
    print("-" * (w + 12 * len(a.ds)))
    cover = {}
    for m in a.models:
        line = f"{m:{w}s}"
        for d in a.ds:
            mdir = os.path.join(a.root, s, f"d{d}", f"ctx{a.ctx}", m)
            if not os.path.isdir(mdir):
                line += f"{'-':>12s}"
                continue
            cv, kinds = scan_cell(mdir)
            cover[(m, d)] = cv
            pct = 100.0 * len(cv) / TOTAL
            line += f"{f'{len(cv)} ({pct:.0f}%)':>12s}"
            if a.detail:
                det = " ".join(f"{k}={v}" for k, v in sorted(kinds.items()))
                print(f"    {m} d{d}: {det}")
        print(line, flush=True)
    if a.dupes:
        print("\n  identical coverage sets (same underlying run):")
        seen = defaultdict(list)
        for (m, d), cv in cover.items():
            if cv:
                seen[(d, frozenset(cv))].append(m)
        for (d, _), ms in sorted(seen.items(), key=lambda x: x[0][0]):
            if len(ms) > 1:
                print(f"    d{d}: {', '.join(sorted(ms))}")
    print()
