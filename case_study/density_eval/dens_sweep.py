"""Density-artifact census across the (shift, d) grid at one context.

TWO dump schemas live in this tree and a scan must know both:

  CausalPFN DENSITY_DUMP (cpfn1d / cpfn2d)
      <model>/<case>/<DATASET>_r###.npz   with p_y0_scaled / p_joint_scaled

  tauC runner (graph2d / uwyk* / dopfn_*), eval_density_tauC.py
      <model>/<case>/<DATASET>_r###.npz              per-method METRICS
      <model>/<case>/predictions/<DATASET>_r###.npz  tau_grid + *_logits

Testing only for the CausalPFN keys classifies every tauC artifact as
"point" and reports 0% density for six of eight models, which is wrong. The
tauC detector below is density_tauc.is_tauc_prediction verbatim.

    python dens_sweep.py $RES --ctx 1000
    python dens_sweep.py $RES --ctx 1000 --shifts shift0 --models graph2d
"""
import argparse, os, glob
import numpy as np

CPFN_KEYS = ("p_joint_scaled", "p_y0_scaled")
SHIFTS = ["shift0", "shift+2", "shift-2"]
DS = [2, 3, 5, 10, 20, 30, 40, 50]
MODELS = ["cpfn1d", "cpfn2d", "graph2d", "uwyk", "uwyk_v3a", "uwyk_noanc",
          "dopfn_native", "dopfn_bb"]


def summary_realizations(path):
    """dopfn_bb writes ONE summary.npz per case holding per-realization ARRAYS
    (pehe[], ate_pred[], true_ate[]) rather than one file per realization, so
    counting files understates it by ~100x. Return the array length."""
    try:
        with np.load(path, allow_pickle=True) as z:
            for k in ("pehe", "pehe_raw", "ate_pred", "true_ate"):
                if k in z.files:
                    return int(np.asarray(z[k]).reshape(-1).shape[0])
    except Exception:
        pass
    return 0


def classify(path):
    """-> 'cpfn' | 'tauc_pred' | 'tauc_metrics' | 'other' | 'bad'"""
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
    if k & {"nll", "l2", "pehe"} or any(s in k for s in ("nll_joint", "l2_joint")):
        return "tauc_metrics"
    return "other"


ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--ctx", type=int, default=1000)
ap.add_argument("--shifts", nargs="*", default=SHIFTS)
ap.add_argument("--ds", nargs="*", type=int, default=DS)
ap.add_argument("--models", nargs="*", default=MODELS)
a = ap.parse_args()

print(f"root={a.root}  ctx={a.ctx}")
print("cell = <density artifacts> / <all npz>.  density = CausalPFN dump OR")
print("tauC prediction dump. Expected 600 per cell (6 cases x 100 realizations).\n")

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
                if os.path.basename(f) == "summary.npz":
                    # per-realization arrays inside one file
                    n += summary_realizations(f)
                    continue
                n += 1
                if classify(f) in ("cpfn", "tauc_pred"):
                    nd += 1
            line += f"{f'{nd}/{n}':>14s}"
        print(line, flush=True)
    print()
