"""In-prior point estimation, 1D vs 2D on the SAME datasets: error of the predicted Y0, Y1 and ITE on R = 1000 datasets
(one new SCM each, 500 queries) drawn from the 2D model's training prior, truths with shared noise.
python inprior_pehe.py DATA RES RHO_CSV OUT_TXT CROSS
  DATA  = $SCRATCH/e2/new/data            truths: <prior>/truths_y/r###.npz (y0, y1), <prior>/truths/r###.npz (delta_shared)
  RES   = $SCRATCH/e2/new/results_y       newworld_y.sh eval: every model on its own prior's data
  CROSS = $SCRATCH/e2/new/results_y_cross cross_1d.sh eval: the UWYK and CausalPFN-C 1D models on their 2D model's data
Predictions: <dir>/<model>/outall_shard*.npz, the per-query means mu0, mu1 of each predicted outcome (2D: the means of
the two marginals of the joint). ITE point estimate: mu1 - mu0. Every model is scored against the same truth.
Per dataset: RMSE over its queries. The Y RMSEs are against the realised outcomes and so include the outcome noise,
which is identical for the two models of a pair (same datasets, same queries). Per family: mean RMSE (SE) of each model,
and the paired ratio 2D / 1D per dataset, overall and by the 2D model's predicted correlation on that dataset."""
import csv, glob, os, re, sys
import numpy as np

DATA, RES, RHO_CSV, OUT, CROSS = sys.argv[1:6]
NW = 1000
FAMILIES = [  # (family, data prior, (1D dir, 1D model), (2D dir, 2D model), name of the 2D model in RHO_CSV)
    ("Do-PFN", "dopfn", (RES, "dopfn_native"), (RES, "dopfn_joint2d_262k"), "Do-PFN 2D"),
    ("UWYK", "uwyk_2d", (os.path.join(CROSS, "on_uwyk_2d"), "uwyk1d"), (RES, "graph2d"), "UWYK 2D"),
    ("CausalPFN-C", "cpfn_2d", (os.path.join(CROSS, "on_cpfn_2d"), "cpfn1d"), (RES, "cpfn2d"), "CausalPFN-C 2D")]
lines = []
say = lambda s="": (print(s), lines.append(s))
se = lambda a: np.std(a, ddof=1) / np.sqrt(len(a))

def load(d, model):
    out = {}
    for f in glob.glob(os.path.join(d, model, "outall_shard*.npz")):
        k = int(re.search(r"shard(\d+)\.npz$", f).group(1)); z = np.load(f)
        for j, i in enumerate(z["idx"]):
            r = k * 100 + int(i)
            if r < NW:
                out[r] = (z["mu0"][j].astype(np.float64), z["mu1"][j].astype(np.float64))
    return out

def score(prior, preds):
    rows = {}
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    for r, (mu0, mu1) in preds.items():
        ty = np.load(os.path.join(DATA, prior, "truths_y", f"r{r:03d}.npz"))
        d = np.load(os.path.join(DATA, prior, "truths", f"r{r:03d}.npz"))["delta_shared"]
        rows[r] = (rmse(mu0, ty["y0"]), rmse(mu1, ty["y1"]), rmse(mu1 - mu0, d))
    return rows

rho_all = {}
for row in csv.DictReader(open(RHO_CSV)):
    rho_all.setdefault(row["model"], {})[int(row["shard"]) * 100 + int(row["dataset"])] = float(row["rho_median"])

say("In-prior point estimation, 1D vs 2D on the same datasets (R = 1000, one new SCM each; shared-noise truths)")
say("RMSE per dataset over its queries; Y0, Y1: predicted mean vs realised outcome; ITE: mu1 - mu0 vs delta_shared")
for fam, prior, (d1, m1), (d2, m2), rho_name in FAMILIES:
    p1, p2 = load(d1, m1), load(d2, m2)
    say(); say(f"== {fam}  (data: {prior}; 1D = {m1}, 2D = {m2})")
    if not p1 or not p2:
        say(f"   missing predictions: 1D {len(p1)} datasets ({d1}/{m1}), 2D {len(p2)} datasets ({d2}/{m2})"); continue
    common = sorted(set(p1) & set(p2))
    A = np.array([score(prior, {r: p1[r]})[r] for r in common]); B = np.array([score(prior, {r: p2[r]})[r] for r in common])
    say(f"   {'model':8s} {'n':>5s}  {'Y0':>16s}  {'Y1':>16s}  {'ITE':>16s}")
    for lab, M in (("1D", A), ("2D", B)):
        say(f"   {lab:8s} {len(M):5d}  " + "  ".join(f"{M[:, c].mean():8.4f} ({se(M[:, c]):.4f})" for c in range(3)))
    say(f"   RMSE per bin of the 2D model's predicted correlation: mean (SE) and [median] over datasets")
    say(f"   {'datasets':30s} {'n':>5s}  {'Y0 1D':>24s} {'Y0 2D':>24s}  {'Y1 1D':>24s} {'Y1 2D':>24s}  {'ITE 1D':>24s} {'ITE 2D':>24s}")
    rr = np.array([rho_all.get(rho_name, {}).get(r, np.nan) for r in common])
    def row(label, m):
        if m.sum() < 2:
            say(f"   {label:30s} {int(m.sum()):5d}  (too few)"); return
        f = lambda M, c: f"{M[m, c].mean():9.4f} ({se(M[m, c]):.4f}) [{np.median(M[m, c]):.4f}]"
        say(f"   {label:30s} {int(m.sum()):5d}  " + "  ".join(f"{f(A, c):>24s} {f(B, c):>24s}" for c in range(3)))
    row("all", np.ones(len(common), bool))
    for lo, hi in [(-1, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.01)]:
        row(f"correlation in [{max(lo, -1):.1f}, {min(hi, 1):.1f})", (rr >= lo) & (rr < hi))
    say(f"   (datasets without a predicted correlation: {int(np.isnan(rr).sum())})")
open(OUT, "w").write("\n".join(lines) + "\n")
print("wrote", OUT)
