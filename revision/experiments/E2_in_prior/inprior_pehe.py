"""In-prior point estimation, 1D vs 2D: error of the predicted Y0, Y1 and ITE on R = 1000 datasets (one new SCM each)
drawn from each model's own training prior, truths with shared noise.
python inprior_pehe.py DATA RES RHO_CSV OUT_TXT     (DATA = $SCRATCH/e2/new/data, RES = $SCRATCH/e2/new/results_y,
                                                    RHO_CSV = R-PFN/revision/results/d262t1/rho_pred_inprior.csv)
Predictions: RES/<model>/outall_shard*.npz from newworld_y.sh eval (per query mean mu0, mu1 of each predicted outcome;
for the 2D models the means of the two marginals of the joint). Point estimate of the ITE: mu1 - mu0.
Truths: truths_y/r###.npz y0, y1 (the prior's own pair, shared noise; CausalPFN-C 1D: its target m0, m1) and
truths/r###.npz delta_shared. Per dataset: RMSE over its 500 queries; reported: mean over datasets (SE).
The Y RMSEs are against the realised outcomes, so they include the outcome noise, which is the same for every model
scored on the same data. Do-PFN 1D and 2D are scored on the same 1000 datasets, so they are also compared per dataset
(paired), overall and by the 2D model's predicted correlation on that dataset (rho_pred_inprior.csv)."""
import csv, glob, os, re, sys
import numpy as np

DATA, RES, RHO_CSV, OUT = sys.argv[1:5]
NW = 1000
MODELS = [("dopfn_native", "Do-PFN", "dopfn"), ("dopfn_joint2d_262k", "Do-PFN 2D", "dopfn"),
          ("uwyk1d", "UWYK", "uwyk_1d"), ("graph2d", "UWYK 2D", "uwyk_2d"),
          ("cpfn1d", "CausalPFN-C", "cpfn_1d"), ("cpfn2d", "CausalPFN-C 2D", "cpfn_2d")]
lines = []
say = lambda s="": (print(s), lines.append(s))

def load(model):
    out = {}
    for f in glob.glob(os.path.join(RES, model, "outall_shard*.npz")):
        k = int(re.search(r"shard(\d+)\.npz$", f).group(1)); z = np.load(f)
        for j, i in enumerate(z["idx"]):
            r = k * 100 + int(i)
            if r < NW:
                out[r] = (z["mu0"][j].astype(np.float64), z["mu1"][j].astype(np.float64))
    return out

per = {}
for model, name, prior in MODELS:
    got = load(model)
    if not got:
        say(f"{name}: no predictions in {RES}/{model}"); continue
    rows = {}
    for r, (mu0, mu1) in got.items():
        ty = np.load(os.path.join(DATA, prior, "truths_y", f"r{r:03d}.npz"))
        td = np.load(os.path.join(DATA, prior, "truths", f"r{r:03d}.npz"))
        y0, y1 = (ty["m0"], ty["m1"]) if prior == "cpfn_1d" else (ty["y0"], ty["y1"])
        d = td["delta_shared"]
        rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
        rows[r] = (rmse(mu0, y0), rmse(mu1, y1), rmse(mu1 - mu0, d))
    per[name] = rows

se = lambda a: np.std(a, ddof=1) / np.sqrt(len(a))
say("In-prior point estimation (R = 1000 datasets, one new SCM each, 500 queries; truths with shared noise)")
say("RMSE per dataset, mean over datasets (SE). Y0, Y1: predicted mean vs realised outcome; ITE: mu1 - mu0 vs delta_shared")
say(f"{'model':16s} {'n':>5s}  {'Y0':>16s}  {'Y1':>16s}  {'ITE':>16s}")
for name, rows in per.items():
    a = np.array(list(rows.values()))
    say(f"{name:16s} {len(a):5d}  " + "  ".join(f"{a[:, c].mean():8.4f} ({se(a[:, c]):.4f})" for c in range(3)))

if "Do-PFN" in per and "Do-PFN 2D" in per:
    common = sorted(set(per["Do-PFN"]) & set(per["Do-PFN 2D"]))
    A = np.array([per["Do-PFN"][r] for r in common]); B = np.array([per["Do-PFN 2D"][r] for r in common])
    rho = {}
    for row in csv.DictReader(open(RHO_CSV)):
        if row["model"] == "Do-PFN 2D":
            rho[int(row["shard"]) * 100 + int(row["dataset"])] = float(row["rho_median"])
    say()
    say(f"Do-PFN 2D vs Do-PFN on the same {len(common)} datasets: ratio of RMSE per dataset (2D / 1D), mean (SE), "
        "and share of datasets where 2D is lower")
    say(f"{'datasets':34s} {'n':>5s}  {'Y0':>20s}  {'Y1':>20s}  {'ITE':>20s}")
    def row(label, m):
        if m.sum() < 2:
            say(f"{label:34s} {int(m.sum()):5d}  (too few)"); return
        q = B[m] / A[m]
        say(f"{label:34s} {int(m.sum()):5d}  " + "  ".join(f"{q[:, c].mean():6.3f} ({se(q[:, c]):.3f}) {100 * (q[:, c] < 1).mean():3.0f}%" for c in range(3)))
    allm = np.ones(len(common), bool); row("all", allm)
    rr = np.array([rho.get(r, np.nan) for r in common])
    for lo, hi in [(-1, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.01)]:
        row(f"predicted correlation in [{lo:.1f}, {min(hi, 1):.1f})", (rr >= lo) & (rr < hi))
    say(f"(datasets without a predicted correlation: {int(np.isnan(rr).sum())})")
open(OUT, "w").write("\n".join(lines) + "\n")
print("wrote", OUT)
