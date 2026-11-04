"""In-prior point estimation, 1D vs 2D on the SAME datasets: error of the predicted Y0, Y1 and ITE on R = 1000 datasets
(one new SCM each, 500 queries) drawn from the 2D model's training prior, truths with shared noise.
python inprior_pehe.py DATA RES RHO_CSV OUT_TXT CROSS
  DATA  = $SCRATCH/e2/new/data            truths: <prior>/truths_y/r###.npz (y0, y1), <prior>/truths/r###.npz (delta_shared)
  RES   = $SCRATCH/e2/new/results_y       newworld_y.sh eval: every model on its own prior's data
  CROSS = $SCRATCH/e2/new/results_y_cross cross_1d.sh eval: the UWYK and CausalPFN-C 1D models on their 2D model's data
Predictions: <dir>/<model>/outall_shard*.npz, the per-query means mu0, mu1 of each predicted outcome (2D: the means of
the two marginals of the joint). ITE point estimate: mu1 - mu0. Every model is scored against the same truth.
Per dataset: RMSE over its queries divided by sigma_Y, the sd of that dataset's observed outcomes Y_train. The Y RMSEs are against the realised outcomes and so include the outcome noise,
which is identical for the two models of a pair (same datasets, same queries). Per family: mean RMSE (SE) of each model,
and paired tests, overall and by the TRUE correlation of each dataset's SCM (estimated from its truths, see rho_true)."""
import csv, glob, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SHARD                                   # datasets per shard of the harness input

DATA, RES, RHO_CSV, OUT, CROSS = sys.argv[1:6]
NW = 1000
CD = lambda prior: os.path.join(CROSS, f"on_{prior}")
FAMILIES = [  # (family, data prior, [(baseline label, dir, model)], (2D dir, 2D model))
    ("Do-PFN", "dopfn", [("1D", RES, "dopfn_native"), ("matched resolution", CD("dopfn"), "dopfn1d_mr"),
                         ("matched label", CD("dopfn"), "dopfn1d_ml")], (RES, "dopfn_joint2d_262k")),
    ("UWYK", "uwyk_2d", [("1D", CD("uwyk_2d"), "uwyk1d"), ("matched resolution", CD("uwyk_2d"), "uwyk1d_mr"),
                         ("matched label", CD("uwyk_2d"), "uwyk1d_ml")], (RES, "graph2d")),
    ("CausalPFN-C", "cpfn_2d", [("1D", CD("cpfn_2d"), "cpfn1d"), ("matched resolution", CD("cpfn_2d"), "cpfn1d_mr"),
                                ("matched label", CD("cpfn_2d"), "cpfn1d_ml")], (RES, "cpfn2d"))]
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

def rho_true(prior, r):
    """correlation of the two outcomes' noise in the dataset's SCM, from its truths (500 queries):
    Cov_noise = Cov(y1, y0) - Cov(y1, y0_ind), Var_noise(y0) = Var(y0) - Cov(y0, y0_ind),
    Var_noise(y1) = Var(y1) - Cov(y1, y1_ind); the covariate-driven part is common to both terms and cancels."""
    t = np.load(os.path.join(DATA, prior, "truths_y", f"r{r:03d}.npz"))
    y0, y1, y0i, y1i = (t[k].astype(np.float64) for k in ("y0", "y1", "y0_ind", "y1_ind"))
    c = lambda a, b: float(np.cov(a, b)[0, 1])
    v0, v1 = c(y0, y0) - c(y0, y0i), c(y1, y1) - c(y1, y1i)
    if v0 <= 1e-12 or v1 <= 1e-12:
        return np.nan                                  # no noise in one outcome: correlation undefined
    return float(np.clip((c(y1, y0) - c(y1, y0i)) / np.sqrt(v0 * v1), -1, 1))

def score(prior, preds):
    rows = {}
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    for r, (mu0, mu1) in preds.items():
        ty = np.load(os.path.join(DATA, prior, "truths_y", f"r{r:03d}.npz"))
        d = np.load(os.path.join(DATA, prior, "truths", f"r{r:03d}.npz"))["delta_shared"]
        sy = float(np.std(np.load(os.path.join(DATA, prior, f"shard{r // SHARD}", "complexmech", "5node", "path_TY",
                                               "hide_0.0", f"r{r:03d}.npz"))["Y_train"]))   # sd of the observed outcomes
        di = np.load(os.path.join(DATA, prior, "truths", f"r{r:03d}.npz"))["delta_indep"]
        rows[r] = (rmse(mu0, ty["y0"]) / sy, rmse(mu1, ty["y1"]) / sy, rmse(mu1 - mu0, d) / sy,          # shared noise
                   rmse(mu0, ty["y0_ind"]) / sy, rmse(mu1, ty["y1_ind"]) / sy, rmse(mu1 - mu0, di) / sy)  # independent
    return rows

rho_all = {}
for row in csv.DictReader(open(RHO_CSV)):
    rho_all.setdefault(row["model"], {})[int(row["shard"]) * 100 + int(row["dataset"])] = float(row["rho_median"])

say("In-prior point estimation, 1D vs 2D on the same datasets (R = 1000, one new SCM each; shared-noise truths)")
say("RMSE per dataset over its queries, divided by that dataset's sigma_Y (sd of its observed outcomes), as in the paper;")
say("Y0, Y1: predicted mean vs realised outcome; ITE: mu1 - mu0 vs the realised ITE; truths with shared noise (y0, y1,")
say("delta_shared) and with independent noise in the two outcomes (y0_ind, y1_ind, delta_indep)")
from scipy import stats
def holm_(ps):
    order = np.argsort(ps); rej = np.zeros(len(ps), bool)
    for k, i in enumerate(order):
        if ps[i] <= 0.05 / (len(ps) - k): rej[i] = True
        else: break
    return rej
BINS = [(-1, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.01)]
for fam, prior, bases, (d2, m2) in FAMILIES:
    say(); say(f"== {fam}  (data: {prior}; 2D = {m2}; baselines: " + ", ".join(f"{l} = {m}" for l, _, m in bases) + ")")
    p2 = load(d2, m2)
    if not p2:
        say(f"   missing 2D predictions ({d2}/{m2})"); continue
    have = [(l, load(d, m)) for l, d, m in bases]
    for l, p in have:
        if not p: say(f"   missing predictions for {l}")
    have = [(l, p) for l, p in have if p]
    common = sorted(set(p2).intersection(*[set(p) for _, p in have]))
    S2 = np.array([score(prior, {r: p2[r]})[r] for r in common])
    SB = {l: np.array([score(prior, {r: p[r]})[r] for r in common]) for l, p in have}
    rr = np.array([rho_true(prior, r) for r in common])
    say(f"   {len(common)} datasets; true correlation of the SCM: median {np.nanmedian(rr):.2f}, 5-95% "
        f"{np.nanpercentile(rr, 5):.2f} to {np.nanpercentile(rr, 95):.2f}, undefined for {int(np.isnan(rr).sum())}")
    masks = [("all", np.ones(len(common), bool))] + [(f"[{max(lo, -1):.1f}, {min(hi, 1):.1f})", (rr >= lo) & (rr < hi)) for lo, hi in BINS]
    models = [(l, SB[l]) for l, _ in have] + [("2D", S2)]
    for c, name in enumerate(("Y0 shared", "Y1 shared", "ITE shared", "Y0 indep", "Y1 indep", "ITE indep")):
        say(f"   {name}: error / sigma_Y by bin of the TRUE correlation: mean (SE) [median]")
        say(f"   {'bin':14s} {'n':>5s}  " + "  ".join(f"{l:>26s}" for l, _ in models))
        for bl, m in masks:
            if m.sum() < 2:
                say(f"   {bl:14s} {int(m.sum()):5d}  (too few)"); continue
            say(f"   {bl:14s} {int(m.sum()):5d}  " + "  ".join(
                f"{M[m, c].mean():8.4f} ({se(M[m, c]):.4f}) [{np.median(M[m, c]):.4f}]".rjust(26) for _, M in models))
    T = []
    for l, M in SB.items():
        for bl, m in masks:
            if m.sum() < 10: continue
            for c, name in enumerate(("Y0 shared", "Y1 shared", "ITE shared", "Y0 indep", "Y1 indep", "ITE indep")):
                a1, a2 = M[m, c], S2[m, c]; lr = np.log(a2 / a1)
                T.append((l, bl, name, int(m.sum()), float(np.exp(lr.mean())), stats.wilcoxon(a2, a1).pvalue,
                          stats.ttest_1samp(lr, 0.0).pvalue))
    rw, rt = holm_(np.array([t[5] for t in T])), holm_(np.array([t[6] for t in T]))
    say(f"   paired tests, 2D vs each baseline on the same datasets (Wilcoxon on the errors; t-test on log(2D/baseline);")
    say(f"   Holm over the {len(T)} tests of this family, alpha 0.05)")
    say(f"   {'baseline':20s} {'bin':14s} {'outcome':11s} {'n':>5s}  {'geo-mean 2D/base':>16s}  {'Wilcoxon p':>11s}  {'t-test p':>9s}  result")
    for (l, bl, name, n, g, pw, pt), a_, b_ in zip(T, rw, rt):
        res = ("2D better" if g < 1 else "2D worse") if (a_ and b_) else (("2D better" if g < 1 else "2D worse") + " (one test)") if (a_ or b_) else "not significant"
        say(f"   {l:20s} {bl:14s} {name:11s} {n:5d}  {g:16.3f}  {pw:11.2g}  {pt:9.2g}  {res}")
open(OUT, "w").write("\n".join(lines) + "\n")
print("wrote", OUT)
