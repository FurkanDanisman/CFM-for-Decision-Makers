"""In-prior: does 2D's PEHE gain grow with the noise that cancels in the ITE?
Per dataset (500 queries), the cancelled noise
  C = (Var(delta_indep) - Var(delta_shared)) / sigma_Y^2,
how much the variance of the ITE Y1 - Y0 shrinks because the two outcomes share their noise (same units, same x;
the two truths differ only in the noise). PEHE = RMSE(mu1 - mu0, delta_shared) / sigma_Y as in inprior_pehe.py.
Per family and baseline: Spearman correlation of C with the PEHE difference 2D - baseline (negative: the gain grows
with C), Holm over the baselines of the family; and the paired PEHE by fifths of C (mean diff, 95% CI, paired t-test).
python inprior_noise_dose.py DATA RES CROSS OUT_TXT   (same DATA / RES / CROSS as inprior_pehe.py)"""
import glob, os, re, sys
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SHARD

DATA, RES, CROSS, OUT = sys.argv[1:5]
NW = 1000
CD = lambda prior: os.path.join(CROSS, f"on_{prior}")
FAMILIES = [
    ("Do-PFN", "dopfn", [("1D", RES, "dopfn_native"), ("matched resolution", CD("dopfn"), "dopfn1d_mr"),
                         ("matched label", CD("dopfn"), "dopfn1d_ml")], (RES, "dopfn_joint2d_262k")),
    ("UWYK", "uwyk_2d", [("1D", CD("uwyk_2d"), "uwyk1d"), ("matched resolution", CD("uwyk_2d"), "uwyk1d_mr"),
                         ("matched label", CD("uwyk_2d"), "uwyk1d_ml")], (RES, "graph2d")),
    ("CausalPFN-C", "cpfn_2d", [("1D", CD("cpfn_2d"), "cpfn1d"), ("matched resolution", CD("cpfn_2d"), "cpfn1d_mr"),
                                ("matched label", CD("cpfn_2d"), "cpfn1d_ml")], (RES, "cpfn2d"))]
lines = []
say = lambda s="": (print(s), lines.append(s))


def load(d, model):
    out = {}
    for f in glob.glob(os.path.join(d, model, "outall_shard*.npz")):
        k = int(re.search(r"shard(\d+)\.npz$", f).group(1)); z = np.load(f)
        for j, i in enumerate(z["idx"]):
            r = k * 100 + int(i)
            if r < NW:
                out[r] = z["mu1"][j].astype(np.float64) - z["mu0"][j].astype(np.float64)
    return out


def truth(prior, r):
    t = np.load(os.path.join(DATA, prior, "truths", f"r{r:03d}.npz"))
    sy = float(np.std(np.load(os.path.join(DATA, prior, f"shard{r // SHARD}", "complexmech", "5node", "path_TY",
                                           "hide_0.0", f"r{r:03d}.npz"))["Y_train"]))
    ds, di = t["delta_shared"].astype(np.float64), t["delta_indep"].astype(np.float64)
    return ds, sy, (np.var(di) - np.var(ds)) / sy ** 2


def holm(ps):
    order = np.argsort(ps); rej = np.zeros(len(ps), bool)
    for k, i in enumerate(order):
        if ps[i] <= 0.05 / (len(ps) - k): rej[i] = True
        else: break
    return rej


say("In-prior: PEHE gain of 2D vs the noise that cancels in the ITE, C = (Var(delta_indep) - Var(delta_shared)) / sigma_Y^2")
say("PEHE = RMSE(mu1 - mu0, delta_shared) / sigma_Y per dataset; diff = 2D - baseline (negative: 2D better)")
for fam, prior, bases, (d2, m2) in FAMILIES:
    say(); say(f"== {fam}  (data: {prior}; 2D = {m2})")
    p2 = load(d2, m2)
    have = [(l, load(d, m)) for l, d, m in bases]
    have = [(l, p) for l, p in have if p]
    if not p2 or not have:
        say("   missing predictions"); continue
    common = sorted(set(p2).intersection(*[set(p) for _, p in have]))
    T = {r: truth(prior, r) for r in common}
    C = np.array([T[r][2] for r in common])
    pe = lambda p: np.array([np.sqrt(np.mean((p[r] - T[r][0]) ** 2)) / T[r][1] for r in common])
    P2 = pe(p2)
    say(f"   {len(common)} datasets; C: median {np.median(C):.3f}, 5-95% {np.percentile(C, 5):.3f} to "
        f"{np.percentile(C, 95):.3f}, share with C > 0.01: {np.mean(C > 0.01):.2f}")
    fifths = np.array_split(np.argsort(C, kind="stable"), 5)
    rows, sp = [], []
    for l, p in have:
        PB = pe(p); dlt = P2 - PB
        rho, prho = stats.spearmanr(C, dlt); sp.append((l, rho, prho))
        for k, ix in enumerate(fifths):
            d = dlt[ix]; m = d.mean(); h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            rows.append((l, k + 1, len(ix), C[ix].min(), C[ix].max(), PB[ix].mean(), P2[ix].mean(), m, m - h, m + h,
                         stats.ttest_rel(P2[ix], PB[ix]).pvalue))
    rej = holm(np.array([s[2] for s in sp]))
    say("   Spearman correlation of C with the PEHE difference 2D - baseline (Holm over the baselines)")
    for (l, rho, p), rj in zip(sp, rej):
        res = ("gain grows with C" if rho < 0 else "gain shrinks with C") if rj else "not significant"
        say(f"   {l:20s} rho {rho:7.3f}  p {p:9.2g}  {res}")
    say("   PEHE by fifths of C: mean baseline, mean 2D, mean diff 2D - baseline [95% CI], paired t-test p")
    say(f"   {'baseline':20s} {'fifth':>5s} {'n':>4s} {'C range':>19s} {'baseline':>9s} {'2D':>9s} {'diff':>9s} {'95% CI':>21s} {'p':>9s}")
    for l, k, n, lo, hi, mb, m2_, d, a, b, p in rows:
        say(f"   {l:20s} {k:5d} {n:4d} {f'[{lo:.3f}, {hi:.3f}]':>19s} {mb:9.4f} {m2_:9.4f} {d:9.4f} "
            f"{f'[{a:.4f}, {b:.4f}]':>21s} {p:9.2g}")
open(OUT, "w").write("\n".join(lines) + "\n")
print("wrote", OUT)
