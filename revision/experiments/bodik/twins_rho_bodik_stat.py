"""Justification of the Twins jittering: the cross-world correlation statistic of Bodik, Huang, Yu (2025, App. A.3.2)
-- Pearson correlation of the twins' outcomes (Y(0), Y(1)) within the most frequent value combinations of two
covariates (the three most frequent per pair of X1, X2, X3), plus the whole sample -- on the binary Twins outcomes and
on the jittered ones (make_worlds._corr_jitter: the jitter of covariate set d is calibrated on the first d
covariates; d = 10 and 1, w = 0.1, 0.25, 0.5), on the same families. 95% bootstrap interval of the binary value (2000
resamples of the group's families). The range row: min-max over the groups with at least 100 deaths.
python twins_rho_bodik_stat.py OUT_PREFIX     -> OUT_PREFIX.csv, OUT_PREFIX.md"""
import csv
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_worlds as MW                                          # noqa: E402

MIN_DEATHS = 100
z = np.load(os.path.join(HERE, "bodik_twins.npz"))
Xa, y0, y1 = z["X"].astype(float), z["Y0"].astype(float), z["Y1"].astype(float)
rng = np.random.default_rng(0)
groups = [("whole sample", np.arange(y0.size))]
for i, j in ((0, 1), (0, 2), (1, 2)):
    for (a, b), n in Counter(zip(Xa[:, i], Xa[:, j])).most_common(3):
        groups.append((f"X{i + 1}={a:g}, X{j + 1}={b:g}", np.flatnonzero((Xa[:, i] == a) & (Xa[:, j] == b))))


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def boot(s):
    v = np.array([corr(y0[k], y1[k]) for k in (rng.choice(s, s.size) for _ in range(2000))])
    v = v[~np.isnan(v)]
    return (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) if v.size > 100 else (float("nan"),) * 2


cols = [(d, w) for d in (10, 1) for w in MW.DEQ_W]
jit = {}
for d, w in cols:
    u0, u1, c, tgt = MW._corr_jitter(Xa[:, :d], y0, y1, w, [MW.SEED, 9, d, int(round(w * 1000))])
    jit[(d, w)] = (y0 + u0, y1 + u1)
names = ["binary"] + [f"jittered d={d} w={w}" for d, w in cols]
recs = []
for name, s in groups:
    lo, hi = boot(s)
    vals = [corr(y0[s], y1[s])] + [corr(jit[k][0][s], jit[k][1][s]) for k in cols]
    recs.append(dict(group=name, families=int(s.size), deaths=int(y0[s].sum() + y1[s].sum()), ci_lo=lo, ci_hi=hi,
                     **dict(zip(names, vals)),
                     inside_ci=(all(lo <= v <= hi for v in vals[1:]) if not np.isnan(lo) else "")))
ok = [r for r in recs if r["deaths"] >= MIN_DEATHS]
rng_row = {n: (min(r[n] for r in ok), max(r[n] for r in ok)) for n in names}
out = sys.argv[1]
with open(out + ".csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(recs[0])); wr.writeheader(); wr.writerows(recs)
    wr.writerow(dict(group=f"range (groups with >= {MIN_DEATHS} deaths)",
                     **{n: f"{a:.3f}-{b:.3f}" for n, (a, b) in rng_row.items()}))
f = lambda v: "--" if v != v else f"{v:.3f}"
with open(out + ".md", "w") as fh:
    fh.write("# Twins: Bodik et al. (2025) cross-world correlation statistic, binary vs jittered\n\n"
             + __doc__.split("python")[0].strip() + "\n\n")
    fh.write("| Group | Families | Deaths | Binary (95% CI) | " + " | ".join(n.replace("jittered ", "") for n in names[1:])
             + " | All jittered inside CI |\n|" + "---|" * (len(names) + 4) + "\n")
    for r in recs:
        fh.write(f"| {r['group']} | {r['families']} | {r['deaths']} | {f(r['binary'])} ({f(r['ci_lo'])}–{f(r['ci_hi'])}) | "
                 + " | ".join(f(r[n]) for n in names[1:]) + f" | {r['inside_ci']} |\n")
    fh.write(f"| **Range** (groups with ≥ {MIN_DEATHS} deaths) | | | " + " | ".join(
        f"{a:.3f}–{b:.3f}" for a, b in rng_row.values()) + " | |\n")
print(open(out + ".md").read())
