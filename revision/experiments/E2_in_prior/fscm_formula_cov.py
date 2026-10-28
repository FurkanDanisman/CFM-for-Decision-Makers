"""Coverage of the fixed-SCM Do-PFN 1D models from the formula, with independent noise and rho = 0.
python fscm_formula_cov.py [DIR]      (DIR = revision/results/d262t1/fscm_moments)
Intervals (95%, Gaussian): y_t: mu_t +- 1.96 sd_t; ITE: (mu1 - mu0) +- 1.96 sqrt(sd1^2 + sd0^2), the variance
formula with rho = 0. Truth with independent noise: y1, y0_ind and ITE y1 - y0_ind.
Per model: coverage per observational set (mean over its 500 queries), then mean and spread over the 1000 sets."""
import os, sys
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../results/d262t1/fscm_moments")
T = np.load(os.path.join(D, "truths.npz")); y0, y1 = T["y0_ind"], T["y1"]; ite = y1 - y0
Z = 1.959963984540054
print(f"{'model':14s} {'target':6s} {'mean':>7s} {'sd':>6s} {'min':>6s} {'max':>6s} {'sets in [0.93,0.97]':>20s}")
for m in ("fscm_natural", "fscm_setup1", "fscm_setup2", "fscm_setup3"):
    f = os.path.join(D, f"{m}.npz")
    if not os.path.exists(f):
        print(f"{m:14s} (missing)"); continue
    z = np.load(f)
    rows = {"y0": (z["mu0"], z["sd0"], y0), "y1": (z["mu1"], z["sd1"], y1),
            "ITE": (z["mu1"] - z["mu0"], np.sqrt(z["sd1"] ** 2 + z["sd0"] ** 2), ite)}
    for name, (mu, sd, y) in rows.items():
        c = (np.abs(y - mu) <= Z * sd).mean(1)                     # per set
        print(f"{m:14s} {name:6s} {c.mean():7.3f} {c.std():6.3f} {c.min():6.3f} {c.max():6.3f} {int(((c >= .93) & (c <= .97)).sum()):14d}/1000")
